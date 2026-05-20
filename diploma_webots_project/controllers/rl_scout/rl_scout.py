import sys
import os
import threading
import tkinter as tk
from tkinter import messagebox
import math
import torch
import numpy as np
from collections import deque
from controller import Supervisor
from dqn_agent import DQNAgent
from scout_env import ScoutEnv

# ======================================================
# CONFIG
# ======================================================

NUM_EPISODES   = 1000
MAX_STEPS      = 1800
EVAL_EPISODES  = 5       # episoade de eval după training
GOAL_KEY       = 'g'     # tasta pentru schimbat goal în eval

# Poziție start pentru eval (configurabilă din popup)
DEFAULT_START  = (0.0, 0.0, 0.0)   # x, y, theta

MODEL_BEST     = "dqn_model_best.pth"
MODEL_FINAL    = "dqn_model.pth"

# ======================================================
# TKINTER HELPERS  (rulează în thread separat față de Webots)
# ======================================================

def _popup_coordinates(title, fields):
    """
    Popup generic Tkinter — returnează dict {label: float} sau None dacă anulat.
    fields = listă de label-uri (ex. ["Goal X", "Goal Y"])
    """
    result = {}
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    root.attributes("-topmost", True)

    entries = {}
    for i, label in enumerate(fields):
        tk.Label(root, text=label + ":").grid(row=i, column=0, padx=10, pady=5, sticky="e")
        e = tk.Entry(root, width=10)
        e.grid(row=i, column=1, padx=10, pady=5)
        e.insert(0, "0.0")
        entries[label] = e

    def on_ok():
        try:
            for label, e in entries.items():
                result[label] = float(e.get())
            root.destroy()
        except ValueError:
            messagebox.showerror("Eroare", "Introduceți valori numerice valide.")

    def on_cancel():
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=10)
    tk.Button(btn_frame, text="OK",     width=8, command=on_ok).pack(side="left",  padx=5)
    tk.Button(btn_frame, text="Anulare", width=8, command=on_cancel).pack(side="right", padx=5)

    root.mainloop()
    return result if result else None


def ask_goal():
    """Popup pentru coordonate goal. Returnează (gx, gy) sau None."""
    data = _popup_coordinates("Setează Goal", ["Goal X", "Goal Y"])
    if data:
        return data["Goal X"], data["Goal Y"]
    return None


def ask_start():
    """Popup pentru start + goal inițial. Returnează (sx, sy, stheta, gx, gy) sau None."""
    data = _popup_coordinates(
        "Configurare Start & Goal",
        ["Start X", "Start Y", "Start θ (rad)", "Goal X", "Goal Y"]
    )
    if data:
        return (data["Start X"], data["Start Y"], data["Start θ (rad)"],
                data["Goal X"],  data["Goal Y"])
    return None


def print_episode_result(prefix):
    dist = np.sqrt((env.goal_x - env.x)**2 + (env.goal_y - env.y)**2)
    if env.done_reason == "goal":
        print(f"[OK] {prefix}: Goal atins! Pos: ({env.x:.2f}, {env.y:.2f})")
    elif env.done_reason == "timeout":
        print(
            f"[TIMEOUT] {prefix}: {env.step_count} pasi. "
            f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanta goal: {dist:.2f} m"
        )
    elif env.done_reason == "collision":
        print(
            f"[COLLISION] {prefix}: proximitate/coliziune LiDAR. "
            f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanta goal: {dist:.2f} m"
        )
    else:
        print(
            f"[DONE] {prefix}: {env.done_reason}. "
            f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanta goal: {dist:.2f} m"
        )


_sanity_last_pos = None
_sanity_stuck_steps = 0
_sanity_recovery_steps = 0
_sanity_recovery_turn = 3


def reset_sanity_controller_memory():
    global _sanity_last_pos, _sanity_stuck_steps, _sanity_recovery_steps, _sanity_recovery_turn
    _sanity_last_pos = None
    _sanity_stuck_steps = 0
    _sanity_recovery_steps = 0
    _sanity_recovery_turn = 3


def select_sanity_action(state):
    global _sanity_last_pos, _sanity_stuck_steps, _sanity_recovery_steps, _sanity_recovery_turn

    front, left, right, front_left, front_right, upper_front, mid_front, \
        height_profile, distance, angle, v, w, z, roll, pitch, yaw_rate = state
    nav_distance, nav_angle, nav_kind = env.get_navigation_error()
    final_distance = math.sqrt((env.goal_x - env.x) ** 2 + (env.goal_y - env.y) ** 2)

    close_to_goal = nav_distance < 0.03
    near_final_goal = nav_kind == "goal" and final_distance < 2.8
    drive_over_traversable = (
        env.should_drive_over_traversable() or
        env.can_attempt_traversable_ahead(state) or
        env.is_on_traversable_terrain()
    )
    unstable_attitude = abs(env.roll) > env.max_safe_roll or abs(env.pitch) > env.max_safe_pitch

    current_pos = (env.x, env.y)
    if _sanity_last_pos is None:
        _sanity_last_pos = current_pos
    moved = math.sqrt((current_pos[0] - _sanity_last_pos[0]) ** 2 + (current_pos[1] - _sanity_last_pos[1]) ** 2)
    _sanity_last_pos = current_pos

    if final_distance > env.goal_tolerance and abs(nav_angle) < 0.75 and moved < 0.002:
        _sanity_stuck_steps += 1
    else:
        _sanity_stuck_steps = max(0, _sanity_stuck_steps - 2)

    if _sanity_recovery_steps > 0:
        _sanity_recovery_steps -= 1
        return -1 if _sanity_recovery_steps > 18 else _sanity_recovery_turn

    if _sanity_stuck_steps > 45:
        _sanity_stuck_steps = 0
        _sanity_recovery_steps = 34
        _sanity_recovery_turn = 3 if left > right else 4
        return -1

    if unstable_attitude:
        return 0 if nav_distance > 0.5 and abs(nav_angle) < 0.35 else (3 if nav_angle > 0 else 4)

    if nav_kind == "ramp_access":
        if nav_angle > 0.35:
            return 3
        if nav_angle < -0.35:
            return 4
        if nav_angle > 0.08:
            return 1
        if nav_angle < -0.08:
            return 2
        return 0

    if drive_over_traversable:
        if nav_angle > 0.45:
            return 3
        if nav_angle < -0.45:
            return 4
        if nav_angle > 0.10:
            return 1
        if nav_angle < -0.10:
            return 2
        return 0

    front_limit = 0.06 if near_final_goal else 0.14
    diagonal_limit = 0.05 if near_final_goal else 0.10
    if not close_to_goal and not drive_over_traversable and (
            front < front_limit or front_left < diagonal_limit or front_right < diagonal_limit):
        return 3 if left > right else 4
    if close_to_goal and front < 0.04:
        return 3 if left > right else 4

    rotate_limit = 0.60 if near_final_goal else 0.45
    steer_limit = 0.18 if near_final_goal else 0.10

    if nav_angle > rotate_limit:
        return 3
    if nav_angle < -rotate_limit:
        return 4
    if nav_angle > steer_limit:
        return 1
    if nav_angle < -steer_limit:
        return 2
    return 0


# ======================================================
# INIT WEBOTS
# ======================================================

robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node        = robot.getFromDef("Pioneer3AT")
translation_field = robot_node.getField("translation")
rotation_field    = robot_node.getField("rotation")

fl = robot.getDevice("front left wheel")
bl = robot.getDevice("back left wheel")
fr = robot.getDevice("front right wheel")
br = robot.getDevice("back right wheel")

motors = [fl, bl, fr, br]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)

def _optional_device(name):
    try:
        device = robot.getDevice(name)
        device.enable(timestep)
        return device
    except Exception:
        print(f"[WARN] Dispozitivul '{name}' nu a fost gasit. Se foloseste fallback unde este posibil.")
        return None

imu = _optional_device("imu")
gps = _optional_device("gps")
gyro = _optional_device("gyro")
accelerometer = _optional_device("accelerometer")

# Marker vizual pentru goal (DEF GOAL_TARGET in world.wbt)
goal_node             = robot.getFromDef("GOAL_TARGET")
goal_translation_field = goal_node.getField("translation") if goal_node else None

if goal_node is None:
    print("[WARN] DEF GOAL_TARGET nu a fost gasit in scena - markerul nu va fi vizibil.")

# Keyboard Webots (pentru tasta G în eval)
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# ======================================================
# ENVIRONMENT & AGENT
# ======================================================

env = ScoutEnv(
    robot, robot_node, motors, lidar,
    imu, gps, gyro, accelerometer,
    translation_field, rotation_field,
    goal_translation_field,
    timestep
)

agent = DQNAgent(env.state_dim, env.action_dim)

# ======================================================
# TRAINING
# ======================================================

def run_training():
    print("=" * 50)
    print("  TRAINING MODE — de la zero")
    print("=" * 50)

    episode_rewards = deque(maxlen=50)
    best_reward     = -float("inf")

    for episode in range(NUM_EPISODES):

        # Start fix (0,0) în training, goal random
        state = env.reset(start_x=0.0, start_y=0.0, random_goal=True)
        total_reward = 0.0

        for step in range(MAX_STEPS):
            action                      = agent.select_action(state)
            next_state, reward, done    = env.step(action)

            if not env.simulation_running:
                return

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            state        = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards)

        if (episode + 1) % 10 == 0:
            print(
                f"Ep {episode+1:4d}/{NUM_EPISODES} | "
                f"Reward: {total_reward:8.2f} | Avg50: {avg_reward:8.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Goal: ({env.goal_x:6.2f}, {env.goal_y:6.2f}) | "
                f"Pos: ({env.x:6.2f}, {env.y:6.2f})"
            )

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_network.state_dict(), MODEL_BEST)

        if (episode + 1) % 100 == 0:
            torch.save(agent.q_network.state_dict(), f"dqn_model_ep{episode+1}.pth")

    torch.save(agent.q_network.state_dict(), MODEL_FINAL)
    print(f"\n✓ Training încheiat. Best reward: {best_reward:.2f}")
    print(f"  Modele salvate: {MODEL_FINAL}, {MODEL_BEST}")


# ======================================================
# EVALUATION
# ======================================================

def run_evaluation():
    print("=" * 50)
    print("  EVALUATION MODE")
    print("=" * 50)

    if not os.path.exists(MODEL_BEST):
        raise FileNotFoundError(f"Nu există {MODEL_BEST}. Rulează training întâi.")

    try:
        agent.q_network.load_state_dict(torch.load(MODEL_BEST))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Modelul salvat {MODEL_BEST} nu mai este compatibil cu noua stare "
            f"({env.state_dim} valori). Ruleaza training de la zero."
        ) from exc
    agent.q_network.eval()
    agent.epsilon = 0.0

    print(f"✓ Model încărcat: {MODEL_BEST}")
    print(f"  Apasă '{GOAL_KEY.upper()}' în simulare pentru a schimba goal-ul.\n")

    # Popup configurare start + goal inițial (din thread separat)
    config = [None]
    def _ask():
        config[0] = ask_start()
    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    t.join()

    if config[0] is None:
        print("[EVAL] Anulat de utilizator.")
        return

    sx, sy, stheta, gx, gy = config[0]

    # Setăm goal-ul înainte de reset
    env.set_goal(gx, gy)
    state = env.reset(start_x=sx, start_y=sy, start_theta=stheta, random_goal=False)

    print(f"  Start: ({sx:.2f}, {sy:.2f}, θ={stheta:.2f} rad)")
    print(f"  Goal:  ({env.goal_x:.2f}, {env.goal_y:.2f})\n")

    # Flag pentru cerere goal nou (din keyboard thread)
    new_goal_request = [False]

    def _listen_goal_key():
        """Thread care ascultă tasta G și deschide popup."""
        while True:
            key = keyboard.getKey()
            if key == ord(GOAL_KEY.upper()) or key == ord(GOAL_KEY.lower()):
                new_goal_request[0] = True

    # Loop evaluare
    while env.simulation_running:

        # Verificare tastă G
        key = keyboard.getKey()
        if key == ord(GOAL_KEY.upper()) or key == ord(GOAL_KEY.lower()):
            new_goal_request[0] = True

        if new_goal_request[0]:
            new_goal_request[0] = False
            coords = [None]
            def _ask_goal():
                coords[0] = ask_goal()
            t = threading.Thread(target=_ask_goal, daemon=True)
            t.start()
            t.join()
            if coords[0] is not None:
                env.set_goal(coords[0][0], coords[0][1])
                print(f"[EVAL] Nou goal setat: ({env.goal_x:.2f}, {env.goal_y:.2f})")

        action = agent.select_action(state)
        state, reward, done = env.step(action)

        if not env.simulation_running:
            break

        if done:
            dist = np.sqrt((env.goal_x - env.x)**2 + (env.goal_y - env.y)**2)
            if env.done_reason == "goal":
                print(f"✅ Goal atins! Pos: ({env.x:.2f}, {env.y:.2f})")
            elif env.done_reason == "timeout":
                print(
                    f"⏱️ Episod încheiat prin timeout ({env.step_count} pași). "
                    f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanță goal: {dist:.2f} m"
                )
            elif env.done_reason == "collision":
                print(
                    f"❌ Episod încheiat prin proximitate/coliziune LiDAR. "
                    f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanță goal: {dist:.2f} m"
                )
            else:
                print(
                    f"❌ Episod încheiat ({env.done_reason}). "
                    f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanță goal: {dist:.2f} m"
                )

            # După terminare episod, popup pentru nou start + goal
            config2 = [None]
            def _ask2():
                config2[0] = ask_start()
            t2 = threading.Thread(target=_ask2, daemon=True)
            t2.start()
            t2.join()

            if config2[0] is None:
                print("[EVAL] Simulare oprită de utilizator.")
                for m in motors:
                    m.setVelocity(0.0)
                break

            sx, sy, stheta, gx, gy = config2[0]
            env.set_goal(gx, gy)
            state = env.reset(start_x=sx, start_y=sy,
                              start_theta=stheta, random_goal=False)
            print(f"  Start: ({sx:.2f}, {sy:.2f}) | Goal: ({env.goal_x:.2f}, {env.goal_y:.2f})")


def run_sanity_check():
    print("=" * 50)
    print("  SANITY CHECK MODE - controller simplu, fara DQN")
    print("=" * 50)

    config = [None]
    def _ask():
        config[0] = ask_start()
    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    t.join()

    if config[0] is None:
        print("[SANITY] Anulat de utilizator.")
        return

    sx, sy, stheta, gx, gy = config[0]
    env.set_goal(gx, gy)
    state = env.reset(start_x=sx, start_y=sy, start_theta=stheta, random_goal=False)
    reset_sanity_controller_memory()

    print(f"  Start: ({sx:.2f}, {sy:.2f}, theta={stheta:.2f} rad)")
    print(f"  Goal:  ({env.goal_x:.2f}, {env.goal_y:.2f})")

    last_nav_signature = None
    while env.simulation_running:
        nav_x, nav_y, nav_kind = env.get_navigation_target()
        nav_signature = (nav_kind, round(nav_x, 2), round(nav_y, 2))
        if nav_signature != last_nav_signature:
            print(f"[NAV] Target activ: {nav_kind} ({nav_x:.2f}, {nav_y:.2f})")
            last_nav_signature = nav_signature

        action = select_sanity_action(state)
        state, reward, done = env.step(action)

        if env.step_count % 50 == 0:
            nav_distance, nav_angle, active_kind = env.get_navigation_error()
            z = robot_node.getPosition()[2]
            print(
                f"[SANITY] pas {env.step_count:4d}: "
                f"pos=({env.x:.2f}, {env.y:.2f}, z={z:.2f}) | "
                f"nav={active_kind} dist={nav_distance:.2f} angle={nav_angle:.2f} | "
                f"roll={env.roll:.2f} pitch={env.pitch:.2f} | "
                f"drive_over={env.should_drive_over_traversable()}"
            )

        if not env.simulation_running:
            break

        if done:
            print_episode_result("Sanity check")
            for m in motors:
                m.setVelocity(0.0)
            break


def run_ramp_physics_test():
    print("=" * 50)
    print("  TEST FIZICA RAMPA - comanda directa inainte")
    print("=" * 50)
    print("  Nu se foloseste DQN si nu se foloseste evitare LiDAR.")

    ramp = env.traversable_regions[0]
    start_x = ramp["x"] + ramp["sx"] / 2.0 + 1.35
    start_y = ramp["y"]
    start_theta = math.pi

    env.set_goal(ramp["x"], ramp["y"])
    env.reset(start_x=start_x, start_y=start_y, start_theta=start_theta, random_goal=False)

    print(
        f"  Start test: ({start_x:.2f}, {start_y:.2f}, theta={start_theta:.2f} rad). "
        "Robotul ar trebui sa urce drept pe rampa."
    )

    max_test_steps = 500
    max_z = robot_node.getPosition()[2]
    for i in range(max_test_steps):
        env._apply_action(0)
        if robot.step(timestep) == -1:
            env.simulation_running = False
            break
        env._update_odometry()
        z = robot_node.getPosition()[2]
        max_z = max(max_z, z)

        if i % 50 == 0:
            print(f"[PHYS] pas {i:3d}: pos=({env.x:.2f}, {env.y:.2f}, z={z:.2f})")

    for m in motors:
        m.setVelocity(0.0)

    z = robot_node.getPosition()[2]
    print(f"[PHYS] final: pos=({env.x:.2f}, {env.y:.2f}, z={z:.2f}) | max_z={max_z:.2f}")
    if max_z > 0.35:
        print("[PHYS] OK: robotul a urcat fizic pe rampa si a coborat. Urmatorul pas este logica de navigatie.")
    else:
        print("[PHYS] FAIL: robotul nu urca nici cu mers drept. Problema este in fizica/geometria rampei.")


# ======================================================
# MAIN
# ======================================================

def ask_mode():
    """Popup la pornire: Training / evaluare / sanity check."""
    choice = [None]
    root = tk.Tk()
    root.title("Mod de rulare")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    model_exists = os.path.exists(MODEL_BEST)
    status = f"Model găsit: {MODEL_BEST}" if model_exists else "Niciun model salvat găsit."
    tk.Label(root, text=status, fg="green" if model_exists else "red").pack(padx=20, pady=(15, 5))

    def _set(val):
        choice[0] = val
        root.destroy()

    tk.Button(root, text="Training de la zero",  width=25,
              command=lambda: _set("train")).pack(padx=20, pady=5)

    tk.Button(root, text="Test controller simplu", width=25,
              command=lambda: _set("sanity")).pack(padx=20, pady=5)

    tk.Button(root, text="Test fizica rampa", width=25,
              command=lambda: _set("ramp_physics")).pack(padx=20, pady=5)

    if model_exists:
        tk.Button(root, text="Doar evaluare (model existent)", width=25,
                  command=lambda: _set("eval")).pack(padx=20, pady=5)

    tk.Button(root, text="Anulare", width=25,
              command=lambda: _set(None)).pack(padx=20, pady=(5, 15))

    root.mainloop()
    return choice[0]


mode = ask_mode()

if mode == "train":
    run_training()
    run_evaluation()
elif mode == "sanity":
    run_sanity_check()
elif mode == "ramp_physics":
    run_ramp_physics_test()
elif mode == "eval":
    run_evaluation()
else:
    print("[MAIN] Anulat.")
