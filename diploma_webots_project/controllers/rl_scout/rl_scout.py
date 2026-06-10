import os
import csv
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import torch
import numpy as np
from collections import deque
from controller import Supervisor
from dqn_agent import DQNAgent
from scout_env import ScoutEnv

# ======================================================
# CONFIG
# ======================================================

NUM_EPISODES   = 1200   # run 5+: stare nouă (12 sectoare) → puțin mai mult buget de învățare
# 6500 × 32 ms = 208 s simulare: ținte la >60 m cu ocoliri nu mai pică
# artificial în timeout (5000 = 160 s era chiar limita fizică pentru ~75 m
# la ~0,5 m/s). Episoadele blocate tot le taie anti-stuck-ul la ~600 pași.
MAX_STEPS      = 6500
EVAL_EPISODES  = 100     # default pentru evaluarea automată pe N episoade
QUICK_EPISODES = 50      # buton „test rapid”
RESUME_EPSILON = 0.30    # epsilon la reluarea antrenării (explorare redusă)
GOAL_KEY       = 'g'     # tasta pentru schimbat goal în eval

# Curriculum: ținte mai apropiate la început, apoi toată arena.
CURRICULUM_EPISODES = 400
CURRICULUM_START_DIST = 5.0

MODEL_BEST     = "dqn_model_best.pth"
MODEL_FINAL    = "dqn_model.pth"

LOG_DIR    = "logs"
FIG_DIR    = "figures"

# ======================================================
# LOGGING (CSV) — grafice generate ulterior cu plot_metrics.py
# ======================================================

def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def dump_obstacles_csv(suffix=""):
    """Salvează harta obstacolelor citită dinamic (pentru graficul de traiectorii).
    `suffix` separă hărțile diferitelor evaluări (ex. harta nevăzută), altfel
    figura de traiectorii a unei evaluări ar desena obstacolele alteia."""
    path = os.path.join(LOG_DIR, f"obstacles{suffix}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "r"])
        for obs in env.current_map_obstacles:
            writer.writerow([f"{obs['x']:.3f}", f"{obs['y']:.3f}", f"{obs['r']:.3f}"])
    print(f"[LOG] {len(env.current_map_obstacles)} obstacole salvate în {path}")


# ======================================================
# TKINTER HELPERS  (rulează în thread separat față de Webots)
# ======================================================

def _center(win):
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")


_HINT_RANGE = "Interval poiană: X, Y ∈ [-45, 45] m   (centru (0,0) = spawn)"


def _popup_coordinates(title, fields, hint=None):
    """fields: listă de (etichetă, valoare_implicită). Întoarce dict sau None."""
    result = {}
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    root.attributes("-topmost", True)
    try:
        ttk.Style(root).theme_use("clam")
    except Exception:
        pass

    frm = ttk.Frame(root, padding=14)
    frm.grid(sticky="nsew")
    ttk.Label(frm, text=title, font=("Segoe UI", 11, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(0, 10))

    entries = {}
    for i, (label, default) in enumerate(fields):
        ttk.Label(frm, text=label + ":").grid(row=i + 1, column=0, padx=6, pady=4, sticky="e")
        e = ttk.Entry(frm, width=12)
        e.grid(row=i + 1, column=1, padx=6, pady=4)
        e.insert(0, str(default))
        entries[label] = e
    r = len(fields) + 1
    if hint:
        ttk.Label(frm, text=hint, foreground="#666").grid(
            row=r, column=0, columnspan=2, pady=(6, 4))
        r += 1

    def on_ok():
        try:
            for label, e in entries.items():
                result[label] = float(e.get())
            root.destroy()
        except ValueError:
            messagebox.showerror("Eroare", "Introduceți valori numerice valide.")

    bf = ttk.Frame(frm)
    bf.grid(row=r, column=0, columnspan=2, pady=(8, 0))
    ttk.Button(bf, text="OK", width=10, command=on_ok).pack(side="left", padx=5)
    ttk.Button(bf, text="Anulare", width=10, command=root.destroy).pack(side="left", padx=5)

    _center(root)
    root.mainloop()
    return result if result else None


def ask_goal():
    data = _popup_coordinates("Setează Goal", [("Goal X", 15.0), ("Goal Y", 0.0)], _HINT_RANGE)
    if data:
        return data["Goal X"], data["Goal Y"]
    return None


def ask_start():
    data = _popup_coordinates(
        "Configurare Start & Goal",
        [("Start X", 0.0), ("Start Y", 0.0), ("Start θ (rad)", 0.0),
         ("Goal X", 15.0), ("Goal Y", 0.0)],
        _HINT_RANGE,
    )
    if data:
        return (data["Start X"], data["Start Y"], data["Start θ (rad)"],
                data["Goal X"], data["Goal Y"])
    return None


def print_episode_result(prefix):
    dist = np.sqrt((env.goal_x - env.x)**2 + (env.goal_y - env.y)**2)
    if env.done_reason == "goal":
        print(f"[OK] {prefix}: Goal atins! Pos: ({env.x:.2f}, {env.y:.2f})")
    else:
        print(
            f"[{env.done_reason}] {prefix}: {env.step_count} pasi. "
            f"Pos: ({env.x:.2f}, {env.y:.2f}) | Distanta goal: {dist:.2f} m"
        )


# ======================================================
# BASELINE REACTIV (fără DQN) — pentru comparație în lucrare.
# Folosește doar starea (LiDAR) și eroarea de navigație spre goal.
# ======================================================

def select_baseline_action(state):
    # state[:12] = sectoarele LiDAR (min-pooling 12 × 15°). Ordinea imaginii
    # Webots e STÂNGA → DREAPTA („scan lines running from left to right",
    # manualul de referință Lidar) → indecșii MICI = stânga robotului.
    sectors = state[:12]
    front = min(sectors[4:8])          # conul frontal ±30°
    left_half = min(sectors[0:6])
    right_half = min(sectors[6:12])
    _, angle, _ = env.get_navigation_error()

    if front < 0.12:
        return 3 if left_half > right_half else 4
    if angle > 0.45:
        return 3
    if angle < -0.45:
        return 4
    if angle > 0.12:
        return 1
    if angle < -0.12:
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

def curriculum_goal_dist(episode):
    if episode >= CURRICULUM_EPISODES:
        return None
    frac = episode / CURRICULUM_EPISODES
    return CURRICULUM_START_DIST + frac * (2.0 * env.arena_limit - CURRICULUM_START_DIST)


class TrainingMonitor:
    """Fereastră live (Tk + matplotlib) cu progresul antrenării: reward/episod,
    media pe 50, rata de succes, epsilon, best. Actualizată la fiecare episod cu
    `root.update()` (fără mainloop → nu blochează bucla Webots). Dacă matplotlib/Tk
    nu sunt disponibile, se dezactivează elegant și antrenarea continuă (consolă)."""

    def __init__(self, total):
        self.ok = False
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            print(f"[MONITOR] Fereastra live dezactivată (matplotlib indisponibil): {exc}")
            return

        self.total = total
        self.xs, self.rew, self.avg = [], [], []
        self.root = tk.Tk()
        self.root.title("Progres antrenare — live")
        try:
            ttk.Style(self.root).theme_use("clam")
        except Exception:
            pass

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        self.lbl = {}
        for i, (k, t) in enumerate([("ep", "Episod"), ("rew", "Reward"), ("avg", "Avg50"),
                                    ("succ", "Succes50"), ("eps", "Epsilon"), ("best", "Best")]):
            ttk.Label(top, text=t + ":", font=("Segoe UI", 9, "bold")).grid(
                row=0, column=2 * i, sticky="e", padx=(8, 2))
            v = ttk.Label(top, text="—", width=8)
            v.grid(row=0, column=2 * i + 1, sticky="w")
            self.lbl[k] = v

        self.fig = Figure(figsize=(6.6, 3.3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Episod")
        self.ax.set_ylabel("Reward")
        self.ax.grid(alpha=0.3)
        (self.l_rew,) = self.ax.plot([], [], color="#c9c9c9", lw=0.8, label="reward")
        (self.l_avg,) = self.ax.plot([], [], color="#d62728", lw=2.0, label="media 50")
        self.ax.legend(loc="upper left", fontsize=8)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.pbar = ttk.Progressbar(self.root, maximum=max(1, total))
        self.pbar.pack(fill="x", padx=8, pady=(0, 8))

        self.ok = True
        try:
            self.root.update()
        except Exception:
            self.ok = False

    def update(self, ep, reward, avg, succ, eps, best, redraw=True):
        if not self.ok:
            return
        try:
            self.xs.append(ep)
            self.rew.append(reward)
            self.avg.append(avg)
            self.lbl["ep"].config(text=f"{ep}/{self.total}")
            self.lbl["rew"].config(text=f"{reward:.1f}")
            self.lbl["avg"].config(text=f"{avg:.1f}")
            self.lbl["succ"].config(text=f"{succ * 100:.0f}%")
            self.lbl["eps"].config(text=f"{eps:.3f}")
            self.lbl["best"].config(text=f"{best:.1f}")
            self.pbar["value"] = len(self.xs)
            if redraw:
                self.l_rew.set_data(self.xs, self.rew)
                self.l_avg.set_data(self.xs, self.avg)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw_idle()
            self.root.update()
        except tk.TclError:
            self.ok = False            # fereastra închisă de utilizator
        except Exception:
            self.ok = False

    def finish(self):
        if not self.ok:
            return
        try:
            self.root.title("Progres antrenare — TERMINAT")
            self.l_rew.set_data(self.xs, self.rew)
            self.l_avg.set_data(self.xs, self.avg)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            self.root.update()
        except Exception:
            self.ok = False


def run_training(num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, resume=False):
    print("=" * 50)
    print(f"  TRAINING MODE ({'CONTINUARE' if resume else 'DE LA ZERO'}) — "
          f"{num_episodes} episoade × {max_steps} pași")
    print("=" * 50)

    _ensure_dirs()
    # Fără asta, ScoutEnv tăia episodul la valoarea lui implicită (timeout
    # intern), ignorând „Pași/ep" din meniu.
    env.max_steps = max_steps
    metrics_path = os.path.join(LOG_DIR, "training_metrics.csv")
    header = ["episode", "reward", "avg50", "steps", "done_reason", "epsilon", "loss"]

    start_ep = 0
    if resume:
        src = MODEL_FINAL if os.path.exists(MODEL_FINAL) else MODEL_BEST
        if os.path.exists(src):
            try:
                agent.q_network.load_state_dict(torch.load(src))
                agent.target_network.load_state_dict(agent.q_network.state_dict())
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Modelul {src} nu mai e compatibil cu starea curentă "
                    f"({env.state_dim} valori). Antrenează de la zero."
                ) from exc
            agent.epsilon = max(agent.epsilon_min, RESUME_EPSILON)
            print(f"✓ Reluat din {src} | epsilon={agent.epsilon:.3f}")
        else:
            print("[WARN] Niciun checkpoint găsit — pornesc de la zero.")
        # Continuă numerotarea episoadelor din CSV-ul existent.
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as rf:
                    rows = list(csv.reader(rf))
                if len(rows) > 1 and rows[-1] and rows[-1][0].isdigit():
                    start_ep = int(rows[-1][0])
            except Exception:
                start_ep = 0
        metrics_file = open(metrics_path, "a", newline="")
        metrics_writer = csv.writer(metrics_file)
        if start_ep == 0:
            metrics_writer.writerow(header)
    else:
        agent.epsilon = 1.0
        metrics_file = open(metrics_path, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(header)

    episode_rewards = deque(maxlen=50)
    best_reward     = -float("inf")   # cel mai bun episod (doar informativ)
    best_avg50      = -float("inf")   # criteriul de salvare MODEL_BEST
    obstacles_dumped = False
    successes = deque(maxlen=50)
    monitor = TrainingMonitor(num_episodes)

    for episode in range(num_episodes):
        ep = start_ep + episode + 1

        state = env.reset(random_goal=True, random_start=True,
                          goal_max_dist=curriculum_goal_dist(ep - 1))

        if not obstacles_dumped:
            dump_obstacles_csv()
            obstacles_dumped = True

        total_reward = 0.0
        loss_sum = 0.0
        loss_count = 0

        for step in range(max_steps):
            action                   = agent.select_action(state)
            next_state, reward, done = env.step(action)

            if not env.simulation_running:
                metrics_file.close()
                return

            agent.store(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                loss_sum += loss
                loss_count += 1

            state        = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()   # decădere epsilon o dată pe episod

        episode_rewards.append(total_reward)
        avg_reward = float(np.mean(episode_rewards))
        avg_loss = (loss_sum / loss_count) if loss_count else 0.0

        metrics_writer.writerow([
            ep, f"{total_reward:.3f}", f"{avg_reward:.3f}",
            env.step_count, env.done_reason or "none",
            f"{agent.epsilon:.4f}", f"{avg_loss:.5f}",
        ])
        metrics_file.flush()

        if episode % 10 == 9 or episode == num_episodes - 1:
            print(
                f"Ep {ep:4d} | Reward: {total_reward:8.2f} | Avg50: {avg_reward:8.2f} | "
                f"ε: {agent.epsilon:.4f} | {env.done_reason} | "
                f"Goal: ({env.goal_x:6.2f}, {env.goal_y:6.2f})"
            )

        if total_reward > best_reward:
            best_reward = total_reward

        # MODEL_BEST = cea mai bună MEDIE pe 50 de episoade, nu un singur
        # episod norocos (un episod lung cu progres mare dădea un „best"
        # zgomotos): media pe 50 e exact criteriul raportat în lucrare, iar
        # evaluările încarcă MODEL_BEST.
        if len(episode_rewards) >= min(50, num_episodes) and avg_reward > best_avg50:
            best_avg50 = avg_reward
            torch.save(agent.q_network.state_dict(), MODEL_BEST)

        if ep % 100 == 0:
            torch.save(agent.q_network.state_dict(), f"dqn_model_ep{ep}.pth")

        successes.append(1.0 if env.done_reason == "goal" else 0.0)
        monitor.update(ep, total_reward, avg_reward, float(np.mean(successes)),
                       agent.epsilon, best_reward,
                       redraw=(episode % 5 == 0 or episode == num_episodes - 1))

    torch.save(agent.q_network.state_dict(), MODEL_FINAL)
    metrics_file.close()
    monitor.finish()
    print(f"\n✓ Training încheiat. Best episod: {best_reward:.2f} | Best Avg50: {best_avg50:.2f}")
    print(f"  Modele salvate: {MODEL_FINAL}, {MODEL_BEST}")
    print(f"  Metrici: {metrics_path}")


# ======================================================
# EVALUATION
# ======================================================

def _open_eval_summary():
    _ensure_dirs()
    path = os.path.join(LOG_DIR, "eval_summary.csv")
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["run", "start_x", "start_y", "goal_x", "goal_y", "steps", "done_reason"])
    return f, writer


def run_evaluation():
    print("=" * 50)
    print("  EVALUATION MODE — RL pur")
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

    summary_file, summary_writer = _open_eval_summary()
    run_index = 0

    config = [None]
    def _ask():
        config[0] = ask_start()
    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    t.join()

    if config[0] is None:
        print("[EVAL] Anulat de utilizator.")
        summary_file.close()
        return

    sx, sy, stheta, gx, gy = config[0]
    env.set_goal(gx, gy)
    state = env.reset(start_x=sx, start_y=sy, start_theta=stheta, random_goal=False)
    dump_obstacles_csv()

    print(f"  Start: ({sx:.2f}, {sy:.2f}, θ={stheta:.2f} rad)")
    print(f"  Goal:  ({env.goal_x:.2f}, {env.goal_y:.2f})\n")

    traj = []
    new_goal_request = [False]

    while env.simulation_running:

        key = keyboard.getKey()
        if key == ord(GOAL_KEY.upper()) or key == ord(GOAL_KEY.lower()):
            new_goal_request[0] = True

        if new_goal_request[0]:
            new_goal_request[0] = False
            coords = [None]
            def _ask_goal():
                coords[0] = ask_goal()
            tg = threading.Thread(target=_ask_goal, daemon=True)
            tg.start()
            tg.join()
            if coords[0] is not None:
                env.set_goal(coords[0][0], coords[0][1])
                print(f"[EVAL] Nou goal setat: ({env.goal_x:.2f}, {env.goal_y:.2f})")

        action = agent.select_action(state)
        state, reward, done = env.step(action)
        traj.append((env.step_count, env.x, env.y, env.goal_x, env.goal_y))

        if not env.simulation_running:
            break

        if done:
            print_episode_result("Eval")
            # Salvează traiectoria pentru graficul X-Y
            traj_path = os.path.join(LOG_DIR, f"eval_traj_{run_index}.csv")
            with open(traj_path, "w", newline="") as tf:
                tw = csv.writer(tf)
                tw.writerow(["step", "x", "y", "goal_x", "goal_y"])
                tw.writerows([(s, f"{x:.3f}", f"{y:.3f}", f"{gx_:.3f}", f"{gy_:.3f}")
                              for (s, x, y, gx_, gy_) in traj])
            summary_writer.writerow([
                run_index, f"{traj[0][1]:.2f}", f"{traj[0][2]:.2f}",
                f"{env.goal_x:.2f}", f"{env.goal_y:.2f}",
                env.step_count, env.done_reason or "none",
            ])
            summary_file.flush()
            run_index += 1
            traj = []

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
            state = env.reset(start_x=sx, start_y=sy, start_theta=stheta, random_goal=False)
            print(f"  Start: ({sx:.2f}, {sy:.2f}) | Goal: ({env.goal_x:.2f}, {env.goal_y:.2f})")

    summary_file.close()


def run_eval_batch(eval_n=EVAL_EPISODES, save_traj=20, random_start=True, tag="",
                   policy_fn=None, max_steps=MAX_STEPS):
    """Evaluare automată: N episoade cu goal random, fără intervenția
    utilizatorului. random_start=True → start random navigabil (ca la antrenare);
    altfel start din centru (figură de comparație). policy_fn=None → politica
    DQN (încarcă MODEL_BEST); altfel folosește funcția dată (ex. baseline-ul
    reactiv, pentru comparația DQN vs. reactiv din lucrare). Raportează rata
    de succes / pași / coliziuni / blocaje și scrie eval_summary{tag}.csv +
    primele `save_traj` traiectorii (pentru grafice)."""
    mode_txt = "start random" if random_start else "start centru"
    policy_txt = "DQN" if policy_fn is None else "baseline reactiv"
    print("=" * 50)
    print(f"  EVALUARE AUTOMATĂ [{policy_txt}] — {eval_n} episoade ({mode_txt}, goal random)")
    print("=" * 50)

    if policy_fn is None:
        if not os.path.exists(MODEL_BEST):
            raise FileNotFoundError(f"Nu există {MODEL_BEST}. Rulează training întâi.")
        try:
            agent.q_network.load_state_dict(torch.load(MODEL_BEST))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Modelul {MODEL_BEST} nu mai e compatibil cu starea "
                f"({env.state_dim} valori). Antrenează de la zero."
            ) from exc
        agent.q_network.eval()
        agent.epsilon = 0.0
        print(f"✓ Model încărcat: {MODEL_BEST}\n")

    env.max_steps = max_steps

    _ensure_dirs()
    summary_path = os.path.join(LOG_DIR, f"eval_summary{tag}.csv")
    sf = open(summary_path, "w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["run", "start_x", "start_y", "goal_x", "goal_y", "steps", "done_reason"])

    results = []
    for i in range(eval_n):
        theta = float(np.random.uniform(-np.pi, np.pi))
        if random_start:
            state = env.reset(random_goal=True, random_start=True)
        else:
            state = env.reset(start_x=0.0, start_y=0.0, start_theta=theta,
                              random_goal=True)
        if i == 0:
            dump_obstacles_csv(tag)

        traj = []
        for step in range(max_steps):
            action = policy_fn(state) if policy_fn else agent.select_action(state)
            state, reward, done = env.step(action)
            if not env.simulation_running:
                sf.close()
                print("[EVAL] Oprit de utilizator.")
                return
            traj.append((env.step_count, env.x, env.y, env.goal_x, env.goal_y))
            if done:
                break

        reason = env.done_reason or "timeout"
        results.append((reason, env.step_count))
        sw.writerow([i, f"{traj[0][1]:.2f}", f"{traj[0][2]:.2f}",
                     f"{env.goal_x:.2f}", f"{env.goal_y:.2f}", env.step_count, reason])
        sf.flush()

        if i < save_traj:
            with open(os.path.join(LOG_DIR, f"eval_traj{tag}_{i}.csv"), "w", newline="") as tf:
                tw = csv.writer(tf)
                tw.writerow(["step", "x", "y", "goal_x", "goal_y"])
                tw.writerows([(s, f"{x:.3f}", f"{y:.3f}", f"{a:.3f}", f"{b:.3f}")
                              for (s, x, y, a, b) in traj])

        if (i + 1) % 10 == 0 or i == eval_n - 1:
            sr = 100.0 * sum(1 for r in results if r[0] == "goal") / len(results)
            print(f"  {i + 1:3d}/{eval_n}  ·  succes curent: {sr:.0f}%")

    sf.close()

    n = len(results)
    known = {"goal", "collision", "rollover", "out_of_bounds", "stuck"}

    def rate(reason):
        return 100.0 * sum(1 for r in results if r[0] == reason) / n

    timeout = 100.0 * sum(1 for r in results if r[0] not in known) / n
    succ_steps = [r[1] for r in results if r[0] == "goal"]

    print("\n" + "=" * 50)
    print(f"  REZULTATE EVALUARE — {n} episoade")
    print("-" * 50)
    print(f"   succes:         {rate('goal'):5.1f}%")
    print(f"   coliziune:      {rate('collision'):5.1f}%")
    print(f"   răsturnare:     {rate('rollover'):5.1f}%")
    print(f"   ieșire hartă:   {rate('out_of_bounds'):5.1f}%")
    print(f"   blocat (stuck): {rate('stuck'):5.1f}%")
    print(f"   timeout:        {timeout:5.1f}%")
    if succ_steps:
        print(f"   pași medii (la succes): {np.mean(succ_steps):.0f}")
    print("-" * 50)
    print(f"   → {summary_path}")
    print("   Rulează 'python plot_metrics.py' pentru grafice.")
    print("=" * 50)


def run_baseline():
    print("=" * 50)
    print("  BASELINE REACTIV - fara DQN (comparatie)")
    print("=" * 50)

    config = [None]
    def _ask():
        config[0] = ask_start()
    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    t.join()

    if config[0] is None:
        print("[BASELINE] Anulat de utilizator.")
        return

    sx, sy, stheta, gx, gy = config[0]
    env.set_goal(gx, gy)
    state = env.reset(start_x=sx, start_y=sy, start_theta=stheta, random_goal=False)

    print(f"  Start: ({sx:.2f}, {sy:.2f}, theta={stheta:.2f} rad)")
    print(f"  Goal:  ({env.goal_x:.2f}, {env.goal_y:.2f})")

    while env.simulation_running:
        action = select_baseline_action(state)
        state, reward, done = env.step(action)

        if env.step_count % 50 == 0:
            dist, angle, _ = env.get_navigation_error()
            print(
                f"[BASELINE] pas {env.step_count:4d}: "
                f"pos=({env.x:.2f}, {env.y:.2f}, z={env.z:.2f}) | "
                f"dist={dist:.2f} angle={angle:.2f} | "
                f"roll={env.roll:.2f} pitch={env.pitch:.2f}"
            )

        if not env.simulation_running:
            break

        if done:
            print_episode_result("Baseline")
            for m in motors:
                m.setVelocity(0.0)
            break


# ======================================================
# MAIN
# ======================================================

def ask_mode():
    cfg = {"mode": None, "episodes": NUM_EPISODES, "max_steps": MAX_STEPS,
           "eval_n": EVAL_EPISODES, "resume": False}
    root = tk.Tk()
    root.title("Navigație autonomă RL — Scout")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    try:
        ttk.Style(root).theme_use("clam")
    except Exception:
        pass

    frm = ttk.Frame(root, padding=16)
    frm.grid(sticky="nsew")

    ttk.Label(frm, text="Navigație autonomă cu Reinforcement Learning",
              font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(frm, text="Pioneer 3-AT  ·  mediu nestructurat  ·  DQN",
              foreground="#555").grid(row=1, column=0, sticky="w", pady=(0, 8))

    model_exists = os.path.exists(MODEL_BEST)
    ttk.Label(
        frm,
        text=("●  Model găsit: " + MODEL_BEST) if model_exists else "○  Niciun model salvat",
        foreground="#1a7f37" if model_exists else "#b00020",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=2, column=0, sticky="w")
    ttk.Separator(frm, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=10)

    cframe = ttk.Frame(frm)
    cframe.grid(row=4, column=0, sticky="w", pady=(0, 6))
    ttk.Label(cframe, text="Episoade:").grid(row=0, column=0, sticky="e", padx=(0, 3))
    e_ep = ttk.Entry(cframe, width=7); e_ep.insert(0, str(NUM_EPISODES)); e_ep.grid(row=0, column=1, padx=(0, 10))
    ttk.Label(cframe, text="Pași/ep:").grid(row=0, column=2, sticky="e", padx=(0, 3))
    e_ms = ttk.Entry(cframe, width=7); e_ms.insert(0, str(MAX_STEPS)); e_ms.grid(row=0, column=3, padx=(0, 10))
    ttk.Label(cframe, text="Eval N:").grid(row=0, column=4, sticky="e", padx=(0, 3))
    e_ev = ttk.Entry(cframe, width=7); e_ev.insert(0, str(EVAL_EPISODES)); e_ev.grid(row=0, column=5)

    def _int(entry, default):
        try:
            return max(1, int(float(entry.get())))
        except Exception:
            return default

    def choose(mode, resume=False, episodes=None):
        cfg["mode"] = mode
        cfg["resume"] = resume
        cfg["episodes"] = episodes if episodes is not None else _int(e_ep, NUM_EPISODES)
        cfg["max_steps"] = _int(e_ms, MAX_STEPS)
        cfg["eval_n"] = _int(e_ev, EVAL_EPISODES)
        root.destroy()

    rowc = [5]

    def add(text, desc, cmd, enabled=True):
        b = ttk.Button(frm, text=text, width=38, command=cmd)
        b.grid(row=rowc[0], column=0, sticky="w", pady=(7, 0))
        if not enabled:
            b.state(["disabled"])
        ttk.Label(frm, text="     " + desc, foreground="#666",
                  font=("Segoe UI", 8)).grid(row=rowc[0] + 1, column=0, sticky="w")
        rowc[0] += 2

    add("Antrenează de la zero",
        "DQN nou · curriculum · metrici CSV", lambda: choose("train"))
    add("Continuă antrenarea",
        "reia din checkpoint (epsilon redus)",
        lambda: choose("train", resume=True), enabled=model_exists)
    add("Evaluare automată (N episoade)",
        "start random, goal random → rată succes / pași / coliziuni",
        lambda: choose("eval_batch"), enabled=model_exists)
    add("Evaluare comparativă (start centru)",
        "start din centru, goal random → figură de comparație",
        lambda: choose("eval_batch_center"), enabled=model_exists)
    add("Evaluare pe hartă nevăzută",
        "deschide worlds/world_eval.wbt întâi! → eval_summary_unseen.csv (generalizare)",
        lambda: choose("eval_batch_unseen"), enabled=model_exists)
    add("Evaluare manuală (start/goal)",
        "alegi start & goal · tasta G schimbă goal-ul live",
        lambda: choose("eval"), enabled=model_exists)
    add("Baseline reactiv (fără DQN)",
        "LiDAR + direcție goal · comparație în lucrare", lambda: choose("baseline"))
    add("Baseline reactiv — N episoade",
        "batch automat fără DQN → eval_summary_baseline.csv (figura DQN vs reactiv)",
        lambda: choose("baseline_batch"))
    add("Test rapid (50 episoade)",
        "antrenare scurtă de verificare",
        lambda: choose("train", episodes=QUICK_EPISODES))

    ttk.Button(frm, text="Anulare", width=38, command=root.destroy).grid(
        row=rowc[0], column=0, sticky="w", pady=(12, 0))

    _center(root)
    root.mainloop()
    return cfg


cfg = ask_mode()
mode = cfg["mode"]

if mode == "train":
    run_training(cfg["episodes"], cfg["max_steps"], resume=cfg["resume"])
    run_eval_batch(cfg["eval_n"], random_start=True,                  # principal: start random
                   max_steps=cfg["max_steps"])
    run_eval_batch(cfg["eval_n"], random_start=False, tag="_center",  # comparație: start centru
                   max_steps=cfg["max_steps"])
elif mode == "eval_batch":
    run_eval_batch(cfg["eval_n"], random_start=True, max_steps=cfg["max_steps"])
elif mode == "eval_batch_center":
    run_eval_batch(cfg["eval_n"], random_start=False, tag="_center",
                   max_steps=cfg["max_steps"])
elif mode == "eval_batch_unseen":
    run_eval_batch(cfg["eval_n"], random_start=True, tag="_unseen",
                   max_steps=cfg["max_steps"])
elif mode == "baseline_batch":
    run_eval_batch(cfg["eval_n"], random_start=True, tag="_baseline",
                   policy_fn=select_baseline_action, max_steps=cfg["max_steps"])
elif mode == "eval":
    run_evaluation()
elif mode == "baseline":
    run_baseline()
else:
    print("[MAIN] Anulat.")
