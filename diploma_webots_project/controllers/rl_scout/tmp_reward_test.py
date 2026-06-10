"""Test unitar offline al functiei de recompensa (fara Webots) — pre-flight
run 4. Verifica directia stimulentelor: parcatul/cercurile nu mai produc
venit, progresul da, terminarile au semnul/valoarea corecte."""
import math
import scout_env as S

env = S.ScoutEnv.__new__(S.ScoutEnv)


def setup(x=0.0, y=0.0, goal=(20.0, 0.0), prev=None, v=0.0, w=0.0,
          roll=0.0, pitch=0.0, step=300, anchor_age=100):
    env.x, env.y = x, y
    env.goal_x, env.goal_y = goal
    env.goal_tolerance = 0.45
    env.max_safe_roll = 0.75
    env.max_safe_pitch = 0.75
    env.bounds_margin = 1.5
    env.roll, env.pitch = roll, pitch
    env.v, env.w = v, w
    d = math.hypot(goal[0] - x, goal[1] - y)
    env.prev_distance = d if prev is None else prev
    env.step_count = step
    env._stuck_anchor_x, env._stuck_anchor_y = x, y
    env._stuck_anchor_step = step - anchor_age
    env.stuck_patience = 600
    env.stuck_move_thresh = 0.5
    env.done_reason = None


def state(front=1.0, flank=1.0, side=1.0, upper=1.0,
          midf=1.0, hp=0.0, dist=0.2, ang=0.0, vn=0.0, wn=0.0,
          z=0.0, ro=0.0, pi=0.0, yr=0.0):
    # Layout 23: [0:12] sectoare LiDAR (5,6=frontale; 3,4,7,8=flancuri),
    # [12:15] profil vertical, [15:23] navigatie/IMU.
    s = [side] * 12
    s[3] = s[4] = s[7] = s[8] = flank
    s[5] = s[6] = front
    return s + [upper, midf, hp, dist, ang, vn, wn, z, ro, pi, yr]


def check(name, cond, detail):
    print(f"{'OK ' if cond else 'FAIL'}  {name:<42} {detail}")
    return cond


ok = True

# 1. Parcat cu fata la goal, camp deschis (exploit-ul run 3) -> trebuie <= 0
setup(v=0.0)
r, done = env._compute_reward(state(ang=0.0))
ok &= check("parcat cu fata la goal", (r < 0) and not done, f"r={r:+.4f} (run3: +1.20/pas)")

# 2. Mers cu viteza plina spre goal -> pozitiv
setup(v=0.55, prev=20.0176)  # progres 0.0176 m/pas
r, done = env._compute_reward(state(ang=0.0, vn=0.275))
ok &= check("mers spre goal (progres maxim)", (r > 0.3) and not done, f"r={r:+.4f}")

# 3. Cerc mic / oscilatie (progres net 0, v>0) -> ~0, fara venit
setup(v=0.45, w=1.0)
r, done = env._compute_reward(state(ang=0.3))
ok &= check("cerc mic (farming-ul rezidual)", (-0.1 < r <= 0) and not done, f"r={r:+.4f} (inainte: ~+0.4/pas)")

# 4. Impins intr-un obstacol (v comandat>0, progres 0, LiDAR aproape) -> negativ clar
setup(v=0.55)
r, done = env._compute_reward(state(front=0.04, flank=0.06))
ok &= check("impins in obstacol", (r < -2.5) and not done, f"r={r:+.4f}")

# 5. Detur temporar (progres negativ moderat) -> negativ mic, nu catastrofal
setup(v=0.55, prev=19.9824)  # se departeaza cu 0.0176 m/pas
r, done = env._compute_reward(state(ang=0.5))
ok &= check("detur (se departeaza temporar)", (-0.5 < r < 0) and not done, f"r={r:+.4f}")

# 6. Anti-stuck: 600 pasi lipit de ancora -> -20 si terminare
setup(v=0.0, anchor_age=600)
r, done = env._compute_reward(state())
ok &= check("anti-stuck dupa 600 pasi", done and env.done_reason == "stuck" and r < -19.5,
            f"r={r:+.2f}, reason={env.done_reason}")

# 7. Coliziune LiDAR -> -20 si terminare
setup(v=0.55)
r, done = env._compute_reward(state(front=0.02))
ok &= check("coliziune", done and env.done_reason == "collision" and r == -20.0,
            f"r={r:+.2f}, reason={env.done_reason}")

# 8. Goal atins -> +200 si terminare
setup(x=19.7, v=0.55)
r, done = env._compute_reward(state(dist=0.003))
ok &= check("goal atins", done and env.done_reason == "goal" and r == 200.0,
            f"r={r:+.2f}, reason={env.done_reason}")

# 9. Rasturnare -> -20 si terminare
setup(roll=0.9)
r, done = env._compute_reward(state(ro=0.28))
ok &= check("rasturnare", done and env.done_reason == "rollover" and r == -20.0,
            f"r={r:+.2f}, reason={env.done_reason}")

# 10. Audit static: niciun termen pe pas nu mai poate fi pozitiv fara progres
setup(v=0.55, w=0.0)
r, done = env._compute_reward(state(ang=0.0))  # progres 0, conditii perfecte
ok &= check("venit maxim posibil FARA progres", (r <= 0) and not done,
            f"r={r:+.4f} (trebuie <= 0)")

print("\n" + ("TOATE TESTELE AU TRECUT — reward-ul e pregatit pentru run 4."
              if ok else "EXISTA TESTE PICATE — NU porni run 4!"))
