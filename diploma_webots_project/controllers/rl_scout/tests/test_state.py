"""Smoke-test offline al extractiei de stare (LiDAR simulat, fara Webots):
verifica dimensiunea cadrului (23), normalizarea la LIDAR_MAX_RANGE, ca un
obstacol sintetic apare in sectorul corect si frame stacking-ul (run 6)."""

import os
import sys

# Bootstrap cale: ruleaza din orice director. Adauga folderul controllerului
# (parintele) la sys.path (pentru `import scout_env` / `dqn_agent`) si fixeaza
# cwd acolo (pentru path-urile relative logs/ si runs/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from collections import deque

import scout_env as S

N, LAYERS = 720, 4
R = S.LIDAR_MAX_RANGE


class FakeLidar:
    def __init__(self, ranges):
        self._r = ranges

    def getRangeImage(self):
        return self._r

    def getHorizontalResolution(self):
        return N

    def getNumberOfLayers(self):
        return LAYERS


def make_env(ranges):
    env = S.ScoutEnv.__new__(S.ScoutEnv)
    env.lidar = FakeLidar(ranges)
    env.x = env.y = 0.0
    env.theta = 0.0
    env.goal_x, env.goal_y = 10.0, 0.0
    env.v = env.w = 0.0
    env.z = 0.1
    env.roll = env.pitch = 0.0
    env.gyro_rate = [0.0, 0.0, 0.0]
    env._frames = deque(maxlen=S.FRAME_STACK)
    return env


def check(name, cond, detail):
    print(f"{'OK ' if cond else 'FAIL'}  {name:<46} {detail}")
    return cond


ok = True

# 1. Camp liber (toate razele la 5 m) -> sectoare 5/R
env = make_env([5.0] * (N * LAYERS))
s = env._get_state()
ok &= check("dimensiunea cadrului", len(s) == 23 and S.N_LIDAR_SECTORS + 11 == 23,
            f"len={len(s)} (asteptat 23)")
ok &= check("sectoare camp liber", all(abs(v - 5.0 / R) < 1e-6 for v in s[:12]),
            f"sector[0]={s[0]:.3f} (asteptat {5.0 / R:.3f} = 5m/{R:.0f}m)")

# 2. Obstacol sintetic la 0.5 m, banda de raze 350..370 (fata), pe toate straturile
ranges = [5.0] * (N * LAYERS)
for layer in range(LAYERS):
    for i in range(350, 371):
        ranges[layer * N + i] = 0.5
env = make_env(ranges)
s = env._get_state()
sec = s[:12]
ok &= check("obstacolul apare in sectorul frontal 6 (i=360)", abs(sec[6] - 0.5 / R) < 0.005,
            f"sector[6]={sec[6]:.3f} (asteptat ~{0.5 / R:.3f})")
ok &= check("sectoarele laterale raman libere", abs(sec[0] - 5.0 / R) < 1e-6 and abs(sec[11] - 5.0 / R) < 1e-6,
            f"sector[0]={sec[0]:.3f}, sector[11]={sec[11]:.3f}")
ok &= check("profilul vertical frontal vede obstacolul", abs(s[12] - 0.5 / R) < 0.005,
            f"upper_front={s[12]:.3f}")

# 3. Obstacol doar in sectorul lateral extrem (raze 0..59)
ranges = [5.0] * (N * LAYERS)
for layer in range(LAYERS):
    for i in range(0, 60):
        ranges[layer * N + i] = 1.0
env = make_env(ranges)
s = env._get_state()
ok &= check("obstacol lateral -> doar sectorul 0", abs(s[0] - 1.0 / R) < 0.005 and abs(s[6] - 5.0 / R) < 1e-6,
            f"sector[0]={s[0]:.3f} (asteptat ~{1.0 / R:.3f}), sector[6]={s[6]:.3f}")

# 4. Distanta/unghi spre goal in cadru
ok &= check("distanta normalizata spre goal", abs(s[15] - 10.0 / 120.0) < 1e-6, f"dist={s[15]:.4f}")
ok &= check("unghi 0 spre goal in fata", abs(s[16]) < 1e-9, f"ang={s[16]:.4f}")

# 5. Frame stacking: starea politicii = FRAME_STACK cadre, CEL MAI RECENT primul.
# Config activa run 5: FRAME_STACK=1 -> starea = un singur cadru (23 dims).
f1 = env._get_state()
for _ in range(S.FRAME_STACK):
    env._frames.append(f1)
env.goal_x = 5.0                      # cadrul urmator difera (alta distanta)
f2 = env._get_state()
env._frames.append(f2)
st = env._stack_frames()
exp = S.FRAME_STACK * 23
ok &= check("dimensiunea starii (FRAME_STACK cadre)", len(st) == exp,
            f"len={len(st)} (asteptat {exp} = {S.FRAME_STACK}x23)")
ok &= check("cadrul CURENT e primul in stiva", st[:23] == f2,
            f"st[15]={st[15]:.4f} (f2 curent)")
if S.FRAME_STACK > 1:
    ok &= check("cadrul anterior urmeaza", st[23:46] == f1, f"st[38]={st[38]:.4f} (f1)")

print("\n" + (f"TOATE TESTELE AU TRECUT — stare {exp}-dim (FRAME_STACK={S.FRAME_STACK}, config run 5)."
              if ok else "EXISTA TESTE PICATE!"))
