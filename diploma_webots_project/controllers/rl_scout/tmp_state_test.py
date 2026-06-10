"""Smoke-test offline al extractiei de stare (LiDAR simulat, fara Webots):
verifica dimensiunea starii (23) si ca un obstacol sintetic apare in sectorul
corect cu valoarea corecta."""
import scout_env as S

N, LAYERS = 720, 4


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
    return env


def check(name, cond, detail):
    print(f"{'OK ' if cond else 'FAIL'}  {name:<46} {detail}")
    return cond


ok = True

# 1. Camp liber (toate razele la 5 m)
env = make_env([5.0] * (N * LAYERS))
s = env._get_state()
ok &= check("dimensiunea starii", len(s) == 23 and S.N_LIDAR_SECTORS + 11 == 23,
            f"len={len(s)} (asteptat 23)")
ok &= check("sectoare camp liber", all(abs(v - 0.5) < 1e-6 for v in s[:12]),
            f"sector[0]={s[0]:.3f} (asteptat 0.500 = 5m/10m)")

# 2. Obstacol sintetic la 0.5 m, banda de raze 350..370 (fata), pe toate straturile
ranges = [5.0] * (N * LAYERS)
for layer in range(LAYERS):
    for i in range(350, 371):
        ranges[layer * N + i] = 0.5
env = make_env(ranges)
s = env._get_state()
sec = s[:12]
ok &= check("obstacolul apare in sectorul frontal 6 (i=360)", abs(sec[6] - 0.05) < 0.01,
            f"sector[6]={sec[6]:.3f} (asteptat ~0.050)")
ok &= check("sectoarele laterale raman libere", abs(sec[0] - 0.5) < 1e-6 and abs(sec[11] - 0.5) < 1e-6,
            f"sector[0]={sec[0]:.3f}, sector[11]={sec[11]:.3f}")
ok &= check("profilul vertical frontal vede obstacolul", abs(s[12] - 0.05) < 0.01,
            f"upper_front={s[12]:.3f}")

# 3. Obstacol doar in sectorul lateral extrem (raze 0..59)
ranges = [5.0] * (N * LAYERS)
for layer in range(LAYERS):
    for i in range(0, 60):
        ranges[layer * N + i] = 1.0
env = make_env(ranges)
s = env._get_state()
ok &= check("obstacol lateral -> doar sectorul 0", abs(s[0] - 0.1) < 0.01 and abs(s[6] - 0.5) < 1e-6,
            f"sector[0]={s[0]:.3f} (asteptat ~0.100), sector[6]={s[6]:.3f}")

# 4. Distanta/unghi spre goal in stare
ok &= check("distanta normalizata spre goal", abs(s[15] - 10.0 / 120.0) < 1e-6, f"dist={s[15]:.4f}")
ok &= check("unghi 0 spre goal in fata", abs(s[16]) < 1e-9, f"ang={s[16]:.4f}")

print("\n" + ("TOATE TESTELE AU TRECUT — starea e pregatita pentru run 5."
              if ok else "EXISTA TESTE PICATE — NU porni run 5!"))
