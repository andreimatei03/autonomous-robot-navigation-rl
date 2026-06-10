import math
import random

import numpy as np


# ----------------------------------------------------------------------
# Tabel rază de gabarit (m) pe tip de PROTO / DEF, folosit DOAR pentru
# plasarea sigură a goal-ului (nu intervine în percepția RL, care e LiDAR).
# Citirea dinamică din Webots completează automat lista de obstacole.
# ----------------------------------------------------------------------
PROTO_RADIUS = {
    # Vegetație de interior (Pine e rezervat pădurii de graniță → NU se înregistrează)
    "SimpleTree": 0.8, "Oak": 1.1, "Cypress": 0.7,
    # Vehicule
    "ToyotaPrius": 2.4, "CitroenCZero": 1.8, "BmwX5": 2.6,
    "MercedesBenzSprinter": 3.2, "TeslaModel3": 2.4, "RangeRoverSportSVR": 2.6,
    # Clădiri (raze de gabarit reale → evită suprapunerile)
    "SimpleTwoFloorsHouse": 7.0, "ModernHouse": 11.0, "HouseWithGarage": 9.0,
    "BungalowStyleHouse": 12.0, "Barn": 9.0, "Warehouse": 14.0, "GasStation": 11.0,
    # Structuri/decor adăugate manual în world.wbt (altfel goal-ul cădea în ele)
    "Windmill": 3.5, "Swing": 2.5, "DogHouse": 0.9, "IntermodalContainer": 3.5,
    # Mobilier urban / trafic
    "StreetLight": 0.4, "FireHydrant": 0.4, "TrafficCone": 0.3,
    # Containere / industrie
    "OilBarrel": 0.5, "CardboardBox": 0.5, "WoodenBox": 0.6,
    "WoodenPalletStack": 1.0,
    # Props naturale / animale (un-DEF'd → altfel invizibile la plasarea goal-ului)
    "Rock": 1.0, "Dog": 0.6,
    # Oameni
    "Pedestrian": 0.5,
}
DEFAULT_OBSTACLE_RADIUS = 0.6

# Rază pe DEF exact pentru obstacole Solid personalizate.
DEF_RADIUS = {
    "OBS_PARTIALWALL1": 2.1, "OBS_PARTIALWALL2": 1.6,
}

# Rază pe PREFIX de DEF, pentru obstacole generate auto-numerotate
# (OBS_ROCK0, OBS_TREE3, OBS_CONTAINER1, OBS_BARREL2, ...).
DEF_PREFIX_RADIUS = {
    "OBS_ROCK": 1.2, "OBS_TREE": 0.9, "OBS_BUSH": 0.6,
    "OBS_CONTAINER": 2.0, "OBS_PIPE": 1.3, "OBS_BARREL": 0.5,
    "OBS_CRATE": 0.7, "OBS_LOG": 2.2,
    # Zone inaccesibile (clustere impenetrabile)
    "OBS_BOULDER": 1.0, "OBS_THKT": 0.9, "OBS_RUBBLE": 0.7,
    # Ruină-landmark (ziduri de piatră) — adăugată direct în world.wbt
    "OBS_RUIN": 2.0,
    # Platforma/peronul casei (slab pavat ridicat, robotul nu urcă pe el) →
    # exclude goal-ul de pe toată dala (raza acoperă semidiagonala de ~12.4 m).
    "OBS_HOUSEPAD": 12.5,
}

# Nodurile care NU sunt obstacole (nu blochează plasarea goal-ului).
# Pinii fără DEF formează pădurea de graniță și sunt ignorați automat:
# tipul "Pine" nu e în PROTO_RADIUS → raza rămâne None → sărit.
NON_OBSTACLE_DEFS = {"Pioneer3AT", "GOAL_TARGET", "TERRAIN"}

# Numărul de sectoare LiDAR din starea RL (min-pooling uniform pe FOV-ul
# orizontal de 180°; 12 × 15°). Folosit și de baseline-ul din rl_scout.py.
N_LIDAR_SECTORS = 12
NON_OBSTACLE_TYPES = {
    "TexturedBackground", "TexturedBackgroundLight", "Viewpoint",
    "WorldInfo", "Robot", "Pioneer3at", "DirectionalLight", "PointLight",
    "SpotLight", "Fog", "UnevenTerrain", "Floor",
}


# ----------------------------------------------------------------------
# Modelul terenului neregulat — SURSĂ UNICĂ DE ADEVĂR.
# Aceleași funcții sunt importate de generate_world.py (care emite
# ElevationGrid-ul din world.wbt) și folosite aici pentru poziționarea
# robotului/goal-ului și validarea pantei → înălțimile pe care le „știe"
# Python sunt EXACT cele randate / cu coliziune din simulare.
# Harta: TEREN NEREGULAT pe toată suprafața (dealuri line, pitch/roll real),
# cu o zonă plată în centru pentru spawn. FĂRĂ rim/bazin/munți — marginea
# navigabilă e închisă de o PĂDURE DEASĂ de pini (vezi generate_world.py).
# ----------------------------------------------------------------------
MAP_SIZE = 150.0
HALF = MAP_SIZE / 2.0
GRID_DIM = 76
GRID_SPACING = MAP_SIZE / (GRID_DIM - 1)

R_START_FLAT = 10.0    # raza zonei plate de spawn (stabilitate la reset)

# ----------------------------------------------------------------------
# Contur navigabil NEREGULAT (poiană organică, NU cerc).
# Raza marginii navigabile variază lin cu unghiul (sumă de armonici joase)
# → o poiană cu formă organică. Pădurea de graniță (generate_world.border_ring)
# urmează EXACT acest contur, iar in_navigable / out_of_bounds / plasarea
# goal-ului îl folosesc → o singură sursă de adevăr pentru forma hărții.
# ----------------------------------------------------------------------
BOUNDARY_R0 = 50.0     # raza medie a poienii
BOUNDARY_AMP = 5.0     # cât de neregulat e conturul (rază în ~[R0-AMP, R0+AMP])
# Raza MAXIMĂ a poienii — păstrată pentru curriculum-ul din rl_scout și pentru
# intervalul de eșantionare al goal-ului (numele istoric "arena_limit").
ARENA_LIMIT = BOUNDARY_R0 + BOUNDARY_AMP


def boundary_r(angle):
    """Raza conturului navigabil la unghiul dat — formă organică netedă și
    deterministă (armonici de ordin mic). Min ≈ R0-AMP, max ≈ R0+AMP.
    Importată de generate_world.py: pădurea de graniță și fundalul de pini
    se plasează pe ACEASTĂ curbă → pădurea urmează forma suprafeței, nu un cerc."""
    return BOUNDARY_R0 + BOUNDARY_AMP * (
        0.60 * math.sin(3.0 * angle + 0.7) +
        0.25 * math.sin(5.0 * angle + 2.1) +
        0.15 * math.cos(2.0 * angle - 1.0))


HILL_AMP = 1.5         # amplitudinea dealurilor (teren neregulat, traversabil)

# ----------------------------------------------------------------------
# Platforme plate (graded pads) sub clădirile-landmark.
# Pe terenul neregulat, clădirile mari (amprentă mare) ar pluti/s-ar îngropa,
# fiindcă înălțimea variază pe sub amprenta lor. Soluție realistă (ca pe un
# șantier): nivelăm o platformă plată sub fiecare clădire. Pozițiile sunt
# DETERMINISTE și partajate cu generate_world.py (aceeași sursă de adevăr),
# de-aliniate (nu un inel regulat) → aspect nestructurat.
#   (proto, x, y, footprint_r, yaw_idx, defname)
# ----------------------------------------------------------------------
BUILDINGS = [
    ("Warehouse",  26.0, -10.0, 12.0, 3, "bld_warehouse"),
    ("Barn",      -24.0,  18.0,  7.5, 0, "bld_barn"),
]
PAD_FLAT_MARGIN = 1.5  # platforma e plată până la footprint_r + această margine
PAD_BLEND = 4.0        # lățimea inelului de racordare lină la terenul natural
_PADS = [(bx, by, fp) for (_p, bx, by, fp, _yi, _dn) in BUILDINGS]


def _hills(x, y):
    """Dealuri line din sumă de sinusoide multi-frecvență (determinist).
    Pante sub ~10° peste tot → teren neregulat, dar traversabil de Pioneer."""
    return (0.55 * math.sin(0.075 * x) * math.cos(0.065 * y) +
            0.30 * math.sin(0.130 * x + 1.3) * math.sin(0.110 * y) +
            0.20 * math.cos(0.170 * y + 0.5) +
            0.15 * math.sin(0.210 * x + 0.7) * math.cos(0.090 * y))


def _base_height(x, y):
    """Înălțimea dealurilor (fără platforme), plat în centru pentru spawn."""
    r = math.hypot(x, y)
    flat = max(0.0, min(1.0, (r - 4.0) / (R_START_FLAT - 4.0)))
    return _hills(x, y) * HILL_AMP * flat


def terrain_height(x, y):
    """Teren neregulat pe TOATĂ harta (dealuri line) + platforme plate sub
    clădiri. FĂRĂ rim/bazin/munți: marginea hărții e închisă de pădurea deasă,
    nu de munți → înălțimile rămân mici și traversabile. Platformele sunt
    plate în interior și se racordează lin (smoothstep) la dealuri la margine
    → clădirile stau flush, fără să plutească sau să se îngroape."""
    h = _base_height(x, y)
    for (px, py, fp) in _PADS:
        flat_r = fp + PAD_FLAT_MARGIN
        out_r = flat_r + PAD_BLEND
        d = math.hypot(x - px, y - py)
        if d < out_r:
            pad_z = _base_height(px, py)
            if d <= flat_r:
                s = 0.0
            else:
                t = (d - flat_r) / PAD_BLEND
                s = t * t * (3.0 - 2.0 * t)   # smoothstep 0→1
            h = pad_z * (1.0 - s) + h * s
    return h


def terrain_slope(x, y, d=0.6):
    """Magnitudinea gradientului local (diferențe finite)."""
    hx = (terrain_height(x + d, y) - terrain_height(x - d, y)) / (2.0 * d)
    hy = (terrain_height(x, y + d) - terrain_height(x, y - d)) / (2.0 * d)
    return math.hypot(hx, hy)


def in_navigable(x, y, margin=1.0):
    """True dacă (x, y) e în interiorul poienii navigabile (contur NEREGULAT
    boundary_r), la fel ca pădurea de graniță și terminarea out_of_bounds →
    fără ținte în pădure, indiferent de forma organică a marginii."""
    return math.hypot(x, y) < (boundary_r(math.atan2(y, x)) - margin)


def heightfield_grid():
    """Array-ul height[] pentru ElevationGrid în ordinea Webots:
    height[i + j*xDimension], i = index x (bucla interioară), j = index y
    (bucla exterioară). Terenul (Solid) e translatat cu (-HALF, -HALF, 0)
    → grila se centrează pe origine."""
    heights = []
    for j in range(GRID_DIM):
        y = j * GRID_SPACING - HALF
        for i in range(GRID_DIM):
            x = i * GRID_SPACING - HALF
            heights.append(terrain_height(x, y))
    return heights


class ScoutEnv:

    def __init__(self, robot, robot_node, motors, lidar,
                 imu, gps, gyro, accelerometer,
                 translation_field, rotation_field,
                 goal_translation_field,
                 timestep):

        self.robot = robot
        self.robot_node = robot_node
        self.motors = motors
        self.lidar = lidar
        self.imu = imu
        self.gps = gps
        self.gyro = gyro
        self.accelerometer = accelerometer
        self.translation_field = translation_field
        self.rotation_field = rotation_field
        self.goal_translation_field = goal_translation_field
        self.timestep = timestep
        self.dt = timestep / 1000.0

        # Parametri robot (Pioneer 3-AT)
        self.R = 0.0975
        self.L = 0.33
        self.MAX_WHEEL_SPEED = 6.4

        # Dinamica
        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0

        self.MAX_ACC_V = 1.5
        self.MAX_ACC_W = 4.0

        # Odometrie internă
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.theta = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gyro_rate = [0.0, 0.0, 0.0]
        self.acceleration = [0.0, 0.0, 0.0]

        # Goal
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.prev_distance = 0.0
        self.step_count = 0
        self.max_steps = 5000
        self.simulation_running = True
        self.last_state = None
        self.done_reason = None

        self.state_dim = N_LIDAR_SECTORS + 11   # 12 sectoare + 3 profil vertical + 8 navigație/IMU
        self.action_dim = 5

        # Goal-urile se plasează în poiană (boundary_r). Robotul, în schimb, e
        # mărginit de PĂDUREA tot mai deasă spre marginea hărții, impenetrabilă
        # la margine → terminarea out_of_bounds e doar plasa de siguranță la
        # marginea pătratului (±HALF), ca robotul să nu cadă de pe hartă.
        self.arena_limit = ARENA_LIMIT       # raza MAXIMĂ a poienii (curriculum/eșantionare)
        self.bounds_margin = 1.5             # față de marginea pătratului (HALF)
        self.goal_clearance = 1.5
        # Nodul GOAL_TARGET stă la nivelul solului; orbul-beacon plutește prin
        # offset-ul local al copiilor → înălțime constantă față de teren, fără
        # clipping pe pante (vezi geometria din generate_world.goal_block).
        self.goal_marker_z = 0.0
        self.goal_tolerance = 0.45
        self.max_goal_slope = 0.30

        # Praguri de stabilitate pe teren denivelat (IMU) — răsturnare la graniță
        self.max_safe_roll = 0.75
        self.max_safe_pitch = 0.75

        # Plasare start random (când e cerută): clearance mai mare decât la goal,
        # fiindcă robotul are gabarit fizic (nu e un punct).
        self.start_clearance = 1.4

        # Anti-blocaj: dacă robotul nu se deplasează mai mult de stuck_move_thresh
        # (m) în ultimii stuck_patience pași → e înțepenit/oscilează pe loc, iar
        # episodul se încheie cu „stuck" (taie episoadele care altfel mergeau
        # până la timeout). Ocolirile reale mișcă robotul > prag → nu sunt afectate.
        self.stuck_patience = 600
        self.stuck_move_thresh = 0.5
        self._stuck_anchor_x = 0.0
        self._stuck_anchor_y = 0.0
        self._stuck_anchor_step = 0

        # Obstacole citite dinamic din scenă (doar pentru plasarea goal-ului)
        self.current_map_obstacles = []
        self.terrain_enabled = True

    # ------------------------------------------------------------------
    # Transformarea coordonatelor (lume → cadrul robotului)
    # ------------------------------------------------------------------
    def world_to_robot_frame(self, wx, wy):
        """Distanța și unghiul punctului (wx, wy) în cadrul robotului.
        Echivalent cu o transformare omogenă 2D lume→robot urmată de
        conversia în coordonate polare (folosit pentru starea RL)."""
        dx = wx - self.x
        dy = wy - self.y
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) - self.theta
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return distance, angle

    def get_navigation_error(self):
        """Distanța și unghiul spre goal (folosit de baseline-ul reactiv)."""
        distance, angle = self.world_to_robot_frame(self.goal_x, self.goal_y)
        return distance, angle, "goal"

    # ------------------------------------------------------------------
    # Citirea dinamică a mediului din arborele scenei (Supervisor)
    # ------------------------------------------------------------------
    def scan_world_obstacles(self):
        """Parcurge nodurile de nivel înalt și extrage (x, y, r) pentru
        fiecare obstacol. Apelat la fiecare reset → Python rămâne mereu
        sincron cu world.wbt, indiferent ce obiecte adăugăm."""
        obstacles = []
        try:
            root = self.robot.getRoot()
            children = root.getField("children")
            count = children.getCount()
        except Exception:
            self.current_map_obstacles = obstacles
            return obstacles

        for i in range(count):
            try:
                node = children.getMFNode(i)
            except Exception:
                continue
            if node is None:
                continue

            type_name = node.getTypeName() or ""
            def_name = node.getDef() or ""

            if def_name in NON_OBSTACLE_DEFS or type_name in NON_OBSTACLE_TYPES:
                continue
            if def_name.startswith(("FLOOR", "Floor", "WALL", "Wall")):
                continue

            t_field = node.getField("translation")
            if t_field is None:
                continue
            try:
                pos = t_field.getSFVec3f()
            except Exception:
                continue

            radius = None
            if def_name.startswith("OBS_"):
                radius = DEF_RADIUS.get(def_name)
                if radius is None:
                    for prefix, pref_r in DEF_PREFIX_RADIUS.items():
                        if def_name.startswith(prefix):
                            radius = pref_r
                            break
                if radius is None:
                    radius = PROTO_RADIUS.get(type_name, DEFAULT_OBSTACLE_RADIUS)
            elif type_name in PROTO_RADIUS:
                radius = PROTO_RADIUS[type_name]
            if radius is None:
                continue

            obstacles.append({"x": pos[0], "y": pos[1], "r": radius})

        self.current_map_obstacles = obstacles
        return obstacles

    # ------------------------------------------------------------------
    # Înălțimea terenului (pentru marker goal / z de start)
    # ------------------------------------------------------------------
    def _surface_z_at(self, x, y):
        if not self.terrain_enabled:
            return 0.0
        return terrain_height(x, y)

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------
    def _clamp_to_arena(self, x, y):
        """Aduce (x, y) în interiorul poienii (contur neregulat boundary_r)."""
        a = math.atan2(y, x)
        lim = boundary_r(a) - self.goal_clearance
        r = math.hypot(x, y)
        if r > lim and r > 1e-6:
            s = lim / r
            x *= s
            y *= s
        return x, y

    def _goal_blocked(self, x, y):
        if not in_navigable(x, y, margin=self.goal_clearance):
            return True
        if terrain_slope(x, y) > self.max_goal_slope:
            return True
        for obs in self.current_map_obstacles:
            if math.hypot(x - obs["x"], y - obs["y"]) <= obs["r"] + self.goal_clearance:
                return True
        return False

    def _make_goal_safe(self, gx, gy):
        """Împinge un goal cerut de utilizator în zona navigabilă, ferit de obstacole."""
        margin = self.goal_clearance
        x, y = self._clamp_to_arena(gx, gy)
        adjusted = (x != gx) or (y != gy)

        for _ in range(25):
            moved = False
            for obs in self.current_map_obstacles:
                dx = x - obs["x"]
                dy = y - obs["y"]
                dist = math.hypot(dx, dy)
                min_dist = obs["r"] + margin
                if dist < min_dist:
                    ang = math.atan2(dy, dx) if dist > 1e-6 else random.uniform(-math.pi, math.pi)
                    x = obs["x"] + math.cos(ang) * (min_dist + 0.1)
                    y = obs["y"] + math.sin(ang) * (min_dist + 0.1)
                    moved = True
                    adjusted = True
            x, y = self._clamp_to_arena(x, y)
            if not moved:
                break
        return x, y, adjusted

    def _sample_free_goal(self, start_x, start_y, min_dist=2.0, max_dist=None):
        lim = self.arena_limit - self.goal_clearance
        for _ in range(600):
            gx = random.uniform(-lim, lim)
            gy = random.uniform(-lim, lim)
            if self._goal_blocked(gx, gy):
                continue
            d = math.hypot(gx - start_x, gy - start_y)
            if d < min_dist:
                continue
            if max_dist is not None and d > max_dist:
                continue
            return gx, gy
        # fallback: împinge un punct random în zona navigabilă, ferit de obstacole
        gx, gy, _ = self._make_goal_safe(
            random.uniform(-lim, lim), random.uniform(-lim, lim))
        return gx, gy

    def _sample_free_start(self, max_tries=400):
        """Alege o poziție de pornire random în poiană: navigabilă, pe pantă
        traversabilă și ferită de obstacole (ține cont de gabaritul robotului).
        Fallback: centrul (zona plată de spawn e mereu liberă)."""
        lim = self.arena_limit - self.start_clearance
        clear = self.start_clearance
        for _ in range(max_tries):
            sx = random.uniform(-lim, lim)
            sy = random.uniform(-lim, lim)
            if not in_navigable(sx, sy, margin=clear):
                continue
            if terrain_slope(sx, sy) > self.max_goal_slope:
                continue
            blocked = False
            for obs in self.current_map_obstacles:
                if math.hypot(sx - obs["x"], sy - obs["y"]) <= obs["r"] + clear:
                    blocked = True
                    break
            if not blocked:
                return sx, sy
        return 0.0, 0.0

    def set_goal(self, gx, gy):
        """Setează goal-ul din exterior (Tkinter / tastă G)."""
        requested_x = float(gx)
        requested_y = float(gy)
        safe_x, safe_y, adjusted = self._make_goal_safe(requested_x, requested_y)

        self.goal_x = safe_x
        self.goal_y = safe_y
        self._move_goal_marker()
        self.prev_distance, _ = self.world_to_robot_frame(self.goal_x, self.goal_y)

        if adjusted:
            print(
                f"[WARN] Goal ({requested_x:.2f}, {requested_y:.2f}) era prea aproape "
                f"de obstacol/perete. Ajustat la ({self.goal_x:.2f}, {self.goal_y:.2f})"
            )
        print(f"[ENV] Goal actualizat → ({self.goal_x:.2f}, {self.goal_y:.2f})")

    def _move_goal_marker(self):
        if self.goal_translation_field is not None:
            marker_z = self._surface_z_at(self.goal_x, self.goal_y) + self.goal_marker_z
            self.goal_translation_field.setSFVec3f([self.goal_x, self.goal_y, marker_z])

    # ------------------------------------------------------------------

    def reset(self, start_x=0.0, start_y=0.0, start_theta=0.0,
              random_goal=True, goal_max_dist=None, random_start=False):
        """Resetează episodul. goal_max_dist permite curriculum (ținte mai
        apropiate la început). random_start=True alege o poziție de pornire
        random navigabilă (altfel folosește start_x/start_y/start_theta date)."""

        for m in self.motors:
            m.setVelocity(0.0)

        # Obstacolele sunt statice → le citim înainte de a plasa robotul/goal-ul,
        # ca eșantionarea start-ului și a goal-ului random să le poată evita.
        self.scan_world_obstacles()

        if random_start:
            start_x, start_y = self._sample_free_start()
            start_theta = random.uniform(-math.pi, math.pi)

        # Roțile Pioneer ating solul când originea e la nivelul suprafeței
        # (ancoră roată z=0.11, rază 0.11 → contact la z=0). Mică gardă de 3 cm.
        start_z = self._surface_z_at(start_x, start_y) + 0.03
        self.translation_field.setSFVec3f([start_x, start_y, start_z])
        self.rotation_field.setSFRotation([0, 0, 1, start_theta])

        self.robot.simulationResetPhysics()
        self.simulation_running = True
        if self.robot.step(self.timestep) == -1:
            self.simulation_running = False

        self._update_odometry()

        self.v = 0.0
        self.w = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0
        self.step_count = 0
        self.done_reason = None

        # Ancoră anti-blocaj la poziția reală de pornire.
        self._stuck_anchor_x = self.x
        self._stuck_anchor_y = self.y
        self._stuck_anchor_step = 0

        if random_goal:
            gx, gy = self._sample_free_goal(start_x, start_y, max_dist=goal_max_dist)
            self.goal_x = gx
            self.goal_y = gy
            self._move_goal_marker()

        self.prev_distance, _ = self.world_to_robot_frame(self.goal_x, self.goal_y)

        self.last_state = self._get_state()
        return self.last_state

    # ------------------------------------------------------------------

    def step(self, action):

        if not self.simulation_running:
            self.done_reason = "simulation_stopped"
            return self.last_state, 0.0, True

        self.step_count += 1
        self._apply_action(action)

        if self.robot.step(self.timestep) == -1:
            self.simulation_running = False
            self.done_reason = "simulation_stopped"
            return self.last_state, 0.0, True

        self._update_odometry()

        state = self._get_state()
        self.last_state = state
        self.done_reason = None
        reward, done = self._compute_reward(state)

        if self.step_count >= self.max_steps:
            done = True
            self.done_reason = "timeout"

        return state, reward, done

    # ------------------------------------------------------------------

    def _apply_action(self, action):
        # 0 - forward, 1 - forward-left, 2 - forward-right,
        # 3 - rotate left, 4 - rotate right, -1 - reverse scurt (baseline)
        if action == -1:
            v_target, w_target = -0.25, 0.0
        elif action == 0:
            v_target, w_target = 0.55, 0.0
        elif action == 1:
            v_target, w_target = 0.45, 1.0
        elif action == 2:
            v_target, w_target = 0.45, -1.0
        elif action == 3:
            v_target, w_target = 0.0, 1.6
        elif action == 4:
            v_target, w_target = 0.0, -1.6
        else:
            v_target, w_target = 0.0, 0.0

        # Control diferențial cu limite dinamice de accelerație
        # (mapare comandă de nivel înalt (v, ω) → viteze de roți).
        dv = v_target - self.v_prev
        dv = max(min(dv, self.MAX_ACC_V * self.dt), -self.MAX_ACC_V * self.dt)
        self.v = self.v_prev + dv

        dw = w_target - self.w_prev
        dw = max(min(dw, self.MAX_ACC_W * self.dt), -self.MAX_ACC_W * self.dt)
        self.w = self.w_prev + dw

        self.v_prev = self.v
        self.w_prev = self.w

        v_left = (self.v - (self.L / 2.0) * self.w) / self.R
        v_right = (self.v + (self.L / 2.0) * self.w) / self.R

        v_left = max(min(v_left, self.MAX_WHEEL_SPEED), -self.MAX_WHEEL_SPEED)
        v_right = max(min(v_right, self.MAX_WHEEL_SPEED), -self.MAX_WHEEL_SPEED)

        # [FL, BL, FR, BR]
        self.motors[0].setVelocity(v_left)
        self.motors[1].setVelocity(v_left)
        self.motors[2].setVelocity(v_right)
        self.motors[3].setVelocity(v_right)

    # ------------------------------------------------------------------

    def _update_odometry(self):
        # Localizarea robotului: GPS pentru poziție, IMU pentru atitudine,
        # gyro/accelerometru pentru rate. Supervisor rămâne fallback.
        pos = self.gps.getValues() if self.gps is not None else self.robot_node.getPosition()
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

        if self.imu is not None:
            self.roll, self.pitch, self.yaw = self.imu.getRollPitchYaw()
            self.theta = self.yaw
        else:
            orientation = self.robot_node.getOrientation()
            self.theta = math.atan2(orientation[3], orientation[0])

        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        if self.gyro is not None:
            self.gyro_rate = self.gyro.getValues()
        if self.accelerometer is not None:
            self.acceleration = self.accelerometer.getValues()

    # ------------------------------------------------------------------
    # LiDAR + filtrare spațială prin convoluție 2D
    # ------------------------------------------------------------------
    def _lidar_dimensions(self, ranges):
        try:
            horizontal_resolution = self.lidar.getHorizontalResolution()
            layers = self.lidar.getNumberOfLayers()
        except AttributeError:
            horizontal_resolution = len(ranges)
            layers = 1

        if horizontal_resolution > 0 and len(ranges) >= horizontal_resolution:
            detected_layers = max(1, len(ranges) // horizontal_resolution)
            layers = min(max(1, layers), detected_layers)
        return max(1, horizontal_resolution), max(1, layers)

    @staticmethod
    def _conv2d_smooth(image):
        """Filtrare spațială (netezire) prin convoluție 2D separabilă cu
        nucleul [1,2,1]/4 pe ambele axe → reduce zgomotul senzorului LiDAR
        înainte de extragerea trăsăturilor. Margini tratate prin replicare."""
        kernel = np.array([1.0, 2.0, 1.0])
        kernel = kernel / kernel.sum()

        padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
        # convoluție pe orizontală
        tmp = (padded[:, :-2] * kernel[0] +
               padded[:, 1:-1] * kernel[1] +
               padded[:, 2:] * kernel[2])
        # convoluție pe verticală
        out = (tmp[:-2, :] * kernel[0] +
               tmp[1:-1, :] * kernel[1] +
               tmp[2:, :] * kernel[2])
        return out

    def _get_lidar_features(self):
        ranges = self.lidar.getRangeImage()
        max_range = 10.0
        n, layers = self._lidar_dimensions(ranges)

        # Construiește imaginea 2D (layers × n), curăță inf/nan, limitează.
        data = np.asarray(ranges, dtype=np.float32)
        data = data[: n * layers].reshape(layers, n) if data.size >= n * layers \
            else np.full((layers, n), max_range, dtype=np.float32)
        data = np.where(np.isfinite(data), data, max_range)
        data = np.clip(data, 0.0, max_range)

        smoothed = self._conv2d_smooth(data) if (layers >= 3 and n >= 3) else data

        lower = layers - 1
        upper = 0
        mid = layers // 2

        # 12 sectoare uniforme (min-pooling) care ACOPERĂ COMPLET FOV-ul
        # orizontal de 180°: trăsătura k = minimul distanțelor din sectorul
        # său de 15°. Vechile 5 sectoare frontale vedeau doar ±56° — zonele
        # laterale (±56°..±90°) erau OARBE, deci robotul „agăța cu umărul"
        # obstacole la viraje strânse printre copaci/moloz (o parte din cele
        # 23% coliziuni la evaluarea run 4). Min-pooling pe sector garantează
        # că cel mai apropiat obstacol din fiecare con intră în starea RL.
        row = smoothed[lower]
        ns = N_LIDAR_SECTORS
        sectors = [float(row[k * n // ns:(k + 1) * n // ns].min()) / max_range
                   for k in range(ns)]

        # Profil vertical pe conul frontal central (±7.5°): straturile de
        # sus/mijloc + diferența sus−jos (obstacol scund vs. perete înalt).
        cw = max(1, n // 24)
        lo, hi = n // 2 - cw, n // 2 + cw + 1
        front_low_raw = float(row[lo:hi].min())
        upper_front_raw = float(smoothed[upper, lo:hi].min())
        mid_front_raw = float(smoothed[mid, lo:hi].min())

        upper_front = upper_front_raw / max_range
        mid_front = mid_front_raw / max_range
        height_profile = max(-1.0, min(1.0, (upper_front_raw - front_low_raw) / max_range))

        return sectors, upper_front, mid_front, height_profile

    # ------------------------------------------------------------------

    def _get_state(self):
        sectors, upper_front, mid_front, height_profile = self._get_lidar_features()

        distance, angle = self.world_to_robot_frame(self.goal_x, self.goal_y)

        distance_norm = min(distance, 120.0) / 120.0
        angle_norm = angle / math.pi
        v_norm = self.v / 2.0
        w_norm = self.w / 4.0
        z_norm = max(0.0, min(self.z, 2.0)) / 2.0
        roll_norm = max(-1.0, min(1.0, self.roll / math.pi))
        pitch_norm = max(-1.0, min(1.0, self.pitch / math.pi))
        yaw_rate_norm = max(-1.0, min(1.0, self.gyro_rate[2] / 5.0))

        # Layout stare (23): [0:12] sectoare LiDAR, [12:15] profil vertical
        # frontal (sus/mijloc/diferență), [15] distanță goal, [16] unghi goal,
        # [17:19] v/ω, [19:23] z/roll/pitch/yaw-rate.
        return sectors + [
            upper_front, mid_front, height_profile,
            distance_norm, angle_norm,
            v_norm, w_norm,
            z_norm, roll_norm, pitch_norm, yaw_rate_norm
        ]

    # ------------------------------------------------------------------

    def _compute_reward(self, state):
        sectors = state[:N_LIDAR_SECTORS]
        angle_norm = state[N_LIDAR_SECTORS + 4]      # vezi layout-ul din _get_state

        # Conul frontal = cele 2 sectoare centrale (±15°); flancurile
        # apropiate = următoarele 2 pe fiecare parte (±15°..±45°).
        front = min(sectors[5], sectors[6])
        flank = min(sectors[3], sectors[4], sectors[7], sectors[8])

        reward = 0.0

        final_distance = math.hypot(self.goal_x - self.x, self.goal_y - self.y)

        # Goal atins
        if final_distance <= self.goal_tolerance:
            self.done_reason = "goal"
            return 200.0, True

        # Răsturnare pe teren denivelat → terminare
        if abs(self.roll) > self.max_safe_roll or abs(self.pitch) > self.max_safe_pitch:
            self.done_reason = "rollover"
            return -20.0, True

        # Ieșire de pe hartă: oprită fizic de pădurea deasă de la marginea
        # pătratului; aici e doar plasa de siguranță RL (robotul nu cade de pe hartă).
        if max(abs(self.x), abs(self.y)) > HALF - self.bounds_margin:
            self.done_reason = "out_of_bounds"
            return -20.0, True

        # 1. Progres spre goal — ȘAPING-UL PRINCIPAL și SINGURUL termen
        # pozitiv pe pas (potential-based shaping, Ng et al.): e anti-simetric,
        # deci orice buclă închisă (cercuri, oscilații) însumează exact 0 —
        # nefarmabil prin construcție.
        progress = self.prev_distance - final_distance
        reward += progress * 20.0

        # 2. Orientare spre goal — DOAR penalizare de dezaliniere (≤ 0).
        # Lecția run 3 (reward hacking): la γ=0.99, orice venit FLAT de r/pas
        # valorează r/(1−γ) = 100·r. Vechiul bonus de orientare (+1/pas)
        # valora ~100 — peste valoarea actualizată a unui goal îndepărtat
        # (~35) — deci politica „parca" robotul cu fața la goal (53% stuck,
        # opriri în câmp deschis). Nici condiționarea de progres nu e sigură
        # (cercuri mici dau progres pozitiv jumătate din timp și ocolesc
        # anti-stuck-ul), nici bonusul de spațiu liber (+0.2 ⇒ valoare ~20).
        # De aceea AMBELE bonusuri flat au fost ELIMINATE: toți termenii pe
        # pas sunt ≤ 0 în afara progresului → reward pozitiv cumulat cere
        # apropiere netă de goal.
        reward -= abs(angle_norm) * 0.05

        min_dist = min(sectors)

        # 3. Penalizări apropiere obstacol (din LiDAR)
        if front < 0.10:
            reward -= 1.0
        if front < 0.05:
            reward -= 2.0
        if flank < 0.05:
            reward -= 0.1

        # 4. Penalizare instabilitate (descurajează pantele riscante)
        reward -= (abs(self.roll) + abs(self.pitch)) * 0.05

        # 5. Living cost + penalizare rotație excesivă + cost de staționare
        # (descurajează activ statul pe loc/pivotarea prelungită; o
        # reorientare legitimă de 90° durează ~30 pași → cost ~0.6, neglijabil)
        reward -= 0.0005
        reward -= abs(self.w) * 0.002
        if self.v < 0.1:
            reward -= 0.02

        # 6. Coliziune (LiDAR foarte aproape)
        if min_dist < 0.025:
            self.done_reason = "collision"
            return -20.0, True

        # 7. Anti-blocaj: dacă robotul nu s-a îndepărtat mai mult de
        # stuck_move_thresh de ancoră în ultimii stuck_patience pași, e
        # înțepenit/oscilează pe loc → încheie episodul (evită „rămas blocat
        # până la reset"). Ocolirile reale mișcă robotul > prag → ancora se
        # mută și contorul se resetează, deci nu sunt penalizate.
        if math.hypot(self.x - self._stuck_anchor_x,
                      self.y - self._stuck_anchor_y) > self.stuck_move_thresh:
            self._stuck_anchor_x = self.x
            self._stuck_anchor_y = self.y
            self._stuck_anchor_step = self.step_count
        elif self.step_count - self._stuck_anchor_step >= self.stuck_patience:
            self.done_reason = "stuck"
            # −20 (ca la coliziune): la −5, blocajul era aproape gratuit față
            # de venitul acumulat stând pe loc — semnalul negativ trebuie să
            # se propage clar în valorile Q ale comportamentului de „parcare".
            return reward - 20.0, True

        self.prev_distance = final_distance
        return reward, False
