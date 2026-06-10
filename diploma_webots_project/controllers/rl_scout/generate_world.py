"""Generator offline pentru worlds/world.wbt.

Rulează în afara Webots (din venv-ul rl_env):
    python generate_world.py

Emite:
  * terenul ElevationGrid cu înălțimi din scout_env.terrain_height (ACEEAȘI
    sursă de adevăr ca RL-ul → Python „știe" exact suprafața), incl. platforme
    plate (graded pads) sub clădiri ca să nu plutească/se îngroape;
  * un PEISAJ NESTRUCTURAT: câteva landmark-uri (case, fermă, depozit, benzinărie)
    DE-ALINIATE pe platforme plate, peste un teren neregulat (dealuri line), cu
    copaci, stânci, bușteni, butoaie și lăzi risipiți (resturi off-road);
  * o graniță naturală: PĂDURE DEASĂ de pini (Pine) pe conturul NEREGULAT al
    poienii (urmează forma suprafeței, NU un cerc), plus pini de fundal care
    umplu exteriorul (fără gazon gol) — fără munți/stânci la margine;
  * robotul DEF Pioneer3AT (LiDAR/IMU/GPS/Gyro/Accelerometer) și DEF GOAL_TARGET.

Re-rulează după ce modifici terenul/conturul în scout_env.py.
"""
import math
import os
import random

import scout_env as S

BASE = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"

# PROTO-uri verificate că există în arborele R2025a (schema GitHub).
PROTOS = {
    "TexturedBackground": "objects/backgrounds/protos/TexturedBackground.proto",
    "TexturedBackgroundLight": "objects/backgrounds/protos/TexturedBackgroundLight.proto",
    "Grass": "appearances/protos/Grass.proto",
    "Pine": "objects/trees/protos/Pine.proto",
    "Oak": "objects/trees/protos/Oak.proto",
    "Cypress": "objects/trees/protos/Cypress.proto",
    "SimpleTree": "objects/trees/protos/SimpleTree.proto",
    "Rock": "objects/rocks/protos/Rock.proto",
    "SimpleTwoFloorsHouse": "objects/buildings/protos/SimpleTwoFloorsHouse.proto",
    "ModernHouse": "objects/buildings/protos/ModernHouse.proto",
    "HouseWithGarage": "objects/buildings/protos/HouseWithGarage.proto",
    "Barn": "objects/buildings/protos/Barn.proto",
    "Warehouse": "objects/buildings/protos/Warehouse.proto",
    "GasStation": "objects/buildings/protos/GasStation.proto",
    "ToyotaPrius": "vehicles/protos/toyota/ToyotaPrius.proto",
    "CitroenCZero": "vehicles/protos/citroen/CitroenCZero.proto",
    "BmwX5": "vehicles/protos/bmw/BmwX5.proto",
    "MercedesBenzSprinter": "vehicles/protos/mercedes_benz/MercedesBenzSprinter.proto",
    "StreetLight": "objects/traffic/protos/StreetLight.proto",
    "FireHydrant": "objects/street_furniture/protos/FireHydrant.proto",
    "TrafficCone": "objects/traffic/protos/TrafficCone.proto",
    "OilBarrel": "objects/obstacles/protos/OilBarrel.proto",
}

HEADER_ORDER = [
    "TexturedBackground", "TexturedBackgroundLight", "Grass",
    "Pine", "Oak", "Cypress", "SimpleTree", "Rock",
    "SimpleTwoFloorsHouse", "ModernHouse", "HouseWithGarage",
    "Barn", "Warehouse", "GasStation",
    "ToyotaPrius", "CitroenCZero", "BmwX5", "MercedesBenzSprinter",
    "StreetLight", "FireHydrant", "TrafficCone", "OilBarrel",
]

SEED = 12
START_CLEAR_R = 13.0    # nimic în zona de spawn (robotul pleacă din centru)
SPACING = 1.5           # margine suplimentară între gabaritele obiectelor
ARENA = S.ARENA_LIMIT   # raza MAXIMĂ a poienii (interval de eșantionare scatter)

# Pădure cu densitate CRESCĂTOARE spre marginea hărții (FĂRĂ inel în jurul
# poienii). Lângă poiană copaci răsfirați; spre marginea pătratului tot mai
# deși; la margine un brâu fin, impenetrabil (robotul nu poate cădea de pe hartă).
FOREST_CLEAR = 2.5      # obiectele interioare stau cel puțin atât în interiorul poienii
FOREST_GAP = 1.5        # pădurea începe la boundary_r + atât (poiana rămâne deschisă)
FOREST_GRID = 3.2       # grila pădurii-gradient (mai mare = mai puțini copaci = mai ușor)
FOREST_TAU = 11.0       # scara exponențială a densității (mic → crește mai repede spre margine)
EDGE_BELT_W = 2.4       # lățimea brâului dens de la marginea pătratului
EDGE_BELT_GRID = 1.2    # pas fin în brâu (< 1.3 m → impenetrabil pentru Pioneer ~0.5 m)
EDGE_MARGIN = 1.0       # brâul stă cu atât în interiorul marginii pătratului (±HALF)

ORTHO = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]

_placed = []            # (x, y, r) — evitarea suprapunerilor în interior
_uid_counter = [0]


def _uid():
    """Index global unic pentru DEF OBS_* (evită nume DEF duplicate → eroare Webots)."""
    _uid_counter[0] += 1
    return _uid_counter[0]


# ----------------------------------------------------------------------
# Plasare
# ----------------------------------------------------------------------
def _free(x, y, r):
    """True dacă (x, y) e liber: în afara zonei de spawn, în interiorul poienii
    (sub conturul neregulat boundary_r, ferit de pădure) și fără suprapuneri."""
    rr = math.hypot(x, y)
    if rr < START_CLEAR_R + r:
        return False
    if rr > S.boundary_r(math.atan2(y, x)) - (r + FOREST_CLEAR):
        return False
    for (px, py, pr) in _placed:
        if math.hypot(x - px, y - py) < r + pr + SPACING:
            return False
    return True


def _z(x, y, off=0.0):
    return S.terrain_height(x, y) + off


def node(proto, x, y, rng, name=None, defname=None, extra="", zoff=0.0, yaw=None):
    """Emite un nod PROTO așezat pe suprafața terenului (z = teren + zoff)."""
    if yaw is None:
        yaw = rng.uniform(-math.pi, math.pi)
    head = f"DEF {defname} {proto}" if defname else proto
    s = f"{head} {{\n"
    s += f"  translation {x:.3f} {y:.3f} {_z(x, y, zoff):.3f}\n"
    s += f"  rotation 0 0 1 {yaw:.4f}\n"
    if name:
        s += f'  name "{name}"\n'
    s += extra
    s += "}\n"
    return s


def place(parts, proto, ax, ay, r, rng, jitter=6.0, tries=140,
          name=None, defname=None, extra="", zoff=0.0, yaw=None, max_slope=None):
    """Caută o poziție liberă lângă ancora (ax, ay) și adaugă nodul în `parts`."""
    for _ in range(tries):
        x = ax + rng.uniform(-jitter, jitter)
        y = ay + rng.uniform(-jitter, jitter)
        if not _free(x, y, r):
            continue
        if max_slope is not None and S.terrain_slope(x, y) > max_slope:
            continue
        _placed.append((x, y, r))
        parts.append(node(proto, x, y, rng, name=name, defname=defname,
                          extra=extra, zoff=zoff, yaw=yaw))
        return (x, y)
    return None


# ----------------------------------------------------------------------
# Obstacole „custom" (Solid) — z controlat exact, primesc DEF OBS_* ca să fie
# văzute de scout_env.scan_world_obstacles (plasarea sigură a goal-ului).
# ----------------------------------------------------------------------
def container_node(x, y, rng, i):
    """Container industrial (Box) — obstacol DEF OBS_CONTAINER#."""
    yaw = rng.uniform(-math.pi, math.pi)
    z = _z(x, y)
    L = rng.uniform(2.4, 3.2)
    W, H = 2.0, 2.2
    col = rng.choice(["0.55 0.27 0.18", "0.25 0.42 0.5", "0.5 0.5 0.32"])
    return (
        f"DEF OBS_CONTAINER{i} Solid {{\n"
        f"  translation {x:.3f} {y:.3f} {z + H / 2.0:.3f}\n"
        f"  rotation 0 0 1 {yaw:.4f}\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {col} roughness 0.85 metalness 0 }}\n"
        f"      geometry Box {{ size {L:.2f} {W:.2f} {H:.2f} }}\n"
        f"    }}\n"
        f"  ]\n"
        f'  name "container_{i}"\n'
        f"  boundingObject Box {{ size {L:.2f} {W:.2f} {H:.2f} }}\n"
        f"}}\n"
    )


def crate_node(x, y, rng, i):
    """Ladă de lemn (Box) — obstacol DEF OBS_CRATE#, așezată pe sol (z = teren + h/2)."""
    yaw = rng.uniform(-math.pi, math.pi)
    s = rng.uniform(0.5, 0.95)
    col = rng.choice(["0.45 0.30 0.16", "0.40 0.26 0.14", "0.50 0.36 0.20"])
    return (
        f"DEF OBS_CRATE{i} Solid {{\n"
        f"  translation {x:.3f} {y:.3f} {_z(x, y) + s / 2.0:.3f}\n"
        f"  rotation 0 0 1 {yaw:.4f}\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {col} roughness 0.95 metalness 0 }}\n"
        f"      geometry Box {{ size {s:.2f} {s:.2f} {s:.2f} }}\n"
        f"    }}\n"
        f"  ]\n"
        f'  name "crate_{i}"\n'
        f"  boundingObject Box {{ size {s:.2f} {s:.2f} {s:.2f} }}\n"
        f"}}\n"
    )


def log_node(x, y, rng, i):
    """Buștean căzut (Cylinder culcat) — obstacol DEF OBS_LOG#. În Webots axa
    cilindrului e Y (orizontală în lumea Z-up) → trunchiul stă lungit; rotația
    pe Z îi dă direcția, iar z = teren + rază îl așază pe sol."""
    yaw = rng.uniform(-math.pi, math.pi)
    L = rng.uniform(2.5, 4.5)
    rad = rng.uniform(0.22, 0.34)
    return (
        f"DEF OBS_LOG{i} Solid {{\n"
        f"  translation {x:.3f} {y:.3f} {_z(x, y) + rad:.3f}\n"
        f"  rotation 0 0 1 {yaw:.4f}\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.36 0.24 0.13 roughness 1 metalness 0 }}\n"
        f"      geometry Cylinder {{ height {L:.2f} radius {rad:.2f} }}\n"
        f"    }}\n"
        f"  ]\n"
        f'  name "log_{i}"\n'
        f"  boundingObject Cylinder {{ height {L:.2f} radius {rad:.2f} }}\n"
        f"}}\n"
    )


# ----------------------------------------------------------------------
# Împrăștiere pe interiorul navigabil (peisaj natural nestructurat)
# ----------------------------------------------------------------------
def scatter_proto(parts, proto, n, r, rng, name_prefix, defname_prefix=None,
                  zoff=0.0, scale_range=None, max_slope=None):
    """Împrăștie n noduri PROTO pe tot interiorul poienii. Pentru obiecte cu
    origine la centru (Rock, OilBarrel) `zoff` ridică nodul ca să stea pe sol;
    cu `scale_range` zoff e scalat odată cu obiectul (zoff·scale)."""
    for i in range(n):
        for _ in range(200):
            x = rng.uniform(-ARENA, ARENA)
            y = rng.uniform(-ARENA, ARENA)
            if not _free(x, y, r):
                continue
            if max_slope is not None and S.terrain_slope(x, y) > max_slope:
                continue
            _placed.append((x, y, r))
            extra, zo = "", zoff
            if scale_range is not None:
                sc = rng.uniform(*scale_range)
                extra = f"  scale {sc:.2f}\n"
                zo = zoff * sc
            defn = f"{defname_prefix}{_uid()}" if defname_prefix else None
            parts.append(node(proto, x, y, rng, name=f"{name_prefix}_{i}",
                              defname=defn, extra=extra, zoff=zo))
            break


def scatter_custom(parts, node_fn, n, r, rng, max_slope=None):
    """Împrăștie n obstacole „custom" (node_fn(x, y, rng, i) → Solid cu DEF OBS_*)."""
    for i in range(n):
        for _ in range(200):
            x = rng.uniform(-ARENA, ARENA)
            y = rng.uniform(-ARENA, ARENA)
            if not _free(x, y, r):
                continue
            if max_slope is not None and S.terrain_slope(x, y) > max_slope:
                continue
            _placed.append((x, y, r))
            parts.append(node_fn(x, y, rng, _uid()))
            break


def _near(anchor, gap, k=0):
    """Punct lângă o clădire, deplasat spre centrul hărții (rămâne în poiană)."""
    ax, ay, fp = anchor
    base = math.atan2(ay, ax) + math.pi          # spre centru
    ang = base + 0.6 * k
    d = fp + gap
    return ax + d * math.cos(ang), ay + d * math.sin(ang)


def forest(rng):
    """Pădure cu densitate CRESCĂTOARE spre marginea hărții — FĂRĂ inel în jurul
    poienii. Copacii încep răsfirați chiar lângă poiană (conturul neregulat
    boundary_r) și se îndesesc spre marginea pătratului; la margine un brâu fin,
    dens, impenetrabil (robotul nu poate cădea de pe hartă). Pinii NU primesc
    DEF/tip cunoscut → nu intră în lista de obstacole pentru goal."""
    out = []
    idx = 0
    edge = S.HALF - EDGE_MARGIN

    # 1) Pădure-gradient: probabilitate de plasare exp(-edge_dist/TAU) → rară
    #    lângă poiană, tot mai deasă spre marginea pătratului. edge_dist =
    #    distanța până la cea mai apropiată latură a pătratului.
    g = FOREST_GRID
    gx = -edge
    while gx <= edge:
        gy = -edge
        while gy <= edge:
            x = gx + rng.uniform(-0.45, 0.45) * g
            y = gy + rng.uniform(-0.45, 0.45) * g
            if math.hypot(x, y) > S.boundary_r(math.atan2(y, x)) + FOREST_GAP:
                edge_dist = S.HALF - max(abs(x), abs(y))
                if rng.random() < math.exp(-edge_dist / FOREST_TAU):
                    out.append(node("Pine", x, y, rng, name=f"pine_{idx}"))
                    idx += 1
            gy += g
        gx += g

    # 2) Brâu fin, dens, impenetrabil pe tot perimetrul pătratului (grilă plină
    #    cu pas < 1.3 m → discurile de coliziune se suprapun → robotul nu trece).
    gb = EDGE_BELT_GRID
    bx = -edge
    while bx <= edge:
        by = -edge
        while by <= edge:
            if S.HALF - max(abs(bx), abs(by)) <= EDGE_BELT_W:
                x = bx + rng.uniform(-0.06, 0.06)
                y = by + rng.uniform(-0.06, 0.06)
                out.append(node("Pine", x, y, rng, name=f"pine_{idx}"))
                idx += 1
            by += gb
        bx += gb
    return out


# ----------------------------------------------------------------------
# Zone INACCESIBILE — clustere dens împachetate (impenetrabile), pe care
# robotul trebuie să le ocolească. Fiecare obstacol primește DEF OBS_* (cu
# index global unic) → e văzut de scanner, deci goal-ul nu cade în interior.
# ----------------------------------------------------------------------
def _on_building(x, y, margin=1.0):
    for (proto, bx, by, fp, yi, dn) in S.BUILDINGS:
        if math.hypot(x - bx, y - by) < fp + margin:
            return True
    return False


def _interior_ok(x, y, bmargin=1.0):
    """Punct în poiană, în afara zonei de spawn și a clădirilor."""
    rr = math.hypot(x, y)
    if rr < START_CLEAR_R - 2.0:
        return False
    if rr > S.boundary_r(math.atan2(y, x)) - FOREST_CLEAR:
        return False
    return not _on_building(x, y, bmargin)


def boulder_field(parts, rng, cx, cy, R):
    """Câmp de bolovani împachetați (stânci mari suprapuse) → impenetrabil."""
    step = 1.3
    gx = cx - R
    while gx <= cx + R:
        gy = cy - R
        while gy <= cy + R:
            x, y = gx + rng.uniform(-0.25, 0.25), gy + rng.uniform(-0.25, 0.25)
            if math.hypot(x - cx, y - cy) <= R and _interior_ok(x, y):
                u = _uid()
                sc = rng.uniform(7.0, 11.0)
                _placed.append((x, y, 0.07 * sc))
                parts.append(node("Rock", x, y, rng, name=f"boulder_{u}",
                                  defname=f"OBS_BOULDER{u}",
                                  extra=f"  scale {sc:.2f}\n", zoff=0.05 * sc))
            gy += step
        gx += step


def thicket(parts, rng, cx, cy, R):
    """Desiș de pini împachetați (pas < 1.3 m → coliziuni suprapuse) → impenetrabil."""
    step = 1.0
    gx = cx - R
    while gx <= cx + R:
        gy = cy - R
        while gy <= cy + R:
            x, y = gx + rng.uniform(-0.18, 0.18), gy + rng.uniform(-0.18, 0.18)
            if math.hypot(x - cx, y - cy) <= R and _interior_ok(x, y):
                u = _uid()
                _placed.append((x, y, 0.4))
                parts.append(node("Pine", x, y, rng, name=f"thkt_{u}", defname=f"OBS_THKT{u}"))
            gy += step
        gx += step


def container_cluster(parts, rng, cx, cy):
    """Grămadă compactă de containere suprapuse (bloc solid) + butoaie în jur →
    zonă inaccesibilă (nu un simplu zid: interiorul e blocat)."""
    offs = [(-1.2, -1.1), (1.2, -1.1), (0.0, 1.2), (-1.3, 1.1), (1.3, 1.1)]
    for (ox, oy) in offs:
        x, y = cx + ox, cy + oy
        if _interior_ok(x, y, bmargin=1.5):
            _placed.append((x, y, 1.6))
            parts.append(container_node(x, y, rng, _uid()))
    for _ in range(4):
        x, y = cx + rng.uniform(-4, 4), cy + rng.uniform(-4, 4)
        if _interior_ok(x, y) and _free(x, y, 0.5):
            u = _uid()
            _placed.append((x, y, 0.5))
            parts.append(node("OilBarrel", x, y, rng, name=f"bar_{u}",
                              defname=f"OBS_BARREL{u}", zoff=0.44))


def rubble_pile(parts, rng, cx, cy, R):
    """Grămadă de moloz: lăzi/blocuri împachetate → impenetrabilă (DEF OBS_RUBBLE)."""
    step = 1.0
    gx = cx - R
    while gx <= cx + R:
        gy = cy - R
        while gy <= cy + R:
            x, y = gx + rng.uniform(-0.2, 0.2), gy + rng.uniform(-0.2, 0.2)
            if math.hypot(x - cx, y - cy) <= R and _interior_ok(x, y):
                u = _uid()
                s = rng.uniform(0.6, 1.05)
                col = rng.choice(["0.5 0.5 0.5", "0.45 0.4 0.35", "0.4 0.42 0.45"])
                _placed.append((x, y, s * 0.7))
                parts.append(
                    f"DEF OBS_RUBBLE{u} Solid {{\n"
                    f"  translation {x:.3f} {y:.3f} {_z(x, y) + s / 2.0:.3f}\n"
                    f"  rotation 0 0 1 {rng.uniform(-math.pi, math.pi):.4f}\n"
                    f"  children [\n"
                    f"    Shape {{\n"
                    f"      appearance PBRAppearance {{ baseColor {col} roughness 1 metalness 0 }}\n"
                    f"      geometry Box {{ size {s:.2f} {s:.2f} {s:.2f} }}\n"
                    f"    }}\n"
                    f"  ]\n"
                    f'  name "rubble_{u}"\n'
                    f"  boundingObject Box {{ size {s:.2f} {s:.2f} {s:.2f} }}\n"
                    f"}}\n")
            gy += step
        gx += step


def build(rng):
    parts = []

    # === Puține structuri (depozit + hambar) — NU sat; restul e teren
    #     nestructurat. Stau pe platforme plate (terenul e nivelat sub ele). ===
    anc = {}
    for (proto, bx, by, fp, yi, dn) in S.BUILDINGS:
        _placed.append((bx, by, fp))
        parts.append(node(proto, bx, by, rng, name=dn, yaw=ORTHO[yi % 4]))
        anc[dn] = (bx, by, fp)

    # Vehicule abandonate + un stâlp/hidrant lângă structuri (fizică → se așază)
    place(parts, "MercedesBenzSprinter", *_near(anc["bld_warehouse"], 6.5, 1), 3.2, rng,
          jitter=2.5, name="sprinter_0", zoff=0.1, max_slope=0.14)
    place(parts, "ToyotaPrius", *_near(anc["bld_warehouse"], 6.0, -2), 2.4, rng,
          jitter=2.5, name="car_0", zoff=0.1, max_slope=0.14)
    place(parts, "CitroenCZero", *_near(anc["bld_barn"], 5.5, 1), 1.8, rng,
          jitter=2.5, name="car_1", zoff=0.1, max_slope=0.14)
    place(parts, "StreetLight", *_near(anc["bld_warehouse"], 8.0, 2), 0.5, rng,
          jitter=2, name="light_0", yaw=0.0)
    place(parts, "FireHydrant", *_near(anc["bld_barn"], 5.0, -2), 0.5, rng,
          jitter=2, name="hydrant_0")

    # === ZONE INACCESIBILE (impenetrabile — robotul le ocolește) ===
    boulder_field(parts, rng, -32.0, -16.0, 4.5)
    boulder_field(parts, rng, 20.0, 26.0, 4.0)
    thicket(parts, rng, 34.0, 12.0, 4.0)
    container_cluster(parts, rng, -2.0, -36.0)
    rubble_pile(parts, rng, 4.0, 34.0, 3.5)

    # === Obstacole risipite, off-road (mai dense ca înainte) ===
    # OilBarrel: origine la centru → zoff = height/2 = 0.44 ca să stea pe sol.
    scatter_proto(parts, "OilBarrel", 14, 0.5, rng, "barrel",
                  defname_prefix="OBS_BARREL", zoff=0.44)
    scatter_custom(parts, crate_node, 16, 0.7, rng)
    scatter_custom(parts, container_node, 5, 2.0, rng)
    scatter_custom(parts, log_node, 16, 2.2, rng)
    scatter_proto(parts, "Oak", 14, S.PROTO_RADIUS["Oak"], rng, "oak")
    scatter_proto(parts, "Cypress", 10, S.PROTO_RADIUS["Cypress"], rng, "cypress")
    scatter_proto(parts, "SimpleTree", 12, S.PROTO_RADIUS["SimpleTree"], rng, "stree")
    scatter_proto(parts, "Rock", 26, 1.2, rng, "rock",
                  defname_prefix="OBS_ROCK", zoff=0.05, scale_range=(3.0, 7.0))

    # === Pădure: densitate crescătoare spre marginea hărții (fără inel) ===
    trees = forest(rng)
    parts += trees
    return parts, len(trees)


# ----------------------------------------------------------------------
# Blocuri fixe (teren, robot, goal)
# ----------------------------------------------------------------------
def terrain_block():
    h = S.heightfield_grid()
    sp = S.GRID_SPACING
    lines = []
    for k in range(0, len(h), 12):
        lines.append("          " + " ".join(f"{v:.3f}" for v in h[k:k + 12]))
    height_str = "\n".join(lines)
    return (
        f"DEF TERRAIN Solid {{\n"
        f"  translation {-S.HALF:.3f} {-S.HALF:.3f} 0\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance Grass {{\n"
        f"        textureTransform TextureTransform {{ scale 80 80 }}\n"
        f"      }}\n"
        f"      geometry DEF TERRAIN_EG ElevationGrid {{\n"
        f"        xDimension {S.GRID_DIM}\n"
        f"        xSpacing {sp:.5f}\n"
        f"        yDimension {S.GRID_DIM}\n"
        f"        ySpacing {sp:.5f}\n"
        f"        thickness 2\n"
        f"        height [\n{height_str}\n        ]\n"
        f"      }}\n"
        f"    }}\n"
        f"  ]\n"
        f'  name "terrain"\n'
        f"  boundingObject USE TERRAIN_EG\n"
        f"}}\n"
    )


def robot_block():
    # Roțile ating solul când originea e la nivelul suprafeței (ancoră roată
    # z=0.11, rază 0.11 → contact la z=0). Gardă mică de 3 cm, nu plutește.
    z = S.terrain_height(0.0, 0.0) + 0.03
    return (
        f"DEF Pioneer3AT Pioneer3at {{\n"
        f"  translation 0 0 {z:.3f}\n"
        f'  controller "rl_scout"\n'
        f"  supervisor TRUE\n"
        f"  extensionSlot [\n"
        f"    Lidar {{\n"
        f"      translation 0.2 0 0.3\n"
        f'      name "lidar"\n'
        f"      horizontalResolution 720\n"
        f"      fieldOfView 3.14159\n"
        f"      numberOfLayers 4\n"
        f"      verticalFieldOfView 0.3\n"
        f"      minRange 0.15\n"
        f"      maxRange 10\n"
        f"    }}\n"
        f'    InertialUnit {{ name "imu" }}\n'
        f'    GPS {{ name "gps" }}\n'
        f'    Gyro {{ name "gyro" }}\n'
        f'    Accelerometer {{ name "accelerometer" }}\n'
        f"  ]\n"
        f"}}\n"
    )


def goal_block():
    # BEACON plutitor: nodul stă la sol (terrain_height), copiii plutesc → orb la
    # înălțime CONSTANTĂ deasupra punctului, independent de teren (fără clipping
    # pe pante) și deasupra conului LiDAR (agentul tot navighează din stare).
    gx, gy = 7.0, 0.0
    z = S.terrain_height(gx, gy)
    return (
        f"DEF GOAL_TARGET Solid {{\n"
        f"  translation {gx:.3f} {gy:.3f} {z:.3f}\n"
        f"  rotation 0 0 1 0\n"
        f"  children [\n"
        f"    Pose {{\n"                                    # rază subțire spre sol
        f"      translation 0 0 0.60\n"
        f"      rotation 1 0 0 1.5708\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 1 0.2 0.15 emissiveColor 0.9 0.15 0.1 transparency 0.3 roughness 1 metalness 0 }}\n"
        f"          geometry Cylinder {{ radius 0.03 height 1.10 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"    Pose {{\n"                                    # orbul plutitor
        f"      translation 0 0 1.20\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 1 0.1 0.1 emissiveColor 0.95 0.1 0.05 roughness 1 metalness 0 }}\n"
        f"          geometry Sphere {{ radius 0.30 subdivision 3 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f'  name "goal_target"\n'
        f"}}\n"
    )


def world_text(seed=SEED):
    rng = random.Random(seed)
    objects, n_forest = build(rng)

    externs = ['#VRML_SIM R2025a utf8', ""]
    for key in HEADER_ORDER:
        externs.append(f'EXTERNPROTO "{BASE}/{PROTOS[key]}"')
    externs.append('EXTERNPROTO "../protos/Pioneer3at.proto"')
    header = "\n".join(externs) + "\n\n"

    world_info = (
        "WorldInfo {\n"
        "  basicTimeStep 32\n"
        "  contactProperties [\n"
        "    ContactProperties {\n"
        "      coulombFriction [ 1.2 ]\n"
        "      bounce 0.0\n"
        "      softCFM 0.0003\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    viewpoint = (
        "Viewpoint {\n"
        "  orientation -0.42 0.40 0.81 1.84\n"
        "  position 95 -105 78\n"
        "}\n"
    )
    lights = "TexturedBackground {\n}\nTexturedBackgroundLight {\n}\n"

    body = (header + world_info + viewpoint + lights +
            terrain_block() + robot_block() + goal_block() +
            "".join(objects))
    return body, len(objects), n_forest


def _validate(text, n_objects, n_forest):
    ok = True
    if text.count("{") != text.count("}"):
        print(f"[X] Acolade dezechilibrate: {{={text.count('{')} }}={text.count('}')}")
        ok = False
    if text.count("[") != text.count("]"):
        print(f"[X] Paranteze drepte: [={text.count('[')} ]={text.count(']')}")
        ok = False
    for token in ("DEF Pioneer3AT", "DEF GOAL_TARGET", "DEF TERRAIN", "USE TERRAIN_EG",
                  'name "lidar"', 'name "imu"', 'name "gps"', 'name "gyro"',
                  'name "accelerometer"'):
        if token not in text:
            print(f"[X] Lipsește: {token}")
            ok = False
    print(f"[i] Obiecte plasate: {n_objects} "
          f"(pădure: {n_forest}, interior: {n_objects - n_forest})")
    print(f"[i] Varfuri teren: {S.GRID_DIM * S.GRID_DIM} ({S.GRID_DIM}x{S.GRID_DIM})")
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Generator de lume. Cu --seed/--out diferite produce un "
                    "layout NOU de obstacole pe ACELAȘI teren (testul de "
                    "generalizare pe hartă nevăzută).")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"seed-ul layout-ului (implicit {SEED})")
    ap.add_argument("--out", default="world.wbt",
                    help="numele fișierului scris în ../../worlds/")
    ap.add_argument("--force", action="store_true",
                    help="permite suprascrierea lui world.wbt (PERSONALIZAT MANUAL!)")
    args = ap.parse_args()

    out_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "worlds", args.out))

    # world.wbt e personalizat manual în Webots (case, moară, zonă de moloz...).
    # Suprascrierea lui pierde tot — cere --force explicit.
    if os.path.basename(out_path) == "world.wbt" and os.path.exists(out_path) \
            and not args.force:
        print("[STOP] worlds/world.wbt există și e PERSONALIZAT MANUAL — nu îl suprascriu.")
        print("       Pentru harta de generalizare: python generate_world.py --seed 77 --out world_eval.wbt")
        print("       Dacă chiar vrei să-l regenerezi: adaugă --force (fă întâi backup!).")
        return

    text, n_objects, n_forest = world_text(args.seed)
    ok = _validate(text, n_objects, n_forest)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"[{'OK' if ok else 'WARN'}] {args.out} (seed {args.seed}) scris in {out_path}")
    if not ok:
        print("  ! Validarea a semnalat probleme - verifica mai sus.")


if __name__ == "__main__":
    main()
