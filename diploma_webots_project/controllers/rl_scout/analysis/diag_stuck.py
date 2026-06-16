"""Diagnoza episoadelor STUCK din evaluare: unde se opreste robotul?
(linga obstacol / in camp deschis / linga goal) — temporar."""

import os
import sys

# Bootstrap cale: ruleaza din orice director. Adauga folderul controllerului
# (parintele) la sys.path (pentru `import scout_env` / `dqn_agent`) si fixeaza
# cwd acolo (pentru path-urile relative logs/ si runs/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import csv
import math
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # "", "_center" sau "_baseline"

def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

obstacles = [(float(r["x"]), float(r["y"]), float(r["r"])) for r in load(r"logs\obstacles.csv")]
summary = load(rf"logs\eval_summary{TAG}.csv")

print(f"{'ep':>3} {'reason':<10} {'d_goal_END':>10} {'d_goal_MIN':>10} {'depl.600':>9} {'obst_apr':>9} {'pasi':>5}")
for i in range(20):
    rows = load(rf"logs\eval_traj{TAG}_{i}.csv")
    if not rows:
        continue
    reason = summary[i]["done_reason"]
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    gx, gy = float(rows[-1]["goal_x"]), float(rows[-1]["goal_y"])
    dists = [math.hypot(gx - x, gy - y) for x, y in zip(xs, ys)]
    d_end = dists[-1]
    d_min = min(dists)
    # deplasarea in ultimii 600 pasi (fereastra anti-stuck)
    k = max(0, len(xs) - 600)
    depl = math.hypot(xs[-1] - xs[k], ys[-1] - ys[k])
    # cel mai apropiat obstacol (margine) la pozitia finala
    near = min((math.hypot(xs[-1] - ox, ys[-1] - oy) - orr) for ox, oy, orr in obstacles)
    print(f"{i:>3} {reason:<10} {d_end:>10.2f} {d_min:>10.2f} {depl:>9.2f} {near:>9.2f} {len(rows):>5}")

# agregat pe toate episoadele stuck din sumar (fara traiectorie: doar steps)
from collections import Counter
c = Counter(r["done_reason"] for r in summary)
print("\nsumar:", dict(c))
stuck_steps = [int(r["steps"]) for r in summary if r["done_reason"] == "stuck"]
goal_steps = [int(r["steps"]) for r in summary if r["done_reason"] == "goal"]
if stuck_steps:
    print(f"pasi medii stuck: {sum(stuck_steps)/len(stuck_steps):.0f} (min {min(stuck_steps)}, max {max(stuck_steps)})")
if goal_steps:
    print(f"pasi medii goal:  {sum(goal_steps)/len(goal_steps):.0f}")
