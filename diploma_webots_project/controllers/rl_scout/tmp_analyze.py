"""Analiza rapida a logurilor de antrenare + evaluare (temporar, se poate sterge)."""
import csv
from collections import Counter

LOGS = r"logs"

# --- training ---
rows = list(csv.DictReader(open(LOGS + r"\training_metrics.csv")))
print("=== TRAINING:", len(rows), "episoade ===")
print("coloane:", list(rows[0].keys()))
B = 100
for s in range(0, len(rows), B):
    blk = rows[s:s + B]
    rew = [float(r["reward"]) for r in blk]
    steps = [float(r["steps"]) for r in blk]
    eps = [float(r["epsilon"]) for r in blk]
    reasons = Counter(r["done_reason"] for r in blk)
    succ = reasons.get("goal", 0)
    rest = "  ".join(f"{k}={v}" for k, v in reasons.most_common() if k != "goal")
    print(f"ep {s+1:4d}-{s+len(blk):4d}  rew={sum(rew)/len(rew):8.1f}  steps={sum(steps)/len(steps):6.0f}  eps {eps[0]:.3f}->{eps[-1]:.3f}  goal={succ:3d}  {rest}")

# best avg50
avg50 = []
rws = [float(r["reward"]) for r in rows]
for i in range(len(rws)):
    lo = max(0, i - 49)
    avg50.append(sum(rws[lo:i+1]) / (i - lo + 1))
mx = max(range(len(avg50)), key=lambda i: avg50[i])
print(f"best avg50 = {avg50[mx]:.1f} la episodul {mx+1}; avg50 final = {avg50[-1]:.1f}")
print(f"best reward single = {max(rws):.1f} la ep {rws.index(max(rws))+1}")

# --- evals ---
for name, path in [("EVAL RANDOM-START", LOGS + r"\eval_summary.csv"),
                   ("EVAL CENTER-START", LOGS + r"\eval_summary_center.csv")]:
    rws_ = list(csv.DictReader(open(path)))
    n = len(rws_)
    reasons = Counter(r["done_reason"] for r in rws_)
    goal_steps = [int(r["steps"]) for r in rws_ if r["done_reason"] == "goal"]
    print(f"\n=== {name}: {n} episoade ===")
    for k, v in reasons.most_common():
        print(f"  {k:12s} {v:3d}  ({100*v/n:.0f}%)")
    if goal_steps:
        print(f"  pasi medii pana la goal: {sum(goal_steps)/len(goal_steps):.0f}  (min {min(goal_steps)}, max {max(goal_steps)})")
    # distanta start-goal vs succes
    import math
    far_goal = far_all = near_goal = near_all = 0
    for r in rws_:
        d = math.hypot(float(r["goal_x"]) - float(r["start_x"]), float(r["goal_y"]) - float(r["start_y"]))
        if d >= 30:
            far_all += 1
            far_goal += r["done_reason"] == "goal"
        else:
            near_all += 1
            near_goal += r["done_reason"] == "goal"
    if near_all:
        print(f"  succes goal apropiat (<30m): {near_goal}/{near_all} ({100*near_goal/near_all:.0f}%)")
    if far_all:
        print(f"  succes goal departat (>=30m): {far_goal}/{far_all} ({100*far_goal/far_all:.0f}%)")
