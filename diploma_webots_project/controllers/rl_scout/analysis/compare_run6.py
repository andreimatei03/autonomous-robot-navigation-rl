"""Analiza run 6 vs run 5 (arhivat) vs run 4 (arhivat): defalcare pe distante,
blocul de antrenare, episodul best-avg50, si unde se termina episoadele stuck."""

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
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

BINS = [(0, 15), (15, 30), (30, 45), (45, 1e9)]


def load(path):
    try:
        return list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return None


def by_distance(rows):
    out = []
    for lo, hi in BINS:
        sel = [r["done_reason"] for r in rows
               if lo <= math.hypot(float(r["goal_x"]) - float(r["start_x"]),
                                   float(r["goal_y"]) - float(r["start_y"])) < hi]
        if sel:
            s = sum(1 for x in sel if x == "goal")
            lbl = "45+" if hi > 100 else f"{hi:.0f}"
            out.append(f"{lo:>2.0f}-{lbl} m: {s:>2}/{len(sel):<2} = {100*s/len(sel):3.0f}%")
        else:
            out.append(f"{lo:>2.0f}-{hi:.0f} m: (0 ep)")
    return out


def reasons(rows):
    c = Counter(r["done_reason"] for r in rows)
    return "  ".join(f"{k}={v}" for k, v in c.most_common())


print("="*64)
print("COMPARATIE EVAL (random-start):  run4 -> run5 -> run6")
print("="*64)
for label, path in [
    ("RUN 4 random", r"runs\run4_20260610\logs\eval_summary.csv"),
    ("RUN 5 random", r"runs\run5_20260612\logs\eval_summary.csv"),
    ("RUN 6 random", r"logs\eval_summary.csv"),
    ("RUN 5 centru ", r"runs\run5_20260612\logs\eval_summary_center.csv"),
    ("RUN 6 centru ", r"logs\eval_summary_center.csv"),
    ("RUN 5 unseen ", r"runs\run5_20260612\logs\eval_summary_unseen.csv"),
    ("RUN 6 unseen ", r"logs\eval_summary_unseen.csv"),
]:
    rows = load(path)
    if rows is None:
        print(f"{label}: LIPSESTE {path}")
        continue
    print(f"\n{label}  ({len(rows)} ep)   {reasons(rows)}")
    for line in by_distance(rows):
        print("   " + line)

# --- Antrenare run 6: bloc de 100 + episodul best-avg50 ---
rows = list(csv.DictReader(open(r"logs\training_metrics.csv")))
print("\n" + "="*64)
print("RUN 6 ANTRENARE: done_reason / bloc de 100 ep + loss + avg50")
print("="*64)
best_avg, best_ep = -1e9, 0
for r in rows:
    a = float(r["avg50"])
    if a > best_avg:
        best_avg, best_ep = a, int(r["episode"])
for b in range(0, len(rows), 100):
    blk = rows[b:b+100]
    c = Counter(r["done_reason"] for r in blk)
    losses = [float(r["loss"]) for r in blk if r.get("loss") not in (None, "", "nan")]
    print(f"ep {b+1:>4}-{b+len(blk):<4} "
          f"goal={c.get('goal',0):>2} coll={c.get('collision',0):>2} "
          f"stuck={c.get('stuck',0):>2} to={c.get('timeout',0):>2} "
          f"roll={c.get('rollover',0):>2} | loss={sum(losses)/max(1,len(losses)):7.3f}")
print(f"\nBest Avg50 = {best_avg:.1f} la episodul {best_ep}  (modelul evaluat)")
total = Counter(r["done_reason"] for r in rows)
print(f"Total 800 ep: {dict(total)}")
