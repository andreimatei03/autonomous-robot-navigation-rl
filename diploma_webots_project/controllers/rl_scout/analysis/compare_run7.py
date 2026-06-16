"""Analiza run 7: eval distance bins (run5/6/7), colapsul de antrenare,
episodul best-avg50, peak-vs-final avg50."""

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
    return out


def reasons(rows):
    return "  ".join(f"{k}={v}" for k, v in Counter(r["done_reason"] for r in rows).most_common())


print("="*60)
print("EVAL random/centru/unseen:  run5 -> run6 -> run7")
print("="*60)
for label, path in [
    ("R5 random", r"runs\run5_20260612\logs\eval_summary.csv"),
    ("R6 random", r"runs\run6_20260613\logs\eval_summary.csv"),
    ("R7 random", r"logs\eval_summary.csv"),
    ("R5 centru", r"runs\run5_20260612\logs\eval_summary_center.csv"),
    ("R7 centru", r"logs\eval_summary_center.csv"),
    ("R5 unseen", r"runs\run5_20260612\logs\eval_summary_unseen.csv"),
    ("R7 unseen", r"logs\eval_summary_unseen.csv"),
]:
    rows = load(path)
    if rows is None:
        print(f"{label}: LIPSA")
        continue
    print(f"\n{label} ({len(rows)})  {reasons(rows)}")
    for ln in by_distance(rows):
        print("   " + ln)

rows = list(csv.DictReader(open(r"logs\training_metrics.csv")))
print("\n" + "="*60)
print("RUN 7 ANTRENARE: bloc de 100 ep")
print("="*60)
best_avg, best_ep = -1e9, 0
peak_block = -1e9
for r in rows:
    a = float(r["avg50"])
    if a > best_avg:
        best_avg, best_ep = a, int(r["episode"])
for b in range(0, len(rows), 100):
    blk = rows[b:b+100]
    c = Counter(r["done_reason"] for r in blk)
    rew = sum(float(r["reward"]) for r in blk) / len(blk)
    losses = [float(r["loss"]) for r in blk if r.get("loss") not in (None, "", "nan")]
    print(f"ep {b+1:>4}-{b+len(blk):<4} goal={c.get('goal',0):>2} coll={c.get('collision',0):>2} "
          f"stuck={c.get('stuck',0):>2} | rew_med={rew:8.1f} loss={sum(losses)/max(1,len(losses)):7.2f}")
print(f"\nBest Avg50 = {best_avg:.1f} @ ep {best_ep} (modelul evaluat)")
print(f"avg50 FINAL (ep1200) = {float(rows[-1]['avg50']):.1f}  <- colaps daca << peak")
print(f"Total: {dict(Counter(r['done_reason'] for r in rows))}")
