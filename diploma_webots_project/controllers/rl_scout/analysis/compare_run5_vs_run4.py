"""Comparatie run 5 vs run 4 pe aceleasi bin-uri + trend loss run 5."""

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
        sel = []
        for r in rows:
            d = math.hypot(float(r["goal_x"]) - float(r["start_x"]),
                           float(r["goal_y"]) - float(r["start_y"]))
            if lo <= d < hi:
                sel.append(r["done_reason"])
        if sel:
            s = sum(1 for x in sel if x == "goal")
            out.append(f"{lo:>2.0f}-{'45+' if hi > 100 else f'{hi:.0f}'} m: {s}/{len(sel)} = {100*s/len(sel):.0f}%")
        else:
            out.append(f"{lo:>2.0f}-{hi:.0f} m: (0 ep)")
    return out


def reasons(rows):
    from collections import Counter
    c = Counter(r["done_reason"] for r in rows)
    return "  ".join(f"{k}={v}" for k, v in c.most_common())


for label, path in [
    ("RUN 4 random", r"runs\run4_20260610\logs\eval_summary.csv"),
    ("RUN 5 random", r"logs\eval_summary.csv"),
    ("RUN 4 centru", r"runs\run4_20260610\logs\eval_summary_center.csv"),
    ("RUN 5 centru", r"logs\eval_summary_center.csv"),
    ("RUN 5 unseen", r"logs\eval_summary_unseen.csv"),
    ("RUN 5 baseline", r"logs\eval_summary_baseline.csv"),
]:
    rows = load(path)
    if rows is None:
        print(f"{label}: LIPSESTE {path}")
        continue
    print(f"{label}  ({len(rows)} ep)   {reasons(rows)}")
    for line in by_distance(rows):
        print("   " + line)
    print()

# Trend loss + steps pe blocuri de 100 (run 5)
rows = list(csv.DictReader(open(r"logs\training_metrics.csv")))
print("=== RUN 5: loss mediu / blocuri de 100 episoade ===")
for b in range(0, len(rows), 100):
    blk = rows[b:b + 100]
    losses = [float(r["loss"]) for r in blk if r.get("loss") not in (None, "", "nan")]
    rew = [float(r["reward"]) for r in blk]
    print(f"ep {b+1:>4}-{b+len(blk):<4} loss={sum(losses)/max(1,len(losses)):8.4f}  rew_med={sum(rew)/len(rew):7.1f}")
