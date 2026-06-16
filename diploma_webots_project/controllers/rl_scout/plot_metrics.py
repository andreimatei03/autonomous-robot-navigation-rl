"""Generează graficele pentru lucrare din CSV-urile produse de rl_scout.py.

Rulare (din folderul controllerului, cu rl_env activat):
    python plot_metrics.py

Citește din logs/ și salvează PNG-uri în figures/.
"""

import os
import csv
import glob
import math
import re
import sys

# Consola Windows e implicit cp1252 (fără „ă/ș/ț") → print-urile cu
# diacritice ar crăpa scriptul. Forțăm UTF-8 unde se poate.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "logs"
FIG_DIR = "figures"


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _moving_average(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_training():
    path = os.path.join(LOG_DIR, "training_metrics.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Lipsește {path} (rulează training întâi).")
        return

    rows = _read_csv(path)
    if not rows:
        print("[SKIP] training_metrics.csv este gol.")
        return

    episodes = [int(r["episode"]) for r in rows]
    rewards = [float(r["reward"]) for r in rows]
    avg50 = [float(r["avg50"]) for r in rows]
    steps = [int(r["steps"]) for r in rows]
    epsilon = [float(r["epsilon"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    reasons = [r["done_reason"] for r in rows]

    # 1. Curba de învățare
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Reward/episod")
    plt.plot(episodes, avg50, color="C1", linewidth=2, label="Media mobilă (50)")
    plt.xlabel("Episod")
    plt.ylabel("Reward cumulat")
    plt.title("Curba de învățare DQN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "learning_curve.png"), dpi=150)
    plt.close()

    # 2. Rata de succes (fereastră glisantă)
    window = 50
    success_flags = [1.0 if r == "goal" else 0.0 for r in reasons]
    success_rate = [100.0 * v for v in _moving_average(success_flags, window)]
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, success_rate, color="C2", linewidth=2)
    plt.xlabel("Episod")
    plt.ylabel(f"Rată de succes (% pe {window} ep.)")
    plt.title("Rata de succes în timpul antrenării")
    plt.ylim(-2, 102)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "success_rate.png"), dpi=150)
    plt.close()

    # 3. Rata de coliziune / răsturnare
    collision_flags = [1.0 if r in ("collision", "rollover") else 0.0 for r in reasons]
    collision_rate = [100.0 * v for v in _moving_average(collision_flags, window)]
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, collision_rate, color="C3", linewidth=2)
    plt.xlabel("Episod")
    plt.ylabel(f"Rată coliziune/răsturnare (% pe {window} ep.)")
    plt.title("Rata de coliziune/răsturnare")
    plt.ylim(-2, 102)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "collision_rate.png"), dpi=150)
    plt.close()

    # 4. Lungimea episodului
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, steps, alpha=0.3, label="Pași/episod")
    plt.plot(episodes, _moving_average([float(s) for s in steps], window),
             color="C4", linewidth=2, label="Media mobilă (50)")
    plt.xlabel("Episod")
    plt.ylabel("Pași până la final")
    plt.title("Lungimea episodului")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "episode_length.png"), dpi=150)
    plt.close()

    # 5. Curba de pierdere DQN
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, loss, color="C5", linewidth=1)
    plt.xlabel("Episod")
    plt.ylabel("TD-loss mediu / episod")
    plt.title("Convergența rețelei (TD-loss)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dqn_loss.png"), dpi=150)
    plt.close()

    # 6. Decăderea epsilon
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, epsilon, color="C6", linewidth=2)
    plt.xlabel("Episod")
    plt.ylabel("ε (explorare)")
    plt.title("Decăderea epsilon (explorare vs exploatare)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "epsilon_decay.png"), dpi=150)
    plt.close()

    print("[OK] Grafice de antrenare salvate în figures/.")


def _load_obstacles(tag=""):
    """Harta obstacolelor evaluării `tag` (obstacles{tag}.csv), cu fallback pe
    obstacles.csv — evaluările pe hărți diferite au hărți de obstacole diferite."""
    path = os.path.join(LOG_DIR, f"obstacles{tag}.csv")
    if not os.path.exists(path):
        path = os.path.join(LOG_DIR, "obstacles.csv")
    if not os.path.exists(path):
        return []
    return [(float(r["x"]), float(r["y"]), float(r["r"])) for r in _read_csv(path)]


def plot_eval_trajectories(tag="", title_mode="start random"):
    # Doar fișierele EXACT eval_traj{tag}_<număr>.csv: glob-ul simplu pentru
    # tag="" prindea și eval_traj_center_*/eval_traj_baseline_* (amesteca
    # traiectoriile celor trei moduri în aceeași figură).
    rx = re.compile(rf"^eval_traj{re.escape(tag)}_(\d+)\.csv$")
    matched = []
    for p in glob.glob(os.path.join(LOG_DIR, f"eval_traj{tag}_*.csv")):
        m = rx.match(os.path.basename(p))
        if m:
            matched.append((int(m.group(1)), p))
    # Desenăm doar primele 20 de trasee (lizibilitate): de la save_traj=100,
    # toate episoadele au traiectorie pe disc (SPL le folosește pe toate), dar
    # 100 de linii ar îneca figura. SPL (compute_spl) NU e plafonat.
    traj_files = [p for _, p in sorted(matched)][:20]
    if not traj_files:
        if tag == "":
            print("[SKIP] Nu există traiectorii de evaluare (logs/eval_traj_*.csv).")
        return

    obstacles = _load_obstacles(tag)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for ox, oy, orr in obstacles:
        ax.add_patch(plt.Circle((ox, oy), orr, color="0.6", alpha=0.5))

    extent = 5.0  # limita axelor se auto-potrivește pe harta reală (±55 m)
    for k, tf in enumerate(traj_files):
        rows = _read_csv(tf)
        if not rows:
            continue
        xs = [float(r["x"]) for r in rows]
        ys = [float(r["y"]) for r in rows]
        ax.plot(xs, ys, linewidth=1.5, label=f"Traseu {k}")
        ax.plot(xs[0], ys[0], "o", color="green", markersize=7)
        gx = float(rows[-1]["goal_x"])
        gy = float(rows[-1]["goal_y"])
        ax.plot(gx, gy, "*", color="red", markersize=13)
        extent = max(extent, max(abs(v) for v in xs + ys + [gx, gy]))

    for ox, oy, orr in obstacles:
        extent = max(extent, abs(ox) + orr, abs(oy) + orr)

    lim = extent * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Traiectorii de evaluare — {title_mode} (verde=start, roșu=goal)")
    ax.grid(True, alpha=0.3)
    if len(traj_files) <= 12:
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"eval_trajectories{tag}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Grafic traiectorii salvat în {out}.")


def plot_eval_summary(tag="", title_mode="start random"):
    path = os.path.join(LOG_DIR, f"eval_summary{tag}.csv")
    if not os.path.exists(path):
        return
    rows = _read_csv(path)
    if not rows:
        return

    goal_steps = [int(r["steps"]) for r in rows if r["done_reason"] == "goal"]
    n_total = len(rows)
    n_goal = len(goal_steps)
    rate = 100.0 * n_goal / n_total if n_total else 0.0

    if goal_steps:
        plt.figure(figsize=(8, 5))
        plt.hist(goal_steps, bins=min(15, max(3, int(math.sqrt(len(goal_steps))))),
                 color="C0", edgecolor="black", alpha=0.8)
        plt.xlabel("Pași până la goal")
        plt.ylabel("Număr episoade")
        plt.title(f"Pași-până-la-goal — {title_mode} (succes {rate:.0f}% din {n_total})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"eval_steps_hist{tag}.png"), dpi=150)
        plt.close()
        print(f"[OK] Histogramă evaluare ({title_mode}). Rată succes: {rate:.0f}% ({n_goal}/{n_total}).")


def plot_comparison():
    """Bar chart comparativ: distribuția rezultatelor (succes/coliziune/...)
    pentru DQN cu start random, DQN cu start din centru și baseline-ul
    reactiv — figura de comparație centrală a lucrării. Folosește doar
    sumarele care există pe disc; cere minimum două serii."""
    series = [("", "DQN — start random"),
              ("_center", "DQN — start centru"),
              ("_unseen", "DQN — hartă nevăzută"),
              ("_baseline", "Baseline reactiv")]
    cats = ["goal", "collision", "stuck", "rollover", "out_of_bounds", "timeout"]
    labels = {"goal": "succes", "collision": "coliziune", "stuck": "blocat",
              "rollover": "răsturnare", "out_of_bounds": "ieșire hartă",
              "timeout": "timeout"}
    known = set(cats) - {"timeout"}

    data = []
    for tag, name in series:
        path = os.path.join(LOG_DIR, f"eval_summary{tag}.csv")
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        n = len(rows)
        rates = []
        for c in cats:
            if c == "timeout":
                k = sum(1 for r in rows if r["done_reason"] not in known)
            else:
                k = sum(1 for r in rows if r["done_reason"] == c)
            rates.append(100.0 * k / n)
        data.append((f"{name} (N={n})", rates, n))

    if len(data) < 2:
        return

    x = range(len(cats))
    w = 0.8 / len(data)
    plt.figure(figsize=(9.5, 5))
    for i, (name, rates, n) in enumerate(data):
        # Interval de încredere 95% pentru proporții (aproximare normală):
        # 1.96·sqrt(p(1−p)/n) — bara de eroare cuantifică incertitudinea
        # estimării pe N episoade.
        errs = [196.0 * math.sqrt(max(v, 1e-9) / 100.0 * (1.0 - v / 100.0) / n)
                for v in rates]
        plt.bar([xi + i * w for xi in x], rates, width=w, label=name,
                yerr=errs, capsize=2, error_kw={"alpha": 0.55, "lw": 1})
        for xi, v, e in zip(x, rates, errs):
            if v > 0.5:
                plt.text(xi + i * w, v + e + 1, f"{v:.0f}", ha="center", fontsize=8)
    plt.xticks([xi + w * (len(data) - 1) / 2 for xi in x],
               [labels[c] for c in cats])
    plt.ylabel("% din episoade")
    plt.title("Comparație: rezultatul episoadelor de evaluare")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eval_comparison.png"), dpi=150)
    plt.close()
    print("[OK] Grafic comparativ salvat (eval_comparison.png).")


def plot_success_by_distance():
    """Rata de succes pe intervale de distanță start→goal (din sumarele de
    evaluare): raza de acțiune efectivă a fiecărei politici — la ce distanțe
    începe să cedeze fiecare."""
    series = [("", "DQN — start random"),
              ("_center", "DQN — start centru"),
              ("_unseen", "DQN — hartă nevăzută"),
              ("_baseline", "Baseline reactiv")]
    bins = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 1e9)]
    bin_labels = ["0–15 m", "15–30 m", "30–45 m", "45+ m"]

    data = []
    for tag, name in series:
        path = os.path.join(LOG_DIR, f"eval_summary{tag}.csv")
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        rates, counts = [], []
        for lo, hi in bins:
            sel = [r for r in rows
                   if lo <= math.hypot(float(r["goal_x"]) - float(r["start_x"]),
                                       float(r["goal_y"]) - float(r["start_y"])) < hi]
            counts.append(len(sel))
            rates.append(100.0 * sum(1 for r in sel if r["done_reason"] == "goal")
                         / len(sel) if sel else 0.0)
        data.append((name, rates, counts))
    if not data:
        return

    x = range(len(bins))
    w = 0.8 / len(data)
    plt.figure(figsize=(9.5, 5))
    for i, (name, rates, counts) in enumerate(data):
        plt.bar([xi + i * w for xi in x], rates, width=w, label=name)
        for xi, v, c in zip(x, rates, counts):
            if c:
                plt.text(xi + i * w, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    plt.xticks([xi + w * (len(data) - 1) / 2 for xi in x], bin_labels)
    plt.xlabel("Distanța start → goal")
    plt.ylabel("Rată de succes (%)")
    plt.ylim(0, 105)
    plt.title("Succes în funcție de distanța până la țintă")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eval_success_by_distance.png"), dpi=150)
    plt.close()
    print("[OK] Grafic succes-pe-distanță salvat (eval_success_by_distance.png).")


def _path_length(rows):
    """Lungimea drumului parcurs (suma segmentelor euclidiene între pașii
    consecutivi ai traiectoriei)."""
    p = 0.0
    for a, b in zip(rows, rows[1:]):
        p += math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
    return p


def compute_spl(tag=""):
    """SPL — Success weighted by Path Length (Anderson et al., 2018), metrica
    standard de eficiență a navigației:

        SPL = (1/N) · Σ Sᵢ · lᵢ / max(pᵢ, lᵢ)

    Sᵢ = 1 dacă episodul i a atins ținta (0 altfel), lᵢ = distanța optimă
    start→goal, pᵢ = lungimea drumului efectiv parcurs. Spre deosebire de rata
    de succes (binară), SPL penalizează ocolurile inutile → 1.0 = drum perfect
    direct, valori mici = ajunge dar pe căi lungi/sinuoase.

    lᵢ = distanța EUCLIDIANĂ start→goal, o limită INFERIOARĂ a drumului optim
    care ocolește obstacolele; prin urmare SPL real ≥ valoarea raportată
    (estimare conservatoare). pᵢ se calculează din traiectoria logată
    (logs/eval_traj{tag}_*.csv), așa că metrica acoperă doar episoadele cu
    traiectorie salvată (vezi `save_traj` în rl_scout.py). Întoarce un dict de
    statistici sau None dacă lipsesc datele."""
    summary_path = os.path.join(LOG_DIR, f"eval_summary{tag}.csv")
    if not os.path.exists(summary_path):
        return None
    summary = _read_csv(summary_path)
    if not summary:
        return None
    reason_by_run = {int(r["run"]): r["done_reason"] for r in summary}

    rx = re.compile(rf"^eval_traj{re.escape(tag)}_(\d+)\.csv$")
    traj = {}
    for p in glob.glob(os.path.join(LOG_DIR, f"eval_traj{tag}_*.csv")):
        m = rx.match(os.path.basename(p))
        if m:
            traj[int(m.group(1))] = p
    if not traj:
        return None

    spls, effs = [], []
    n_succ_subset = 0
    for idx, path in sorted(traj.items()):
        rows = _read_csv(path)
        if len(rows) < 2:
            continue
        sx, sy = float(rows[0]["x"]), float(rows[0]["y"])
        gx, gy = float(rows[-1]["goal_x"]), float(rows[-1]["goal_y"])
        l = math.hypot(gx - sx, gy - sy)
        p_len = _path_length(rows)
        success = reason_by_run.get(idx) == "goal"
        if success and l > 1e-6:
            n_succ_subset += 1
            effs.append(min(1.0, l / p_len) if p_len > 1e-6 else 0.0)
            spls.append(l / max(p_len, l))
        else:
            spls.append(0.0)

    if not spls:
        return None
    n_sub = len(spls)
    full_succ = 100.0 * sum(1 for r in summary if r["done_reason"] == "goal") / len(summary)
    return {
        "tag": tag,
        "n_traj": n_sub,
        "n_full": len(summary),
        "success_subset": 100.0 * n_succ_subset / n_sub,
        "success_full": full_succ,
        "spl": sum(spls) / n_sub,
        "path_eff": (sum(effs) / len(effs)) if effs else 0.0,
    }


def plot_spl():
    """Figură + tabel SPL pentru toate evaluările existente pe disc. Scrie
    figures/eval_spl.png și logs/spl_summary.csv (tabelul din lucrare)."""
    series = [("", "DQN — start random"),
              ("_center", "DQN — start centru"),
              ("_unseen", "DQN — hartă nevăzută"),
              ("_baseline", "Baseline reactiv")]
    rows = []
    for tag, name in series:
        d = compute_spl(tag)
        if d:
            rows.append((name, d))
    if not rows:
        print("[SKIP] SPL: lipsesc traiectoriile de evaluare (logs/eval_traj*_*.csv).")
        return

    # CSV pentru tabelul din lucrare (UTF-8: numele seriilor au diacritice/„—")
    with open(os.path.join(LOG_DIR, "spl_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "n_traiectorii", "succes_subset_%", "succes_total_%",
                    "SPL", "eficienta_drum_succes_%"])
        for name, d in rows:
            w.writerow([name, d["n_traj"], f"{d['success_subset']:.0f}",
                        f"{d['success_full']:.0f}", f"{d['spl']:.3f}",
                        f"{100 * d['path_eff']:.0f}"])

    labels = [name for name, _ in rows]
    spls = [d["spl"] for _, d in rows]
    plt.figure(figsize=(8.5, 5))
    plt.bar(range(len(rows)), spls,
            color=["C0", "C1", "C2", "C3"][:len(rows)], alpha=0.85)
    for i, (_, d) in enumerate(rows):
        plt.text(i, d["spl"] + 0.012, f"{d['spl']:.3f}", ha="center", fontsize=10)
    plt.xticks(range(len(rows)), labels, rotation=12)
    plt.ylabel("SPL (Success weighted by Path Length)")
    plt.ylim(0, max(0.1, max(spls)) * 1.20)
    plt.title(f"Eficiența drumului (SPL)  ·  N={rows[0][1]['n_traj']} traiectorii/set")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eval_spl.png"), dpi=150)
    plt.close()

    print("[OK] SPL salvat (eval_spl.png, logs/spl_summary.csv):")
    print(f"  {'set':22s} {'N':>3} {'succ_sub':>8} {'succ_tot':>8} {'SPL':>6} {'ef.drum':>8}")
    for name, d in rows:
        print(f"  {name:22s} {d['n_traj']:>3} {d['success_subset']:>7.0f}% "
              f"{d['success_full']:>7.0f}% {d['spl']:>6.3f} {100 * d['path_eff']:>7.0f}%")


def write_results_summary():
    """Tabel consolidat (un singur CSV) cu toate metricile per set de evaluare:
    ratele de done_reason, pașii medii la succes, SPL și eficiența drumului.
    Sursă unică pentru tabelul de rezultate din lucrare → logs/results_summary.csv."""
    series = [("", "DQN — start random"),
              ("_center", "DQN — start centru"),
              ("_unseen", "DQN — hartă nevăzută"),
              ("_baseline", "Baseline reactiv")]
    known = {"goal", "collision", "rollover", "out_of_bounds", "stuck"}

    out_rows = []
    for tag, name in series:
        path = os.path.join(LOG_DIR, f"eval_summary{tag}.csv")
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        n = len(rows)

        def rate(reason):
            return 100.0 * sum(1 for r in rows if r["done_reason"] == reason) / n

        timeout = 100.0 * sum(1 for r in rows if r["done_reason"] not in known) / n
        goal_steps = [int(r["steps"]) for r in rows if r["done_reason"] == "goal"]
        avg_steps = (sum(goal_steps) / len(goal_steps)) if goal_steps else 0.0
        spl = compute_spl(tag)
        out_rows.append({
            "set": name, "N": n,
            "succ": rate("goal"), "coll": rate("collision"), "stuck": rate("stuck"),
            "roll": rate("rollover"), "oob": rate("out_of_bounds"), "to": timeout,
            "steps": avg_steps,
            "spl": spl["spl"] if spl else float("nan"),
            "spl_n": spl["n_traj"] if spl else 0,
            "eff": 100.0 * spl["path_eff"] if spl else float("nan"),
        })

    if not out_rows:
        print("[SKIP] results_summary: lipsesc sumarele de evaluare.")
        return

    out = os.path.join(LOG_DIR, "results_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "N", "succes_%", "coliziune_%", "stuck_%", "rollover_%",
                    "out_of_bounds_%", "timeout_%", "pasi_medii_goal", "SPL",
                    "SPL_n_traiectorii", "eficienta_drum_%"])
        for r in out_rows:
            w.writerow([r["set"], r["N"], f"{r['succ']:.0f}", f"{r['coll']:.0f}",
                        f"{r['stuck']:.0f}", f"{r['roll']:.0f}", f"{r['oob']:.0f}",
                        f"{r['to']:.0f}", f"{r['steps']:.0f}", f"{r['spl']:.3f}",
                        r["spl_n"], f"{r['eff']:.0f}"])

    print(f"[OK] Tabel consolidat salvat ({out}):")
    print(f"  {'set':22s} {'N':>3} {'succ':>5} {'coll':>5} {'stuck':>6} "
          f"{'roll':>5} {'oob':>4} {'t/o':>4} {'pași':>6} {'SPL':>6} {'ef%':>4}")
    for r in out_rows:
        print(f"  {r['set']:22s} {r['N']:>3} {r['succ']:>4.0f}% {r['coll']:>4.0f}% "
              f"{r['stuck']:>5.0f}% {r['roll']:>4.0f}% {r['oob']:>3.0f}% {r['to']:>3.0f}% "
              f"{r['steps']:>6.0f} {r['spl']:>6.3f} {r['eff']:>3.0f}%")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_training()
    # Evaluare principală (start random) + comparații (centru / baseline, dacă există).
    plot_eval_trajectories(tag="", title_mode="start random")
    plot_eval_summary(tag="", title_mode="start random")
    plot_eval_trajectories(tag="_center", title_mode="start centru")
    plot_eval_summary(tag="_center", title_mode="start centru")
    plot_eval_trajectories(tag="_baseline", title_mode="baseline reactiv")
    plot_eval_summary(tag="_baseline", title_mode="baseline reactiv")
    plot_eval_trajectories(tag="_unseen", title_mode="hartă nevăzută")
    plot_eval_summary(tag="_unseen", title_mode="hartă nevăzută")
    plot_comparison()
    plot_success_by_distance()
    plot_spl()
    write_results_summary()
    print("Gata. Verifică folderul figures/.")


if __name__ == "__main__":
    main()
