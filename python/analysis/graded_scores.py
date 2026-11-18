# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/graded_scores.py --session python/main/real_time_data/sub01/ses1 --out python/results/graded_scores
"""
Graded per-trial scores for the A/B switching test (known issue H1), compared with right/wrong scoring.

Per trial and model, from run{N}/predictions.csv and the trial's vote time (AT in the log):
  margin: Unity-weighted share of the last 100 predictions that are the correct class (weights 1 -> 5).
          The vote is correct when margin >= 0.5, so it is the vote itself, before rounding to right/wrong.
  conf:   the same weighted mean of the predicted probability of the correct class (needs prob1).
Paired difference per trial: D = score(B) - score(A).

Outputs: trials.csv, summary.txt and figures g1-g4 in --out.
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wald_sprt_explainer import (SURFACE, INK, INK2, MUTED, GRID, AXIS, MODEL_COLOR, RULE_COLOR, NEUTRAL_DARK,
                                 style, header, SPRT, PY_DIR)

HISTORY = 100
LINE_RE = re.compile(r"Trial (\d+) END: [\d.]+ LABEL: (\d) MODEL_PREDICTION: A=(\w+) B=(\w+) LIVE=(\w) AT: ([\d.]+)")


# ---------------- data ----------------
def unity_weights(n):
    return 1.0 + 4.0 * np.arange(n) / (n - 1) if n > 1 else np.ones(n)


def unity_vote(preds):
    """Exact replica of trial_scoring.unity_vote (rounded weights for class-1 count)."""
    n = len(preds)
    if n < 2:
        return None
    w = unity_weights(n)
    count1 = sum(round(x) for x, p in zip(w, preds) if p == 1)
    return 1 if count1 >= w.sum() / 2.0 else 0


def load_predictions(path):
    out = {"A": [], "B": []}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["model"]].append((float(r["time"]), int(r["pred"]), float(r["prob1"])))
    res = {}
    for m, rows in out.items():
        a = np.array(rows) if rows else np.zeros((0, 3))
        a = a[np.argsort(a[:, 0], kind="stable")]
        res[m] = a
    return res


def load_trials(session_dir):
    """-> list of dicts, one per scored trial, in recorded order."""
    trials = []
    run_dirs = sorted(glob.glob(os.path.join(session_dir, "run*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
    for d in run_dirs:
        logs = sorted(glob.glob(os.path.join(d, "log_*.txt")))
        pred_path = os.path.join(d, "predictions.csv")
        blocks_path = os.path.join(d, "ab_blocks.csv")
        if not logs or not os.path.exists(pred_path) or not os.path.exists(blocks_path):
            continue
        preds = load_predictions(pred_path)
        with open(blocks_path, encoding="utf-8") as f:
            blocks = [int(r["n_trials"]) for r in csv.DictReader(f)]
        run_trials = []
        for log in logs:
            with open(log, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = LINE_RE.search(line)
                    if not m or m.group(3) == "NA" or m.group(4) == "NA":
                        continue
                    y, at = int(m.group(2)), float(m.group(6))
                    t = {"run": os.path.basename(d), "label": y, "at": at,
                         "logged": {"A": int(m.group(3)), "B": int(m.group(4))}}
                    for name in ("A", "B"):
                        p = preds[name]
                        k = np.searchsorted(p[:, 0], at, side="right")
                        last = p[max(0, k - HISTORY):k]
                        w = unity_weights(len(last))
                        correct = (last[:, 1] == y).astype(float)
                        t[f"vote_{name}"] = unity_vote(list(last[:, 1].astype(int)))
                        t[f"ok_{name}"] = t["logged"][name] == y
                        t[f"margin_{name}"] = float((w * correct).sum() / w.sum()) if len(last) else math.nan
                        pc = last[:, 2] if y == 1 else 1 - last[:, 2]
                        good = ~np.isnan(pc)
                        t[f"conf_{name}"] = float((w[good] * pc[good]).sum() / w[good].sum()) if good.sum() >= 2 else math.nan
                        t[f"n_{name}"] = len(last)
                    run_trials.append(t)
        if sum(blocks) != len(run_trials):
            raise ValueError(f"{d}: blocks sum to {sum(blocks)} but the log has {len(run_trials)} trials")
        i = 0
        for b, size in enumerate(blocks):
            for t in run_trials[i:i + size]:
                t["block_end"] = False
            run_trials[i + size - 1]["block_end"] = True
            i += size
        trials.extend(run_trials)
    return trials


# ---------------- sequential rules ----------------
class GaussSPRT:
    """SPRT on the paired difference D = challenger - live, H0 mean 0 vs H1 mean delta, known sd sigma."""

    def __init__(self, delta, sigma, alpha=0.05, beta=0.2, name=""):
        self.delta, self.sigma, self.alpha, self.beta, self.name = delta, sigma, alpha, beta, name
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))

    def step(self, d):
        return self.delta / self.sigma ** 2 * (d - self.delta / 2)


class SignSPRT:
    """The right/wrong SPRT from wald_sprt_explainer, written as a step on D = ok(challenger) - ok(live)."""

    def __init__(self, p1=0.7, alpha=0.05, beta=0.2, name=""):
        s = SPRT(p1, alpha, beta)
        self.win, self.loss, self.upper, self.lower, self.alpha, self.name = s.win, s.loss, s.upper, s.lower, alpha, name

    def step(self, d):
        return self.win if d > 0 else self.loss if d < 0 else 0.0


def run_rule(rule, dA, dB, block_end, start_live="A"):
    """dA, dB: per-trial scores (binary or graded) of A and B. Decisions at block ends, reset at the lower line.
    Returns (trace of score, live per trial, events)."""
    live, llr = start_live, 0.0
    trace, lives, events = [], [], []
    for i in range(len(dA)):
        d = (dB[i] - dA[i]) if live == "A" else (dA[i] - dB[i])
        llr += rule.step(d)
        trace.append(llr)
        lives.append(live)
        if block_end[i]:
            if llr >= rule.upper:
                events.append((i + 1, "switch", live, llr))
                live, llr = ("B" if live == "A" else "A"), 0.0
            elif llr <= rule.lower:
                events.append((i + 1, "reset", live, llr))
                llr = 0.0
    return np.array(trace), lives, events


def first_switch_batch(rule, DA, DB, block=4):
    """Vectorised over sessions: DA, DB shape (n_sim, n_trials), live starts as A. Returns first-switch trial or -1."""
    n_sim, n = DA.shape
    llr = np.zeros(n_sim)
    first = np.full(n_sim, -1)
    for b in range(n // block):
        sl = slice(b * block, (b + 1) * block)
        d = DB[:, sl] - DA[:, sl]
        if isinstance(rule, SignSPRT):
            llr += np.where(d > 0, rule.win, np.where(d < 0, rule.loss, 0.0)).sum(1)
        else:
            llr += (rule.delta / rule.sigma ** 2 * (d - rule.delta / 2)).sum(1)
        hit = (llr >= rule.upper) & (first < 0)
        first[hit] = (b + 1) * block
        llr[llr <= rule.lower] = 0.0
        llr[first >= 0] = -1e9
    return first


# ---------------- figures ----------------
def g1_validity(T, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    for name in ("A", "B"):
        m = np.array([t[f"margin_{name}"] for t in T])
        ok = np.array([t[f"ok_{name}"] for t in T])
        ax.hist(m[ok], bins=bins, color=MODEL_COLOR[name], alpha=0.55, label=f"{name}: vote right")
        ax.hist(m[~ok], bins=bins, color=MODEL_COLOR[name], alpha=0.55, hatch="///", edgecolor=SURFACE,
                label=f"{name}: vote wrong")
    ax.axvline(0.5, color=INK, lw=1, ls="--")
    ax.text(0.51, ax.get_ylim()[1] * 0.92, "vote flips at 0.5", fontsize=8.5)
    ax.set_xlabel("Margin = weighted share of the last 100 predictions that are the correct class")
    ax.set_ylabel("Trials")
    ax.set_title("g1a. The margin is the vote before rounding")
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[1]
    runs = sorted({t["run"] for t in T}, key=lambda r: int(r[3:]))
    for name in ("A", "B"):
        acc = [np.mean([t[f"ok_{name}"] for t in T if t["run"] == r]) for r in runs]
        mar = [np.mean([t[f"margin_{name}"] for t in T if t["run"] == r]) for r in runs]
        ax.plot(acc, mar, "o", color=MODEL_COLOR[name], ms=8, mec=SURFACE, mew=1.5, label=name)
        for r, x, y in zip(runs, acc, mar):
            ax.text(x + 0.004, y, r.replace("run", ""), fontsize=8, color=MUTED, va="center")
    ax.set_xlabel("Accuracy of the vote in the run")
    ax.set_ylabel("Mean margin in the run")
    ax.set_title("g1b. Per run, mean margin follows accuracy (labels = run)")
    ax.legend(fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.13, wspace=0.22)
    fig.savefig(os.path.join(out, "g1_margin_validity.png"), dpi=150)
    plt.close(fig)


def g2_information(T, out, stats):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    rest = [t for t in T if t["run"] != "run1"]
    defs = [("Right/wrong", "ok"), ("Margin", "margin"), ("Confidence", "conf")]
    for ax, (lab, key) in zip(axes[:2], defs[:2]):
        D = np.array([float(t[f"{key}_B"]) - float(t[f"{key}_A"]) for t in rest])
        if key == "ok":
            vals, cnt = np.unique(D, return_counts=True)
            ax.bar(vals, cnt, width=0.5, color=[MODEL_COLOR["A"], NEUTRAL_DARK, MODEL_COLOR["B"]])
            ax.set_xticks([-1, 0, 1], ["A right,\nB wrong", "same\n(no evidence)", "B right,\nA wrong"])
        else:
            ax.hist(D, bins=np.linspace(-1, 1, 41), color=NEUTRAL_DARK)
            ax.hist(D[D > 0], bins=np.linspace(-1, 1, 41), color=MODEL_COLOR["B"])
            ax.hist(D[D < 0], bins=np.linspace(-1, 1, 41), color=MODEL_COLOR["A"])
            ax.set_xlabel("D = margin(B) − margin(A)")
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_ylabel("Trials (runs 2–7)")
        nz = np.mean(np.abs(D) > 1e-9)
        ax.set_title(f"g2{'ab'[defs.index((lab, key))]}. {lab}: {nz:.0%} of trials carry evidence")
    ax = axes[2]
    labels = [lab for lab, _ in defs]
    eff = [stats[k]["d"] for _, k in defs]
    ax.bar(range(3), eff, width=0.55, color=[NEUTRAL_DARK, RULE_COLOR[2], RULE_COLOR[3]])
    for i, (v, (_, k)) in enumerate(zip(eff, defs)):
        ax.text(i, v + 0.005, f"d = {v:.2f}\n≈{stats[k]['n80']:.0f} trials", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, max(eff) * 1.45)
    ax.set_ylabel("Effect size per trial: mean(D) / sd(D)")
    ax.set_title("g2c. Evidence per trial, runs 2–7")
    ax.grid(axis="x", visible=False)
    fig.text(0.02, 0.01, "Trials ≈ trials a fixed-size one-sided test needs for 80% power at α = 0.05, "
             "(1.645 + 0.842)² / d². Needed trials scale with 1/d², so 2× the effect size = 4× fewer trials.",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.2, wspace=0.3)
    fig.savefig(os.path.join(out, "g2_information.png"), dpi=150)
    plt.close(fig)


def g3_replay(T, rules, out):
    blk = [t["block_end"] for t in T]
    fig, axes = plt.subplots(len(rules), 1, figsize=(12, 2.6 * len(rules) + 0.8), sharex=True)
    starts, prev = [], None
    for i, t in enumerate(T):
        if t["run"] != prev:
            starts.append((i, t["run"]))
            prev = t["run"]
    for ax, (rule, key) in zip(axes, rules):
        dA = np.array([float(t[f"{key}_A"]) for t in T])
        dB = np.array([float(t[f"{key}_B"]) for t in T])
        tr, lives, ev = run_rule(rule, dA, dB, blk)
        x = np.arange(1, len(tr) + 1)
        seg = 0
        for j in range(1, len(tr) + 1):
            if j == len(tr) or lives[j] != lives[seg]:
                chal = "B" if lives[seg] == "A" else "A"
                ax.plot(x[seg:j], np.clip(tr[seg:j], -4.2, 4), color=MODEL_COLOR[chal], lw=1.3)
                seg = j
        ax.axhline(rule.upper, color=INK, ls="--", lw=1)
        ax.axhline(rule.lower, color=INK2, ls=":", lw=1)
        ax.axhline(0, color=AXIS, lw=0.8)
        for s, r in starts:
            ax.axvline(s, color=AXIS, lw=0.8)
        for (tt, kind, lv, v) in ev:
            if kind == "switch":
                to = "B" if lv == "A" else "A"
                ax.plot(tt, min(v, 4), "o", ms=8, color=MODEL_COLOR[to], mec=SURFACE, mew=2, zorder=5)
                ax.text(tt, 4.3, f"{lv}→{to} @{tt}", fontsize=8.5, ha="center")
            else:
                ax.plot(tt, max(v, -4.2), "x", ms=5, color=INK2, zorder=5)
        ax.set_ylim(-4.6, 5.2)
        ax.set_ylabel("Score")
        ax.set_title(f"{rule.name}", fontsize=10.5, loc="left")
    for s, r in starts:
        axes[0].text(s + 2, 4.9, r.replace("run", "Run "), fontsize=8, color=MUTED, va="top")
    axes[-1].set_xlabel("Trial (2026-10-15 recording, all runs in order)")
    header(fig, "g3. Replay: right/wrong vs graded scores",
           "Orange = B challenging A, blue = A challenging B. Scores clipped to ±4 for display. ● switch, × reset.")
    fig.subplots_adjust(left=0.06, right=0.98, top=1 - 1.2 / fig.get_figheight(), bottom=0.07, hspace=0.45)
    fig.savefig(os.path.join(out, "g3_replay.png"), dpi=150)
    plt.close(fig)


def g4_resampling(results, out):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    worlds = [("null", "g4a. Equal models: any switch", "Sessions with ≥1 switch"),
              ("worse", "g4b. B worse: switch to the worse model", "Sessions with ≥1 switch"),
              ("better", "g4c. B better (as in runs 2–7): switch to B", "Sessions switched to B")]
    y = np.arange(len(names))[::-1]
    for ax, (w, title, xl) in zip(axes, worlds):
        p48 = [results[n][w][0] for n in names]
        p480 = [results[n][w][1] for n in names]
        ax.barh(y + 0.18, p48, height=0.34, color=RULE_COLOR[0], label="within 48 trials (1 run)")
        ax.barh(y - 0.18, p480, height=0.34, color=RULE_COLOR[1], label="within 480 trials (10 runs)")
        for yi, a, b in zip(y, p48, p480):
            ax.text(a + 0.01, yi + 0.18, f"{a:.0%}", va="center", fontsize=8.5)
            ax.text(b + 0.01, yi - 0.18, f"{b:.0%}", va="center", fontsize=8.5)
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(names))
    for yi, n in zip(y, names):
        med = results[n]["better"][2]
        axes[2].text(0.02, yi - 0.42, "median " + (f"{med:.0f} trials" if med == med else ">480 trials"), fontsize=8,
                     va="center", color=INK2)
    axes[0].legend(fontsize=8.5, loc="lower right")
    header(fig, "g4. Resampling experiment on runs 2–7",
           "Sessions are built by resampling real trials (with replacement), 4-trial blocks, live = A at the start. "
           "Equal: A and B swapped at random per trial. Worse: A and B swapped on every trial.")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.8, bottom=0.12, wspace=0.12)
    fig.savefig(os.path.join(out, "g4_resampling.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join(PY_DIR, "main", "real_time_data", "sub01", "ses1"))
    ap.add_argument("--out", default=os.path.join(PY_DIR, "results", "graded_scores"))
    ap.add_argument("--n_sim", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()
    T = load_trials(args.session)
    lines = [f"session: {args.session}", f"trials: {len(T)}"]

    # replica check: our vote from predictions.csv == the vote logged online
    agree = np.mean([t[f"vote_{n}"] == t["logged"][n] for t in T for n in "AB"])
    nan_conf = np.mean([math.isnan(t[f"conf_{n}"]) for t in T for n in "AB"])
    short = min(min(t["n_A"], t["n_B"]) for t in T)
    lines.append(f"vote recomputed from predictions.csv matches logged vote: {agree:.4f}; fewest predictions in a window: {short}")
    lines.append(f"trial-model pairs without prob1 (conf = nan): {nan_conf:.3f}")
    # margin >= 0.5 <-> vote right
    mm = np.array([t[f"margin_{n}"] for t in T for n in "AB"])
    ok = np.array([t[f"ok_{n}"] for t in T for n in "AB"])
    lines.append(f"margin>=0.5 agrees with logged right/wrong: {np.mean((mm >= 0.5) == ok):.4f}")

    with open(os.path.join(args.out, "trials.csv"), "w", encoding="utf-8", newline="") as f:
        cols = ["run", "label", "at", "block_end", "ok_A", "ok_B", "margin_A", "margin_B", "conf_A", "conf_B"]
        w = csv.writer(f)
        w.writerow(cols)
        for t in T:
            w.writerow([t[c] if not isinstance(t[c], float) else f"{t[c]:.4f}" for c in cols])

    # information per trial
    z = 1.6449 + 0.8416
    stats = {}
    lines.append("")
    lines.append("effect size per trial d = mean(D)/sd(D), D = B - A; n80 = trials for 80% power (fixed n, one-sided α 0.05)")
    for key in ("ok", "margin", "conf"):
        for scope, sel in (("run1", lambda t: t["run"] == "run1"), ("runs2-7", lambda t: t["run"] != "run1")):
            D = np.array([float(t[f"{key}_B"]) - float(t[f"{key}_A"]) for t in T if sel(t)])
            D = D[~np.isnan(D)]
            d = D.mean() / D.std(ddof=1)
            row = {"mean": D.mean(), "sd": D.std(ddof=1), "d": d, "n80": (z / d) ** 2 if d > 0 else math.inf,
                   "nz": np.mean(np.abs(D) > 1e-9), "n": len(D)}
            if scope == "runs2-7":
                stats[key] = row
            else:
                stats[key + "_run1"] = row
            lines.append(f"  {key:6s} {scope:8s} n={row['n']:3d} mean={row['mean']:+.4f} sd={row['sd']:.4f} d={d:+.3f} "
                         f"nonzero={row['nz']:.2f} n80={row['n80']:.0f}")
    # per-run d for the margin and right/wrong
    lines.append("per run d (right/wrong | margin):")
    for r in sorted({t['run'] for t in T}, key=lambda r: int(r[3:])):
        ds = []
        for key in ("ok", "margin"):
            D = np.array([float(t[f"{key}_B"]) - float(t[f"{key}_A"]) for t in T if t["run"] == r])
            ds.append(D.mean() / D.std(ddof=1))
        lines.append(f"  {r}: {ds[0]:+.3f} | {ds[1]:+.3f}")

    # graded SPRT: sigma fixed from run 1 (not the runs we evaluate), delta = 0.25 sigma
    sig_m = stats["margin_run1"]["sd"]
    rules = [
        (SignSPRT(0.7, 0.05, 0.2, "Right/wrong SPRT, p1=0.7, α=0.05"), "ok"),
        (SignSPRT(0.7, 0.10, 0.2, "Right/wrong SPRT, p1=0.7, α=0.10"), "ok"),
        (GaussSPRT(0.25 * sig_m, sig_m, 0.05, 0.2, "Margin SPRT, δ=0.25σ, α=0.05"), "margin"),
        (GaussSPRT(0.25 * sig_m, sig_m, 0.10, 0.2, "Margin SPRT, δ=0.25σ, α=0.10"), "margin"),
    ]
    lines.append("")
    lines.append(f"margin SPRT: sigma = sd of D in run 1 = {sig_m:.4f}; delta = 0.25 sigma = {0.25 * sig_m:.4f}")
    lines.append("replay on the recording (t = trial index at block end):")
    blk = [t["block_end"] for t in T]
    for rule, key in rules:
        dA = np.array([float(t[f"{key}_A"]) for t in T])
        dB = np.array([float(t[f"{key}_B"]) for t in T])
        _, _, ev = run_rule(rule, dA, dB, blk)
        lines.append(f"  {rule.name}: " + ", ".join(f"{k}@{tt}({lv})" for tt, k, lv, _ in ev))
    g3_replay(T, [rules[0], rules[2], rules[3]], args.out)

    # resampling experiment on runs 2-7
    rng = np.random.default_rng(0)
    pool = [t for t in T if t["run"] != "run1"]
    n_trials = 480
    idx = rng.integers(0, len(pool), size=(args.n_sim, n_trials))
    swap_rand = rng.random((args.n_sim, n_trials)) < 0.5
    results = {}
    lines.append("")
    lines.append(f"resampling (runs 2-7 pool of {len(pool)} trials, {args.n_sim} sessions x {n_trials} trials, live A at start):")
    lines.append("  rule | equal: P(sw<=48) P(sw<=480) | B worse: P(sw<=48) P(sw<=480) | B better: P(sw<=48) P(sw<=480) median")
    for rule, key in rules + [(GaussSPRT(0.15 * sig_m, sig_m, 0.10, 0.2, "Margin SPRT, δ=0.15σ, α=0.10"), "margin"),
                              (GaussSPRT(0.35 * sig_m, sig_m, 0.10, 0.2, "Margin SPRT, δ=0.35σ, α=0.10"), "margin")]:
        A = np.array([float(t[f"{key}_A"]) for t in pool])[idx]
        B = np.array([float(t[f"{key}_B"]) for t in pool])[idx]
        res = {}
        for world in ("null", "worse", "better"):
            if world == "null":
                a, b = np.where(swap_rand, B, A), np.where(swap_rand, A, B)
            elif world == "worse":
                a, b = B, A
            else:
                a, b = A, B
            f = first_switch_batch(rule, a, b)
            sw = f >= 0
            med = float(np.median(np.where(sw, f, np.inf))) if sw.mean() >= 0.5 else float("nan")
            res[world] = (np.mean(sw & (f <= 48)), sw.mean(), med)
        results[rule.name] = res
        lines.append(f"  {rule.name:32s} | {res['null'][0]:.3f} {res['null'][1]:.3f} | {res['worse'][0]:.3f} {res['worse'][1]:.3f}"
                     f" | {res['better'][0]:.3f} {res['better'][1]:.3f} {res['better'][2]:.0f}")
    g1_validity(T, args.out)
    g2_information(T, args.out, stats)
    g4_resampling({n: results[n] for n in [r.name for r, _ in rules]}, args.out)

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
