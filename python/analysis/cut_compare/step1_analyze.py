# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/cut_compare/step1_analyze.py
"""
Analysis of the cut comparison (per_trial.csv from step1_cut_compare.py): cut 1 (first 4 s from START) vs cut 2 (4 s before the vote).

Per model (A, B) and pooled (A and B trial pairs stacked; the two models see the same EEG, so the pooled tests are somewhat optimistic):
  - accuracy of Unity's vote per seed and averaged over seeds; difference cut 2 - cut 1
  - exact McNemar (two-sided binomial on the discordant trials) per seed; on the majority outcome over seeds; sign test on the
    seed-averaged per-trial difference (ties dropped)
  - 95% CI of the seed-averaged difference: two-stage bootstrap (resample runs, then trials within the sampled runs), plus a plain trial bootstrap
  - per-run differences, seed spread, secondary scores (vote margin, window accuracy on the held-out trial's own 16 windows)

Outputs in --out: summary.txt, g1_accuracy_by_cut.png, g2_per_run_difference.png, g3_secondary_scores.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from wald_sprt_explainer import INK, INK2, MUTED, GRID, AXIS, MODEL_COLOR, RULE_COLOR, style, header

CUT_COLOR = {1: MUTED, 2: RULE_COLOR[2]}
GROUPS = ["A", "B", "pooled"]
CUT_NAMES = {1: "online as is (~1-5 s after START, = data_*.pt)", 2: "4 s before the vote (~0.25-4.25 s)", 3: "intended (0-4 s from START)"}
CUT_SHORT = {1: "online", 2: "vote_end", 3: "intended"}
LABEL = {1: "cut 1", 2: "cut 2"}  # set in main: the two compared cuts, in the slots 1 (baseline) and 2 (new)


def mcnemar(ok1, ok2):
    """b = only cut 1 right, c = only cut 2 right, two-sided exact p."""
    b, c = int(np.sum(ok1 & ~ok2)), int(np.sum(~ok1 & ok2))
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


def sign_test(d):
    pos, neg = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
    return pos, neg, (binomtest(min(pos, neg), pos + neg, 0.5).pvalue if pos + neg else 1.0)


def two_stage_boot(d, run_ids, n_boot, rng):
    """Mean of d over a bootstrap that resamples runs, then trials inside each sampled run."""
    groups = [np.flatnonzero(run_ids == r) for r in np.unique(run_ids)]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([rng.choice(groups[g], len(groups[g])) for g in pick])
        out[b] = d[idx].mean()
    return out


def trial_boot(d, n_boot, rng):
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    return d[idx].mean(1)


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def load_arrays(df, seeds):
    """-> trials frame (run, trial_seq, label), and arrays[(model, cut, key)] of shape (S, T) in trial order."""
    base = df[(df.model == "A") & (df.cut == 1) & (df.seed == seeds[0])].sort_values(["run_n", "trial_seq"])
    order = base[["run", "trial_seq"]].reset_index(drop=True)
    arr = {}
    for m in "AB":
        for cut in (1, 2):
            for key in ("vote_ok", "margin", "window_acc"):
                rows = []
                for s in seeds:
                    d = df[(df.model == m) & (df.cut == cut) & (df.seed == s)].sort_values(["run_n", "trial_seq"])
                    assert len(d) == len(order) and (d.trial_seq.to_numpy() == order.trial_seq.to_numpy()).all()
                    rows.append(d[key].to_numpy(float))
                arr[(m, cut, key)] = np.array(rows)
    return base[["run", "run_n", "trial_seq", "label"]].reset_index(drop=True), arr


def group_array(arr, g, cut, key):
    """(S, T) for A or B; (S, 2T) with both stacked for pooled."""
    if g == "pooled":
        return np.concatenate([arr[("A", cut, key)], arr[("B", cut, key)]], axis=1)
    return arr[(g, cut, key)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.RESULTS_DIR)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair", default="3,2", help="baseline,new cut ids (1 online as is, 2 4 s before the vote, 3 intended 0-4 s)")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    base_cut, new_cut = (int(x) for x in args.pair.split(","))
    LABEL[1], LABEL[2] = CUT_SHORT[base_cut], CUT_SHORT[new_cut]
    src = args.out
    out = os.path.join(src, f"{CUT_SHORT[base_cut]}_vs_{CUT_SHORT[new_cut]}")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(src, "per_trial.csv"))
    df = df[df.cut.isin([base_cut, new_cut])].copy()
    df["cut"] = df.cut.map({base_cut: 1, new_cut: 2})  # the rest of the script compares slot 1 (baseline) with slot 2 (new)
    df["run_n"] = df.run.str.replace("run", "").astype(int)
    meta = json.load(open(os.path.join(src, "step1_meta.json"))) if os.path.exists(os.path.join(src, "step1_meta.json")) else {}
    n_trials = df[(df.model == "A") & (df.cut == 1) & (df.seed == df.seed.min())].shape[0]
    full = [s for s in sorted(df.seed.unique()) if all(len(df[(df.model == m) & (df.cut == c) & (df.seed == s)]) == n_trials for m in "AB" for c in (1, 2))]
    seeds = full
    T, arr = load_arrays(df, seeds)
    S, n = len(seeds), len(T)
    run_ids = T.run_n.to_numpy()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"Cut comparison: cut 1 = {LABEL[1]}: {CUT_NAMES[base_cut]}  vs  cut 2 = {LABEL[2]}: {CUT_NAMES[new_cut]}")
    say("All spans are wall-clock time after START; the vote is at ~START + 4.25 s. Below, 'cut 1' / 'cut 2' always mean these two.")
    say(f"cut 2 alignment: {meta.get('cut2_align', '?')} (wall = lag from step 0a applied, so no EEG the predictor could not have had at AT; file = file clock as is)")
    say(f"lags used (s): {meta.get('lags', '?')}")
    say(f"design: leave-one-run-out over runs 1-7, {S} seeds {seeds}, {n} held-out trials per model and cut, epochs {meta.get('epochs', '?')}, "
        f"batch {meta.get('batch_size', '?')}, lr {meta.get('lr', '?')} (x2 inside the trainer), failed-trial weight {meta.get('fail_weight', '?')}")
    say("score = Unity vote (last 100 predictions before AT, weights 1-5) of the model trained on the other 6 runs; accuracy = share of trials voted correctly")
    say()

    ok = {g: {c: group_array(arr, g, c, "vote_ok") for c in (1, 2)} for g in GROUPS}
    T_run = np.concatenate([run_ids, run_ids])
    results = {}
    say("== 1. Vote accuracy by cut ==")
    say("group   cut   acc (mean over seeds)   per-seed accuracies                      sd over seeds   min-max")
    for g in GROUPS:
        for c in (1, 2):
            a = ok[g][c].mean(1)
            say(f"{g:6s}  {c}     {a.mean():.4f}                  {' '.join(f'{x:.4f}' for x in a)}   {a.std(ddof=1):.4f}          {a.min():.4f}-{a.max():.4f}")
    say()

    say("== 2. Difference cut 2 - cut 1 (accuracy, percentage points) ==")
    say("group  diff(seed-avg)  95% CI two-stage (runs, trials)  95% CI plain trial bootstrap  | per-seed diffs (pp)  mean+-sd | McNemar per seed p: range")
    for g in GROUPS:
        d_trial = ok[g][2].mean(0) - ok[g][1].mean(0)
        rid = T_run if g == "pooled" else run_ids
        if g == "pooled":
            dt = (ok["A"][2].mean(0) - ok["A"][1].mean(0) + ok["B"][2].mean(0) - ok["B"][1].mean(0)) / 2
            b2 = two_stage_boot(dt, run_ids, args.n_boot, rng)
            b1 = trial_boot(dt, args.n_boot, rng)
        else:
            b2 = two_stage_boot(d_trial, run_ids, args.n_boot, rng)
            b1 = trial_boot(d_trial, args.n_boot, rng)
        per_seed = ok[g][2].mean(1) - ok[g][1].mean(1)
        pvals = [mcnemar(ok[g][1][s] > 0.5, ok[g][2][s] > 0.5) for s in range(S)]
        maj1, maj2 = ok[g][1].sum(0) > S / 2, ok[g][2].sum(0) > S / 2
        mb, mc, mp = mcnemar(maj1, maj2)
        sp, sn, sp_p = sign_test(d_trial)
        lo2, hi2 = ci(b2)
        lo1, hi1 = ci(b1)
        results[g] = dict(diff=d_trial.mean(), ci2=(lo2, hi2), ci1=(lo1, hi1), per_seed=per_seed, p_seed=[p[2] for p in pvals], maj=(mb, mc, mp), sign=(sp, sn, sp_p))
        say(f"{g:6s}  {100 * d_trial.mean():+6.2f}          [{100 * lo2:+6.2f}, {100 * hi2:+6.2f}]                 [{100 * lo1:+6.2f}, {100 * hi1:+6.2f}]"
            f"                  | {' '.join(f'{100 * x:+.2f}' for x in per_seed)}  {100 * per_seed.mean():+.2f}+-{100 * per_seed.std(ddof=1):.2f} | "
            f"{min(p[2] for p in pvals):.3f}-{max(p[2] for p in pvals):.3f}")
    say()
    say("Tests on the vote outcome (two-sided exact binomial on discordant trials; b = only cut 1 right, c = only cut 2 right):")
    say("  primary = majority outcome over the seeds (a trial counts as right for a cut when >= half+ of the seeds vote right) - one outcome per trial and cut,")
    say("  so the seeds do not inflate n. Per-seed tests are listed because the seeds are not independent (same data) and only show the noise.")
    for g in GROUPS:
        r = results[g]
        say(f"  {g:6s} majority over seeds: b={r['maj'][0]} c={r['maj'][1]} p={r['maj'][2]:.4f}   | sign test on seed-averaged per-trial difference: "
            f"cut2 better in {r['sign'][0]} trials, cut1 better in {r['sign'][1]} (ties dropped) p={r['sign'][2]:.4f}")
        for s in range(S):
            b, c, p = mcnemar(ok[g][1][s] > 0.5, ok[g][2][s] > 0.5)
            say(f"         seed {seeds[s]}: b={b:3d} c={c:3d} p={p:.4f}")
    say()

    say("== 3. Per run (seed-averaged accuracy, held-out run) ==")
    say("run   model   n   acc cut1   acc cut2   diff (pp)   per-seed diff range (pp)")
    runs = sorted(T.run_n.unique())
    per_run = {}
    for m in "AB":
        for r in runs:
            sel = run_ids == r
            a1, a2 = ok[m][1][:, sel], ok[m][2][:, sel]
            ps = a2.mean(1) - a1.mean(1)
            per_run[(m, r)] = (a1.mean(), a2.mean(), ps)
            say(f"run{r}  {m}     {sel.sum():3d}   {a1.mean():.4f}     {a2.mean():.4f}     {100 * (a2.mean() - a1.mean()):+6.2f}      {100 * ps.min():+.2f} .. {100 * ps.max():+.2f}")
    for m in "AB":
        d = np.array([per_run[(m, r)][1] - per_run[(m, r)][0] for r in runs])
        say(f"{m}: cut 2 better in {int((d > 0).sum())} of {len(runs)} runs, worse in {int((d < 0).sum())}, equal in {int((d == 0).sum())}")
    say()

    say("== 4. Seed spread vs cut difference ==")
    for g in GROUPS:
        a1, a2 = ok[g][1].mean(1), ok[g][2].mean(1)
        per_seed = a2 - a1
        t1, t2 = ok[g][1], ok[g][2]
        unan1 = np.mean((t1.min(0) == t1.max(0)))
        unan2 = np.mean((t2.min(0) == t2.max(0)))
        say(f"  {g:6s} same cut, different seeds: accuracy sd {100 * a1.std(ddof=1):.2f} pp (cut 1), {100 * a2.std(ddof=1):.2f} pp (cut 2); "
            f"range {100 * (a1.max() - a1.min()):.2f} / {100 * (a2.max() - a2.min()):.2f} pp. Cut difference {100 * (a2.mean() - a1.mean()):+.2f} pp "
            f"(per-seed {100 * per_seed.min():+.2f} .. {100 * per_seed.max():+.2f}, sd {100 * per_seed.std(ddof=1):.2f}). "
            f"Trials with the same vote outcome in all seeds: {100 * unan1:.0f}% (cut 1), {100 * unan2:.0f}% (cut 2)")
    say()

    say("== 5. Secondary scores (seed-averaged per trial) ==")
    say("score         group   cut1     cut2     diff     95% CI two-stage")
    sec = {}
    for key, name in (("margin", "vote margin"), ("window_acc", "window acc")):
        for g in GROUPS:
            v1, v2 = group_array(arr, g, 1, key).mean(0), group_array(arr, g, 2, key).mean(0)
            if g == "pooled":
                dt = ((arr[("A", 2, key)] - arr[("A", 1, key)]).mean(0) + (arr[("B", 2, key)] - arr[("B", 1, key)]).mean(0)) / 2
            else:
                dt = v2 - v1
            lo, hi = ci(two_stage_boot(dt, run_ids, args.n_boot, rng))
            sec[(key, g)] = (v1.mean(), v2.mean())
            say(f"{name:12s}  {g:6s}  {v1.mean():.4f}   {v2.mean():.4f}   {dt.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]")
    say("  vote margin = weighted share (weights 1-5) of the 100 window predictions that equal the label (the vote is right at >= 0.5)")
    say("  window acc = share of the held-out trial's own 16 windows (same cut as the model was trained on) classified correctly")
    say()

    say("== 6. Reference points (not part of the comparison) ==")
    bp = os.path.join(src, "per_trial_base.csv")
    if os.path.exists(bp):
        b = pd.read_csv(bp)
        for m in "AB":
            x = b[(b.model == m) & (b.cut == b.cut.min())]  # the base model's vote does not depend on the cut
            say(f"  base checkpoint {'cA_000' if m == 'A' else 'cB_000'} (no finetuning), same test windows: vote accuracy {x.vote_ok.mean():.4f}")
    v0 = os.path.join(C.RESULTS_DIR, "step0a_votes.csv")
    if os.path.exists(v0):
        v = pd.read_csv(v0)
        for m in "AB":
            x = v[v.model == m]
            say(f"  online as it ran (incrementally finetuned chain, logged votes): {m} accuracy {np.mean(x.vote_logged == x.label):.4f} over {len(x)} trials")
    say()
    say("Note: every trial of the 6 training runs is used (no 48-trial FIFO replay limit as online); failed-trial flags come from the Unity result for both models.")
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    style()
    fig_accuracy(ok, results, seeds, out)
    fig_per_run(per_run, runs, results, out)
    fig_secondary(arr, out)


def fig_accuracy(ok, results, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    rng = np.random.default_rng(1)
    for i, g in enumerate(GROUPS):
        for c, off in ((1, -0.17), (2, 0.17)):
            a = ok[g][c].mean(1)
            ax.scatter(i + off + rng.uniform(-0.05, 0.05, len(a)), a, s=22, color=CUT_COLOR[c], alpha=0.8, zorder=3)
            ax.hlines(a.mean(), i + off - 0.13, i + off + 0.13, color=INK, lw=1.8, zorder=4)
    ax.set_xticks(range(3), ["A", "B", "A + B"])
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set_ylabel("Vote accuracy (held-out runs)")
    ax.set_title("g1a. Accuracy by cut; dots = seeds, bar = mean")
    ax.scatter([], [], color=CUT_COLOR[1], label=LABEL[1])
    ax.scatter([], [], color=CUT_COLOR[2], label=LABEL[2])
    ax.legend(fontsize=8.5, loc="lower right")
    ax = axes[1]
    for i, g in enumerate(GROUPS):
        r = results[g]
        ax.scatter(np.full(len(r["per_seed"]), i) + np.linspace(-0.12, 0.12, len(r["per_seed"])), 100 * r["per_seed"], s=22, color=INK2, alpha=0.6, zorder=3)
        ax.errorbar(i + 0.3, 100 * r["diff"], yerr=[[100 * (r["diff"] - r["ci2"][0])], [100 * (r["ci2"][1] - r["diff"])]], fmt="D", color=INK, ms=6, capsize=4, zorder=4)
        ax.text(i + 0.3, 100 * r["ci2"][1] + 0.4, f"p={r['maj'][2]:.2f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(3), ["A", "B", "A + B"])
    ax.set_ylabel(f"{LABEL[2]} − {LABEL[1]} (percentage points)")
    ax.set_title("g1b. Difference; dots = seeds, diamond = mean with 95% CI")
    header(fig, f"Does the cut matter? Unity-vote accuracy of models trained on {LABEL[1]} vs {LABEL[2]}",
           f"Leave-one-run-out over runs 1-7, {len(seeds)} seeds, base checkpoints cA_000 / cB_000. CI: bootstrap over runs then trials. p: exact McNemar on the majority outcome over seeds.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.1, wspace=0.22)
    fig.savefig(os.path.join(out, "g1_accuracy_by_cut.png"), dpi=150)
    plt.close(fig)


def fig_per_run(per_run, runs, results, out):
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    w = 0.38
    x = np.arange(len(runs))
    for k, m in enumerate("AB"):
        d = np.array([per_run[(m, r)][1] - per_run[(m, r)][0] for r in runs]) * 100
        lo = np.array([per_run[(m, r)][2].min() for r in runs]) * 100
        hi = np.array([per_run[(m, r)][2].max() for r in runs]) * 100
        ax.bar(x + (k - 0.5) * w, d, w * 0.92, color=MODEL_COLOR[m], label=f"{m} (mean over runs {100 * results[m]['diff']:+.2f} pp)")
        ax.errorbar(x + (k - 0.5) * w, d, yerr=[d - lo, hi - d], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x, [f"run{r}" for r in runs])
    ax.set_ylabel(f"{LABEL[2]} − {LABEL[1]} (percentage points)")
    ax.legend(fontsize=9)
    header(fig, f"Per held-out run: change in vote accuracy from {LABEL[1]} to {LABEL[2]}",
           "Bar = mean over seeds; whiskers = range over the per-seed differences (the training noise a single seed carries).")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.1)
    fig.savefig(os.path.join(out, "g2_per_run_difference.png"), dpi=150)
    plt.close(fig)


def fig_secondary(arr, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for ax, m in zip(axes[:2], "AB"):
        x, y = arr[(m, 1, "margin")].mean(0), arr[(m, 2, "margin")].mean(0)
        ax.scatter(x, y, s=12, color=MODEL_COLOR[m], alpha=0.45, lw=0)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9)
        ax.axvline(0.5, color=GRID, lw=0.8)
        ax.axhline(0.5, color=GRID, lw=0.8)
        ax.scatter([x.mean()], [y.mean()], s=70, color=INK, marker="D", zorder=4)
        ax.set_xlabel(f"Vote margin, {LABEL[1]} (seed mean)")
        ax.set_ylabel(f"Vote margin, {LABEL[2]} (seed mean)")
        ax.set_title(f"g3{'ab'['AB'.index(m)]}. Model {m}: margin per trial (diamond = mean)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax = axes[2]
    for i, m in enumerate("AB"):
        for c, off in ((1, -0.18), (2, 0.18)):
            v = arr[(m, c, "window_acc")]
            ax.bar(i + off, v.mean(), 0.32, color=CUT_COLOR[c], label=LABEL[c] if i == 0 else None)
            ax.text(i + off, v.mean() + 0.005, f"{v.mean():.3f}", ha="center", fontsize=8.5)
    ax.axhline(0.5, color=AXIS, lw=0.9)
    ax.set_xticks([0, 1], ["A", "B"])
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("Window accuracy (held-out trial's own 16 windows)")
    ax.set_title("g3c. Window accuracy by cut")
    ax.legend(fontsize=8.5)
    header(fig, "Secondary scores", "Vote margin = weighted share of the last 100 predictions that are right (vote right at 0.5). Window accuracy is on the same cut the model was trained on.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.12, wspace=0.25)
    fig.savefig(os.path.join(out, "g3_secondary_scores.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
