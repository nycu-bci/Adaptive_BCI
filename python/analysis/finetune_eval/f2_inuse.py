# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f2_inuse.py --subs sub01,sub05
# other sessions: ... f2_inuse.py --subs sub02,sub03,sub04,sub05 --name _sub02-05
"""
F2, in-use windows: wrong fires and missed fires of the current gate. Same models as f2_analyze (parent = online checkpoint at the block, child =
its V0 finetune on H4-correct trials, 5 seeds), but each block is judged only on the trials where its choice would be in use: from the next block
through the next block the gate really fired on (or the session end), capped at 48 trials (the trials F2 scored).
effect = child - parent accuracy on that window (5-seed mean); effect_trials = effect x n.
Clear up: effect_trials >= MIN_TRIALS and >= MIN_SEEDS of 5 seeds > 0; clear down: the mirror; otherwise unclear.
  fired:   up = correct fire, down = WRONG FIRE   (keeping the old model would have been better)
  skipped: up = MISSED FIRE (a finetune would have been better), down = correct skip
Windows of fired blocks tile the session; windows of consecutive skipped blocks overlap (they end at the same fire).
Selection: a window that ends at the next fire ends on a block chosen for the live model's errors, which favours the finetune at skipped blocks.
Every number is also given on the window without that block (x_ columns; windows ending at the session end or the cap are kept whole); a window
of only that block has no trials left. Trust a conclusion only where both versions agree.
Outputs in results/finetune_eval/f2/: inuse_per_decision.csv, inuse_pooled.csv, inuse_summary.txt (with --name X: inuse_per_decisionX.csv, ...)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2_analyze as A
CF = A.CF

MIN_TRIALS = 1.0
MIN_SEEDS = 4
LABELS = {1: {1: "correct fire", -1: "wrong fire", 0: "unclear"}, 0: {1: "missed fire", -1: "correct skip", 0: "unclear"}}


def inuse_seqs(last_seqs, fired, i, n_total, cap=CF.TEST_N):
    """Trials after decision i (session order) until the end of the next fired block, or the session end; at most cap."""
    later = np.flatnonzero(np.asarray(fired)[i + 1:])
    end = last_seqs[i + 1 + later[0]] if len(later) else n_total - 1
    return np.arange(last_seqs[i] + 1, min(end, last_seqs[i] + cap) + 1, dtype=int)


def direction(effect_trials, seed_effects):
    """+1 clear up, -1 clear down, 0 unclear."""
    s = np.asarray(seed_effects, float)
    if effect_trials >= MIN_TRIALS and (s > 0).sum() >= MIN_SEEDS:
        return 1
    if effect_trials <= -MIN_TRIALS and (s < 0).sum() >= MIN_SEEDS:
        return -1
    return 0


def without_selecting_block(win, last, fired, i):
    """The window minus its last block when that block is the next fired block (chosen for the live model's errors); else the whole window."""
    later = np.flatnonzero(np.asarray(fired)[i + 1:])
    if len(later) and len(win) and last[i + 1 + later[0]] == win[-1]:
        return win[win <= last[i + later[0]]]
    return win


def score(lab, pv, M, j, fired, pre):
    e = A.decision_effect(lab[j], pv[j], M[:, j])
    rec = {pre + "n": e["n"], pre + "keep_acc": e["parent_acc"], pre + "ft_acc": e["child_acc"], pre + "effect": e["effect"],
           pre + "effect_trials": e["effect"] * e["n"] if e["n"] else np.nan}
    for si, v in enumerate(e["seed_effects"]):
        rec[f"{pre}effect_s{si}"] = v
    rec[pre + "direction"] = direction(rec[pre + "effect_trials"], e["seed_effects"]) if e["n"] else 0
    rec[pre + "label"] = LABELS[fired][rec[pre + "direction"]] if e["n"] else "no trials"
    return rec


VERSIONS = (("", "full window"), ("x_", "window without the selecting block"))


def build(subs):
    dec = pd.read_csv(os.path.join(A.OUT, "decisions.csv")).sort_values("d")
    order = {}
    for (s, m), g in dec.groupby(["session", "model"]):
        order[(s, m)] = (g.d.to_list(), g.last_seq.to_numpy(int), g.fired.to_numpy(int))
    recs = []
    for d, lab, tag, pv, M, online_ok in A.decision_votes(subs):
        ds, last, fired = order[(d.session, d.model)]
        i = ds.index(d.d)
        win = inuse_seqs(last, fired, i, len(lab))
        if len(win) == 0:
            continue
        rec = {k: d[k] for k in ("session", "model", "d", "run", "block", "n_trials", "is_cal", "errors", "run_acc", "fired", "last_seq")}
        rec.update(score(lab, pv, M, win, int(d.fired), ""))
        if rec["n"] == 0:
            continue
        later = np.flatnonzero(fired[i + 1:])
        oo = online_ok[win]
        rec.update(win_first=int(win[0]), win_last=int(win[-1]), n_win=len(win), to_end=int(len(later) == 0),
                   capped=int(len(win) == CF.TEST_N and (len(later) == 0 or last[i + 1 + later[0]] > win[-1])),
                   online_acc=float(np.nanmean(oo)) if (~np.isnan(oo)).any() else np.nan)
        rec.update(score(lab, pv, M, without_selecting_block(win, last, fired, i), int(d.fired), "x_"))
        recs.append(rec)
    return pd.DataFrame(recs)


def summarize(g, rng, pre=""):
    """Pooled numbers of one group of decisions (all fired or all skipped); pre = column prefix of the window version."""
    g = g[g[pre + "n"] > 0]
    n = g[pre + "n"]
    lo, hi = A.boot_ci(g[pre + "effect"], n, rng)
    dr = g[pre + "direction"]
    return dict(n_blocks=len(g), n_trials=int(n.sum()), keep_acc=A.pooled(g[pre + "keep_acc"], n), ft_acc=A.pooled(g[pre + "ft_acc"], n),
                effect=A.pooled(g[pre + "effect"], n), ci_lo=lo, ci_hi=hi, trials_gained=float(g[pre + "effect_trials"].sum()),
                up=int((dr == 1).sum()), down=int((dr == -1).sum()), unclear=int((dr == 0).sum()))


def report(df, subs):
    rng = np.random.default_rng(0)
    L, rows = [], []
    L.append("F2 in-use windows: wrong fires and missed fires of the current gate (V0 finetune on H4-correct trials, 5 seeds).")
    L.append(f"window = trials from the next block to the end of the next fired block (or the session end), capped at {CF.TEST_N}; sessions {', '.join(subs)}")
    L.append(f"effect = finetuned - kept accuracy on the window (5-seed mean), pooled trial-weighted, bootstrap {A.N_BOOT} over blocks (seed 0).")
    L.append(f"clear = |5-seed mean| >= {MIN_TRIALS:g} trial and >= {MIN_SEEDS}/5 seeds on the same side. Skipped windows overlap: their trials gained double-count.")
    L.append(f"windows: fired median {df[df.fired == 1].n.median():.0f} trials, skipped median {df[df.fired == 0].n.median():.0f}; "
             f"capped at {CF.TEST_N}: {int(df.capped.sum())} blocks")
    L.append("")
    units = [(f"{s} {m}", df[(df.session == s) & (df.model == m)]) for s in subs for m in "AB"]
    units += [(name, df[df.model.isin(ms)]) for name, ms in A.GROUPS]
    for fired, title, cols in ((1, "Wrong fires (blocks the gate fired on): keep the old model vs the finetune the gate made", ("correct", "WRONG")),
                               (0, "Missed fires (blocks the gate skipped): keep (what the gate did) vs a finetune", ("MISSED", "correct skip"))):
        for pre, vname in VERSIONS:
            L.append(f"== {title}; {vname} ==")
            L.append(f"  {'unit':10} {'blocks':>6} {'trials':>6}  keep   ft    effect  95% CI             gained  {cols[0]:>8} {cols[1]:>12} unclear  rate")
            for name, u in units:
                g = u[(u.fired == fired) & (u[pre + "n"] > 0)]
                if g.empty:
                    L.append(f"  {name:10} {0:6}")
                    continue
                r = summarize(g, rng, pre)
                bad = r["down"] if fired else r["up"]
                rows.append(dict(window=vname, unit=name, fired=fired, **r, rate=bad / r["n_blocks"]))
                L.append(f"  {name:10} {r['n_blocks']:6} {r['n_trials']:6}  {r['keep_acc']:.3f} {r['ft_acc']:.3f} {A.fmt(r['effect'])} "
                         f"[{A.fmt(r['ci_lo'])}, {A.fmt(r['ci_hi'])}] {r['trials_gained']:+7.1f}  {r['up']:8} {r['down']:12} {r['unclear']:7}  "
                         f"{bad}/{r['n_blocks']} = {bad / r['n_blocks']:.2f}")
            L.append(f"  rate = {'wrong fires / fired blocks' if fired else 'missed fires / skipped blocks'}; gained = sum of (finetuned - kept) correct trials")
            L.append("")
    L.append("== Wrong / missed fires by window length (A+B) ==")
    for fired, bad, name in ((1, -1, "wrong fire"), (0, 1, "missed fire")):
        g = df[df.fired == fired]
        parts = []
        for lo, hi in ((4, 4), (5, 12), (13, 24), (25, 48)):
            h = g[(g.n >= lo) & (g.n <= hi)]
            if len(h):
                parts.append(f"{lo}-{hi} trials: {int((h.direction == bad).sum())}/{len(h)} {name}, effect {A.fmt(A.pooled(h.effect, h.n))}")
        L.append(f"  {'fired' if fired else 'skipped':7}: " + " | ".join(parts))
    return L, pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub01,sub05")
    ap.add_argument("--name", default="", help="suffix of the output files, e.g. _sub02-05")
    args = ap.parse_args(argv)
    subs = args.subs.split(",")
    df = build(subs)
    df.to_csv(os.path.join(A.OUT, f"inuse_per_decision{args.name}.csv"), index=False)
    lines, pool = report(df, subs)
    pool.to_csv(os.path.join(A.OUT, f"inuse_pooled{args.name}.csv"), index=False)
    with open(os.path.join(A.OUT, f"inuse_summary{args.name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
