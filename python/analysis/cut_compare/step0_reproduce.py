# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/cut_compare/step0_reproduce.py
"""
Step 0a: does an offline predictor reproduce the online predictions?

For every logged prediction of A and B (runs 1-7) the 1 s window ending at that time is re-predicted with the checkpoint that
was live then (base cA_000 / cB_000, then each block's ckpt_<model> from its swap_time). The EEG file clock differs from
time.time() by a per-run lag (the CSV starts at the first LSL sample, 'Record datetime' is taken before the inlet connects),
so the lag is searched per run on a subsample (coarse over -1..3 s, then 2 ms steps around the peak) and then fixed.
Also checks the Unity vote per trial and per model (last 100 predictions before AT) against the END line of the log.

Outputs in --out: step0a_summary.txt, step0a_lag_curves.png, step0a_accuracy_over_trial.png, step0a_lags.csv, step0a_votes.csv,
step0a_lags.json (the per-run lags used by step 1)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from wald_sprt_explainer import INK, INK2, MUTED, MODEL_COLOR, AXIS, style, header

PASS_PRED, PASS_VOTE = 0.95, 0.98


def eligible_rows(run, name, timeline, lag, guard=1.0):
    """Rows with a prob1, a full 500-sample window at `lag`, and not within `guard` s of a weight swap."""
    P = run.preds[name]
    ends = run.window_ends(P[:, 0], lag)
    ok = ~np.isnan(P[:, 2]) & (ends >= C.config.BUFFER_SIZE) & (ends <= len(run.ts))
    swaps = np.array([e[0] for e in timeline[1:]])
    if len(swaps):
        d = np.abs(P[:, 0][:, None] - swaps[None, :]).min(1)
        ok &= d > guard
    return np.flatnonzero(ok)


def agreement_curve(run, timelines, lags, n, rng, chunk=C.CHUNK):
    """{name: array of agreement per lag} on n random eligible rows per model (same rows for every lag)."""
    out = {}
    for name in "AB":
        rows = eligible_rows(run, name, timelines[name], 1.0)
        rows = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
        truth = run.preds[name][rows, 1].astype(int)
        out[name] = np.array([(C.predict_rows(run, timelines[name], name, rows, lag, chunk)[0] == truth).mean() for lag in lags])
    return out


def smooth(y, k=5):
    return np.convolve(y, np.ones(k) / k, mode="same")


def find_lag(run, timelines, rng, n_coarse, n_fine, log):
    """Plain single lag (chunk=1) and lag + 50-sample chunk delivery (the model used downstream)."""
    coarse_lags = np.round(np.arange(-1.0, 3.0001, 0.05), 3)
    state = rng.bit_generator.state
    plain = agreement_curve(run, timelines, coarse_lags, n_coarse, rng, chunk=1)
    rng.bit_generator.state = state
    coarse = agreement_curve(run, timelines, coarse_lags, n_coarse, rng)
    both = (coarse["A"] + coarse["B"]) / 2
    centre = coarse_lags[int(np.argmax(both))]
    fine_lags = np.round(np.arange(centre - 0.1, centre + 0.1001, 1.0 / C.FS), 4)
    fine = agreement_curve(run, timelines, fine_lags, n_fine, rng)
    fboth = (fine["A"] + fine["B"]) / 2
    k = int(np.argmax(smooth(fboth)))
    pb = (plain["A"] + plain["B"]) / 2
    log(f"  {run.name}: plain single lag: best {pb.max():.3f} at {coarse_lags[int(np.argmax(pb))]:+.2f} s (50 ms grid). "
        f"With {C.CHUNK}-sample chunks: coarse peak {centre:+.2f} s ({both.max():.3f}); 2 ms steps: smoothed peak {fine_lags[k]:+.3f} s "
        f"(A {fine['A'][k]:.3f}, B {fine['B'][k]:.3f}); raw max {fboth.max():.3f} at {fine_lags[int(np.argmax(fboth))]:+.3f}")
    return {"coarse_lags": coarse_lags, "coarse": coarse, "plain": plain, "fine_lags": fine_lags, "fine": fine, "lag": float(fine_lags[k])}


def chunk_diagnostic(runs, timelines, rng, say, n_rows=120, half=100, which=(2, 4, 6)):
    """Recover, for logged predictions, the window end that reproduces the logged prob1 (1-sample steps around the lag-1.0 end),
    to show that the preprocessing/weights are reproduced exactly and that the live buffer ends on chunk boundaries."""
    say("== Diagnostic: which window end reproduces each logged prob1? (1-sample steps, +-%d samples around the plain-lag end) ==" % half)
    say("  control = same search with the previous checkpoint of the chain (wrong weights); a real match is |dprob1| < 2e-4 (prob1 is logged with 4 decimals)")
    say("run model  rows  matched(<2e-4)  matched ends on a multiple of 50  median min|dprob1|  control: matched   control median min|dprob1|")
    tot = {"n": 0, "m": 0, "q": 0}
    for r in runs:
        if r.run not in which:
            continue
        for m in "AB":
            P = r.preds[m]
            tl = timelines[m]
            rows = np.sort(rng.choice(eligible_rows(r, m, tl, 1.0), n_rows, replace=False))
            t = P[rows, 0]
            k = np.searchsorted(np.array([e[0] for e in tl]), t, side="right") - 1
            e0 = r.window_ends(t, 1.0, chunk=1)
            shifts = np.arange(-half, half + 1)
            errs, ctrl = np.zeros((len(rows), len(shifts))), np.zeros((len(rows), len(shifts)))
            for ki in np.unique(k):
                sel = np.flatnonzero(k == ki)
                model, pos = C.load_model(tl[ki][1])
                for j, sh in enumerate(shifts):
                    _, pr = C.predict_ends(model, pos, r.eeg, e0[sel] + sh)
                    errs[sel, j] = np.abs(pr - P[rows[sel], 2])
            # control: previous checkpoint of the chain for rows whose checkpoint has a predecessor
            ctrl_ok = np.zeros(len(rows), bool)
            ctrl_min = np.full(len(rows), np.nan)
            for ki in np.unique(k):
                if ki < 1 or tl[ki][1] == tl[ki - 1][1]:
                    continue
                sel = np.flatnonzero(k == ki)
                cm, cp = C.load_model(tl[ki - 1][1])
                for j, sh in enumerate(shifts):
                    _, pr = C.predict_ends(cm, cp, r.eeg, e0[sel] + sh)
                    ctrl[sel, j] = np.abs(pr - P[rows[sel], 2])
                ctrl_min[sel] = ctrl[sel].min(1)
            mn = errs.min(1)
            best_end = e0 + shifts[errs.argmin(1)]
            ok = mn < 2e-4
            q = ok & (best_end % C.CHUNK == 0)
            cm_ok = ~np.isnan(ctrl_min)
            say(f"{r.name}  {m}   {len(rows):4d}  {ok.mean():.3f}           {q.sum() / max(ok.sum(), 1):.3f}                          {np.median(mn):.5f}          "
                + (f"{np.mean(ctrl_min[cm_ok] < 2e-4):.3f}              {np.median(ctrl_min[cm_ok]):.5f}" if cm_ok.any() else "n/a (no predecessor in this run)"))
            tot["n"] += ok.sum()
            tot["m"] += len(rows)
            tot["q"] += q.sum()
    say(f"  all: {tot['n']}/{tot['m']} predictions matched exactly; {tot['q']}/{tot['n']} of those at a multiple of {C.CHUNK} samples")
    say()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=C.SESSION_DIR)
    ap.add_argument("--out", default=C.RESULTS_DIR)
    ap.add_argument("--n_coarse", type=int, default=250, help="predictions per model and run in the coarse lag search")
    ap.add_argument("--n_fine", type=int, default=500, help="predictions per model and run in the 2 ms lag search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="1,2,3,4,5,6,7", help="subset of runs to analyse (debugging); the checkpoint timeline always uses all runs")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    all_runs = [C.Run(r, args.session) for r in C.RUNS]
    timelines = {m: C.checkpoint_timeline(m, all_runs) for m in "AB"}
    runs = [r for r in all_runs if str(r.run) in args.runs.split(",")]
    say(f"device: {C.DEVICE}   session: {args.session}")
    say("Checkpoint timeline: base, then each block's checkpoint from its swap_time")
    for m in "AB":
        say(f"  {m}: " + ", ".join(f"{os.path.basename(p)[:-4]}@{'start' if t == -np.inf else f'{t:.0f}'}" for t, p in timelines[m]))
    say()

    say("== Lag search (EEG file clock vs time.time()) ==")
    say("  lag L: the live buffer at time T holds the samples with file time <= T - rec_epoch - L, cut back to whole 50-sample LSL chunks")
    say("  (the plain single-lag model without chunks is shown too: it is the model the task describes, and it tops out well below the gate)")
    lag_info = {r.name: find_lag(r, timelines, rng, args.n_coarse, args.n_fine, say) for r in runs}
    with open(os.path.join(args.out, "step0a_lags.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "chosen_lag_s", "model", "search", "lag", "agreement"])
        for r in runs:
            li = lag_info[r.name]
            for kind in ("plain", "coarse", "fine"):
                for m in "AB":
                    for lag, a in zip(li["fine_lags" if kind == "fine" else "coarse_lags"], li[kind][m]):
                        w.writerow([r.name, f"{li['lag']:.4f}", m, kind, f"{lag:.4f}", f"{a:.4f}"])
    with open(os.path.join(args.out, "step0a_lags.json"), "w") as f:
        json.dump({n: li["lag"] for n, li in lag_info.items()}, f, indent=1)
    say()
    chunk_diagnostic(runs, timelines, rng, say)

    say("== Full reproduction of every logged prediction (runs 1-7), lag fixed per run ==")
    res = {}
    for r in runs:
        for m in "AB":
            res[(r.name, m)] = C.predict_rows(r, timelines[m], m, np.arange(len(r.preds[m])), lag_info[r.name]["lag"])
        say(f"  {r.name} done ({time.time() - t_start:.0f} s)")
    say()

    say("prediction agreement over rows that have a logged prob1 and a full window (run1 starts with nan prob1 rows)")
    say("run  model  n_rows  n_valid  agree   agree_excl_swap  mean|dprob1|  median|dprob1|  corr(prob1)  frac|dprob1|>0.1")
    pooled = {m: [0, 0, 0, 0] for m in "AB"}
    per_run_agree = {}
    for r in runs:
        for m in "AB":
            P = r.preds[m]
            pred, prob = res[(r.name, m)]
            valid = ~np.isnan(P[:, 2]) & (pred >= 0)
            ok = pred[valid] == P[valid, 1].astype(int)
            near = np.zeros(len(P), bool)
            for s in (e[0] for e in timelines[m][1:]):
                near |= np.abs(P[:, 0] - s) <= 1.0
            ex = valid & ~near
            ok_ex = pred[ex] == P[ex, 1].astype(int)
            dp = np.abs(prob[valid] - P[valid, 2])
            per_run_agree[(r.name, m)] = ok.mean()
            say(f"{r.name}  {m}      {len(P):6d}  {valid.sum():6d}  {ok.mean():.4f}  {ok_ex.mean():.4f}           {dp.mean():.4f}        {np.median(dp):.4f}        "
                f"{np.corrcoef(prob[valid], P[valid, 2])[0, 1]:.4f}       {(dp > 0.1).mean():.4f}")
            pooled[m][0] += ok.sum()
            pooled[m][1] += valid.sum()
            pooled[m][2] += ok_ex.sum()
            pooled[m][3] += ex.sum()
    pooled_agree = {}
    for m in "AB":
        pooled_agree[m] = pooled[m][0] / pooled[m][1]
        say(f"ALL  {m}  pooled agreement {pooled[m][0] / pooled[m][1]:.4f} ({pooled[m][0]}/{pooled[m][1]}); excluding +-1 s around weight swaps {pooled[m][2] / pooled[m][3]:.4f}")
    say()

    say("== Agreement per checkpoint segment (between consecutive weight swaps of the model) ==")
    for m in "AB":
        for k, (t0, path) in enumerate(timelines[m]):
            t1 = timelines[m][k + 1][0] if k + 1 < len(timelines[m]) else np.inf
            n = ag = 0
            for r in runs:
                P = r.preds[m]
                pred, _ = res[(r.name, m)]
                sel = (P[:, 0] >= t0) & (P[:, 0] < t1) & ~np.isnan(P[:, 2]) & (pred >= 0)
                n += sel.sum()
                ag += (pred[sel] == P[sel, 1].astype(int)).sum()
            if n:
                say(f"  {m} {os.path.basename(path)[:-4]}: n={n:5d} agree={ag / n:.4f}" + ("   <-- below 0.90" if ag / n < 0.9 else ""))
    say()

    say("== Unity vote per trial: last 100 predictions with time <= AT vs the vote in the END line of the log ==")
    say("  replica = logged predictions.csv preds through my vote code (checks vote code + AT convention); offline = re-predicted preds")
    vote_rows = []
    agree = {m: {"replica": [0, 0], "offline": [0, 0]} for m in "AB"}
    for r in runs:
        for t in r.trials:
            if t["at"] is None:
                continue
            for m in "AB":
                P = r.preds[m]
                idx = C.last_before(P[:, 0], t["at"])
                logged_pred = P[idx, 1].astype(int)
                v_rep = C.unity_vote(logged_pred)
                off = res[(r.name, m)][0][idx]
                v_off = C.unity_vote(off) if (off >= 0).all() else None
                v_log = t["logged"][m]
                vote_rows.append([r.name, t["idx"], t["block"], t["label"], m, t["at"], len(idx), v_log, v_rep, v_off,
                                  int(np.sum(off != logged_pred)) if (off >= 0).all() else ""])
                if v_log is not None:
                    for key, v in (("replica", v_rep), ("offline", v_off)):
                        agree[m][key][1] += 1
                        agree[m][key][0] += int(v == v_log)
    with open(os.path.join(args.out, "step0a_votes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "trial", "block", "label", "model", "at", "n_preds", "vote_logged", "vote_replica", "vote_offline", "n_pred_diff"])
        w.writerows(vote_rows)
    say("run  model  trials  replica_agree  offline_agree  offline_acc_vs_label  logged_acc_vs_label")
    for r in runs:
        for m in "AB":
            rr = [row for row in vote_rows if row[0] == r.name and row[4] == m and row[7] is not None]
            say(f"{r.name}  {m}      {len(rr):3d}    {np.mean([x[8] == x[7] for x in rr]):.4f}         {np.mean([x[9] == x[7] for x in rr]):.4f}         "
                f"{np.mean([x[9] == x[3] for x in rr]):.4f}               {np.mean([x[7] == x[3] for x in rr]):.4f}")
    vote_agree = {}
    for m in "AB":
        vote_agree[m] = agree[m]["offline"][0] / agree[m]["offline"][1]
        say(f"ALL  {m}  replica {agree[m]['replica'][0]}/{agree[m]['replica'][1]} = {agree[m]['replica'][0] / agree[m]['replica'][1]:.4f}   "
            f"offline {agree[m]['offline'][0]}/{agree[m]['offline'][1]} = {vote_agree[m]:.4f}")
    say()

    verdict = all(pooled_agree[m] >= PASS_PRED and vote_agree[m] >= PASS_VOTE for m in "AB")
    say(f"== Gate: prediction agreement >= {PASS_PRED:.0%} and vote agreement >= {PASS_VOTE:.0%} for both models (pooled over runs 1-7) ==")
    for m in "AB":
        say(f"  {m}: predictions {pooled_agree[m]:.4f} {'OK' if pooled_agree[m] >= PASS_PRED else 'FAIL'}   votes {vote_agree[m]:.4f} {'OK' if vote_agree[m] >= PASS_VOTE else 'FAIL'}")
    say("  lowest per-run prediction agreement: " + ", ".join(f"{k[0]}{k[1]} {v:.3f}" for k, v in sorted(per_run_agree.items(), key=lambda kv: kv[1])[:4]))
    say("RESULT: " + ("PASS" if verdict else "FAIL"))
    say("lags used (s): " + ", ".join(f"{n} {li['lag']:+.3f}" for n, li in lag_info.items()))
    say(f"elapsed {time.time() - t_start:.0f} s")
    with open(os.path.join(args.out, "step0a_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    style()
    fig_lags(runs, lag_info, args.out)
    fig_accuracy(runs, args.out)
    return verdict


def fig_lags(runs, lag_info, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharey=True)
    for ax, r in zip(axes.flat, runs):
        li = lag_info[r.name]
        for m in "AB":
            ax.plot(li["coarse_lags"], li["plain"][m], color=MODEL_COLOR[m], lw=1, ls="--", alpha=0.55, label=f"{m}, plain lag")
            ax.plot(li["coarse_lags"], li["coarse"][m], color=MODEL_COLOR[m], lw=1.4, label=f"{m}, lag + {C.CHUNK}-sample chunks")
        ax.axvline(li["lag"], color=INK, lw=0.8, ls="--")
        ax.axhline(PASS_PRED, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{r.name}: lag {li['lag']:+.3f} s", fontsize=10.5)
        ins = ax.inset_axes([0.58, 0.08, 0.38, 0.38])
        for m in "AB":
            ins.plot(li["fine_lags"], li["fine"][m], color=MODEL_COLOR[m], lw=1)
        ins.axvline(li["lag"], color=INK, lw=0.7, ls="--")
        ins.tick_params(labelsize=6.5)
        ins.set_title("2 ms steps", fontsize=7, pad=2)
        ax.set_xlabel("lag (s)")
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("agreement with logged pred")
    axes[1, 0].set_ylabel("agreement with logged pred")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    header(fig, "Step 0a. Offline vs online predictions as a function of the EEG-file lag",
           "Per run, on a subsample of logged predictions. Dashed curves: plain single lag; solid: lag + 50-sample chunks. Vertical dashed = chosen lag, dotted = 95% gate. Inset: 2 ms steps.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.08, wspace=0.12, hspace=0.32)
    fig.savefig(os.path.join(out, "step0a_lag_curves.png"), dpi=150)
    plt.close(fig)


def fig_accuracy(runs, out, n_boot=500, seed=0):
    """Accuracy of the logged predictions by prediction time relative to START (0.25 s bins, 0..6 s); descriptive."""
    centers = np.arange(0, 6.0, 0.25) + 0.125
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    per_trial = {m: [] for m in "AB"}
    for r in runs:
        for t in r.trials:
            if t["at"] is None:
                continue
            for m in "AB":
                P = r.preds[m]
                sel = (P[:, 0] >= t["start"]) & (P[:, 0] < t["start"] + 6.0) & ~np.isnan(P[:, 2])
                b = np.floor((P[sel, 0] - t["start"]) / 0.25).astype(int)
                ok = (P[sel, 1].astype(int) == t["label"]).astype(float)
                per_trial[m].append((np.bincount(b, ok, 24)[:24], np.bincount(b, minlength=24)[:24]))
    for m in "AB":
        s = np.array([x[0] for x in per_trial[m]])
        n = np.array([x[1] for x in per_trial[m]])
        acc = s.sum(0) / n.sum(0)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(s), len(s))
            boots.append(s[i].sum(0) / n[i].sum(0))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(centers, lo, hi, color=MODEL_COLOR[m], alpha=0.18, lw=0)
        ax.plot(centers, acc, "o-", color=MODEL_COLOR[m], ms=4, lw=1.5, label=f"{m} (n = {len(s)} trials)")
    ax.axhline(0.5, color=AXIS, lw=0.9)
    ax.axvline(4.25, color=INK, lw=0.9, ls="--")
    ax.text(4.3, ax.get_ylim()[0] + 0.01, "vote (4.25 s)", fontsize=8.5, color=INK2)
    ax.set_xlabel("Prediction time relative to trial START (s)")
    ax.set_ylabel("Accuracy of logged predictions")
    ax.set_xlim(0, 6)
    ax.legend(fontsize=9)
    header(fig, "Accuracy over the trial", "Logged predictions of runs 1-7 (all 384 trials incl. run 1 calibration), 0.25 s bins; band = 95% bootstrap over trials. Descriptive.")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.12)
    fig.savefig(os.path.join(out, "step0a_accuracy_over_trial.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
