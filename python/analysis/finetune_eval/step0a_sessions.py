# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/step0a_sessions.py --subs sub02,sub03,sub04,sub05
"""
Step 0a for the other sessions (plan_finetune_evaluation.md): lag search and full reproduction of every logged prediction, per run.

Same method as cut_compare/step0_reproduce.py (lag per run from a subsample, 50-sample chunks; each prediction re-made with the checkpoint
that was live at its time). Reports per run: lag, prediction agreement A/B, vote agreement against the END-line votes A/B, and
agreement per checkpoint segment (the base segment shows whether the session started from c?_000). Pass per session: pooled
prediction agreement >= 95% for both models.

Outputs per session in results/finetune_eval/step0a_<sub>/: step0a_summary.txt, step0a_lags.json, step0a_lag_curves.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_common as F
C = F.C
from step0_reproduce import find_lag, fig_lags, PASS_PRED  # noqa: E402
from wald_sprt_explainer import style  # noqa: E402


def timeline_c(sess, model):
    """cut_compare-style timeline (a swap_time row re-lists the same checkpoint when only the other model changed)."""
    return C.checkpoint_timeline(model, sess.runs)


def reproduce_session(sub, out_root, n_coarse, n_fine, seed):
    t_start = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    out = os.path.join(out_root, f"step0a_{sub}")
    os.makedirs(out, exist_ok=True)
    sess = F.Session(sub)
    rng = np.random.default_rng(seed)
    timelines = {m: timeline_c(sess, m) for m in "AB"}
    say(f"device: {C.DEVICE}   session: {sess.dir}")
    say("runs: " + ", ".join(f"{r.name} ({len(r.trials)} trials{', test-only' if r.test_only else ''})" for r in sess.runs))
    for m in "AB":
        say(f"  {m} chain: " + ", ".join(f"{os.path.basename(p)[:-4]}@{'start' if t == -np.inf else f'{t:.0f}'}" for t, p in timelines[m]))
        n_ft = sum(1 for b in sess.runs for bl in b.blocks if str(bl[f"ft_{m}"]) == "1")
        say(f"  {m} finetunes (ft_{m}=1 rows): {n_ft}; checkpoint files in folder: {len(__import__('glob').glob(os.path.join(sess.dir, 'checkpoints', f'c{m}_*.pth')))}")
    say()
    say("== Lag search ==")
    lag_info = {r.name: find_lag(r, timelines, rng, n_coarse, n_fine, say) for r in sess.runs}
    with open(os.path.join(out, "step0a_lags.json"), "w") as f:
        json.dump({n: li["lag"] for n, li in lag_info.items()}, f, indent=1)
    say()

    res = {}
    for r in sess.runs:
        for m in "AB":
            res[(r.name, m)] = C.predict_rows(r, timelines[m], m, np.arange(len(r.preds[m])), lag_info[r.name]["lag"])
    say(f"== Full reproduction of every logged prediction ({time.time() - t_start:.0f} s so far) ==")
    say("run   model  n_rows  n_valid  agree   agree_excl_swap(+-1 s)")
    pooled = {m: [0, 0] for m in "AB"}
    per_run = {}
    for r in sess.runs:
        for m in "AB":
            P = r.preds[m]
            pred, prob = res[(r.name, m)]
            valid = ~np.isnan(P[:, 2]) & (pred >= 0)
            ok = pred[valid] == P[valid, 1].astype(int)
            near = np.zeros(len(P), bool)
            for s in (e[0] for e in timelines[m][1:]):
                near |= np.abs(P[:, 0] - s) <= 1.0
            ex = valid & ~near
            per_run[(r.name, m)] = ok.mean()
            say(f"{r.name}  {m}      {len(P):6d}  {valid.sum():6d}  {ok.mean():.4f}  {(pred[ex] == P[ex, 1].astype(int)).mean():.4f}")
            pooled[m][0] += ok.sum()
            pooled[m][1] += valid.sum()
    pooled_agree = {m: pooled[m][0] / pooled[m][1] for m in "AB"}
    for m in "AB":
        say(f"ALL  {m}  pooled agreement {pooled_agree[m]:.4f} ({pooled[m][0]}/{pooled[m][1]})")
    say()

    say("== Agreement per checkpoint segment (distinct consecutive checkpoints of the model; the first is the base c?_000) ==")
    seg_base = {}
    for m in "AB":
        segs = []
        for k, (t0, path) in enumerate(timelines[m]):
            if segs and segs[-1][1] == path:
                continue
            segs.append([t0, path, None])
        for k in range(len(segs)):
            segs[k][2] = segs[k + 1][0] if k + 1 < len(segs) else np.inf
        for t0, path, t1 in segs:
            n = ag = 0
            for r in sess.runs:
                P = r.preds[m]
                pred, _ = res[(r.name, m)]
                sel = (P[:, 0] >= t0) & (P[:, 0] < t1) & ~np.isnan(P[:, 2]) & (pred >= 0)
                n += sel.sum()
                ag += (pred[sel] == P[sel, 1].astype(int)).sum()
            if n:
                say(f"  {m} {os.path.basename(path)[:-4]}: n={n:5d} agree={ag / n:.4f}" + ("   <-- below 0.90" if ag / n < 0.9 else ""))
                if t0 == -np.inf:
                    seg_base[m] = ag / n
    say()

    say("== Unity vote per trial: last 100 predictions with time <= AT vs the vote in the END line ==")
    say("run   model  trials  replica_agree  offline_agree  offline_acc_vs_label  logged_acc_vs_label")
    agree = {m: [0, 0, 0] for m in "AB"}
    for r in sess.runs:
        for m in "AB":
            rows = []
            for t in r.trials:
                P = r.preds[m]
                idx = C.last_before(P[:, 0], t["at"])
                off = res[(r.name, m)][0][idx]
                rows.append((C.unity_vote(P[idx, 1].astype(int)), C.unity_vote(off) if (off >= 0).all() else None, t["logged"][m], t["label"]))
            rows = [x for x in rows if x[2] is not None]
            say(f"{r.name}  {m}      {len(rows):3d}    {np.mean([x[0] == x[2] for x in rows]):.4f}         {np.mean([x[1] == x[2] for x in rows]):.4f}         "
                f"{np.mean([x[1] == x[3] for x in rows]):.4f}               {np.mean([x[2] == x[3] for x in rows]):.4f}")
            agree[m][0] += sum(x[1] == x[2] for x in rows)
            agree[m][1] += sum(x[0] == x[2] for x in rows)
            agree[m][2] += len(rows)
    vote_agree = {m: agree[m][0] / agree[m][2] for m in "AB"}
    for m in "AB":
        say(f"ALL  {m}  replica {agree[m][1]}/{agree[m][2]}   offline {agree[m][0]}/{agree[m][2]} = {vote_agree[m]:.4f}")
    say()

    verdict = all(pooled_agree[m] >= PASS_PRED for m in "AB")
    say(f"== Gate (session {sub}): pooled prediction agreement >= {PASS_PRED:.0%} for both models ==")
    for m in "AB":
        say(f"  {m}: predictions {pooled_agree[m]:.4f} {'OK' if pooled_agree[m] >= PASS_PRED else 'FAIL'}   votes {vote_agree[m]:.4f}   base-checkpoint segment {seg_base.get(m, float('nan')):.4f}")
    say("  lowest per-run prediction agreement: " + ", ".join(f"{k[0]}{k[1]} {v:.3f}" for k, v in sorted(per_run.items(), key=lambda kv: kv[1])[:4]))
    say("RESULT: " + ("PASS" if verdict else "FAIL"))
    say("lags used (s): " + ", ".join(f"{n} {li['lag']:+.3f}" for n, li in lag_info.items()))
    say(f"elapsed {time.time() - t_start:.0f} s")
    with open(os.path.join(out, "step0a_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    style()
    fig_lags(sess.runs, lag_info, out)
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub02,sub03,sub04,sub05")
    ap.add_argument("--out", default=F.OUT_DIR)
    ap.add_argument("--n_coarse", type=int, default=250)
    ap.add_argument("--n_fine", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ok = {sub: reproduce_session(sub, args.out, args.n_coarse, args.n_fine, args.seed) for sub in args.subs.split(",")}
    print({k: ("PASS" if v else "FAIL") for k, v in ok.items()})
    return all(ok.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
