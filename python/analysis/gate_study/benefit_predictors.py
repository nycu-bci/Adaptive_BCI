# python python/analysis/gate_study/benefit_predictors.py --sim <simulate_gates.py --out dir> [--cond always]
"""
Premise check for any accuracy-based gate (block, SPRT): does what a gate can see before a finetune predict whether that
finetune helps?
Uses a condition that finetunes at every block end (always), so every block end has a measured effect:
  effect = child - parent vote accuracy on the next 48 trials (both fixed models on the same future trials, from matrix.npz)
Features known at the decision time t (online outcomes, model in use):
  acc_block (last 4 trials = the parent's own block), acc_last8, acc_last16, acc_run (run so far), sprt_llr16
  (SPRT 0.65/0.85 log-likelihood ratio of the last 16 trials), margin_block (share of windows on the right class),
  pos_in_run, t (session time ~ buffer size), and the session's start accuracy (never condition).
Reports: mean effect by feature bin, Spearman correlation pooled and within session (features and effect centred per
session x seed), AUC for "helped" (effect > 0), and the share of all the benefit that firing below a cut-off would keep.
Writes benefit_features.csv and benefit_summary.txt into --sim.
"""

import argparse
import collections
import csv
import glob
import math
import os

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

NEXT = 48
WIN, LOSS = math.log(0.85 / 0.65), math.log(0.15 / 0.35)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def auc(score, y):
    """P(score of a helped finetune > score of a not-helped one); ties count half"""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # 平手取平均名次
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--cond", default="always")
    ap.add_argument("--outdir", default=None, help="default: --sim")
    args = ap.parse_args()

    start_acc = {}
    for d in glob.glob(os.path.join(args.sim, "*_s[12]", "never_seed0")):
        tr = read_csv(os.path.join(d, "trials.csv"))
        start_acc[os.path.basename(os.path.dirname(d))] = np.mean([int(r["correct"]) for r in tr])

    rows = []
    for d in sorted(glob.glob(os.path.join(args.sim, "*_s[12]", f"{args.cond}_seed*"))):
        sess = os.path.basename(os.path.dirname(d))
        seed = int(d.rsplit("_seed", 1)[1])
        tr = read_csv(os.path.join(d, "trials.csv"))
        up = read_csv(os.path.join(d, "updates.csv"))
        m = np.load(os.path.join(d, "matrix.npz"))["correct"]
        ok = np.array([int(r["correct"]) for r in tr])
        margin = np.array([float(r["margin"]) for r in tr])
        run = np.array([int(r["run"]) for r in tr])
        pos = np.array([int(r["pos_in_run"]) for r in tr])
        for u in up:
            v, t = int(u["version"]), int(u["after_trial"])
            t_load = int(u.get("load_trial") or t + 1)
            w = slice(t_load, min(t_load + NEXT, m.shape[1]))
            if w.stop - w.start < 12:  # 太短的未來不算
                continue
            parent = int(tr[t]["version"])
            in_run = np.where((run == run[t]) & (np.arange(len(ok)) <= t))[0]
            last = lambda k: ok[max(0, t + 1 - k) : t + 1]
            l16 = last(16)
            rows.append(
                dict(
                    session=sess,
                    seed=seed,
                    t=t,
                    pos_in_run=int(pos[t]),
                    start_acc=start_acc.get(sess, np.nan),
                    acc_block=last(4).mean(),
                    acc_last8=last(8).mean(),
                    acc_last16=l16.mean(),
                    acc_run=ok[in_run].mean(),
                    sprt_llr16=l16.sum() * WIN + (len(l16) - l16.sum()) * LOSS,
                    margin_block=margin[max(0, t - 3) : t + 1].mean(),
                    parent_next=m[parent, w].mean(),
                    effect=m[v, w].mean() - m[parent, w].mean(),
                )
            )
    if not rows:
        raise SystemExit("no finetunes found")
    with open(os.path.join(args.outdir or args.sim, "benefit_features.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    eff = np.array([r["effect"] for r in rows])
    helped = eff > 0
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["session"], r["seed"])].append(i)
    feats = [
        "acc_block",
        "acc_last8",
        "acc_last16",
        "acc_run",
        "sprt_llr16",
        "margin_block",
        "pos_in_run",
        "t",
        "start_acc",
    ]
    lines = [
        f"Premise check: {args.sim}, condition {args.cond}",
        f"{len(rows)} finetunes, mean effect {100 * eff.mean():+.2f} pp, helped {helped.mean():.2f}, "
        f"hurt {(eff < 0).mean():.2f}, no change {(eff == 0).mean():.2f}",
        "",
        "Feature vs effect. rho = Spearman (pooled / within session x seed, centred). "
        "AUC = P(feature lower for a finetune that helped), 0.5 = no information.",
    ]
    for k in feats:
        x = np.array([r[k] for r in rows], float)
        ok_ = ~np.isnan(x)
        xc, ec = x.copy(), eff.copy()
        for idx in groups.values():
            idx = np.array(idx)
            xc[idx] -= np.nanmean(x[idx])
            ec[idx] -= eff[idx].mean()
        rho = spearmanr(x[ok_], eff[ok_])[0] if spearmanr else float("nan")
        rho_w = spearmanr(xc[ok_], ec[ok_])[0] if spearmanr else float("nan")
        lines.append(f"  {k:13s} rho {rho:+.3f}  within {rho_w:+.3f}  AUC(low = helps) {auc(-x[ok_], helped[ok_]):.3f}")

    lines += ["", "Mean effect by accuracy of the last 8 trials (pp; n)"]
    x8 = np.array([r["acc_last8"] for r in rows])
    for lo in np.arange(0, 1.0, 0.125):
        sel = (x8 >= lo - 1e-9) & (x8 < lo + 0.125 - 1e-9) if lo < 0.875 else x8 >= lo - 1e-9
        if sel.sum():
            lines.append(
                f"  acc8 {lo:.3f}-{min(lo + 0.125, 1):.3f}: {100 * eff[sel].mean():+6.2f} pp  helped {helped[sel].mean():.2f}  n={sel.sum()}"
            )
    lines += ["", "Mean effect by accuracy of the run so far"]
    xr = np.array([r["acc_run"] for r in rows])
    for lo, hi in [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        sel = (xr >= lo) & (xr < hi)
        if sel.sum():
            lines.append(
                f"  run acc {lo:.2f}-{min(hi, 1):.2f}: {100 * eff[sel].mean():+6.2f} pp  helped {helped[sel].mean():.2f}  n={sel.sum()}"
            )
    lines += [
        "",
        "Mean effect by SPRT evidence of the last 16 trials (0.65/0.85 log-likelihood ratio; < 0 = looks bad)",
    ]
    xs = np.array([r["sprt_llr16"] for r in rows])
    for lo, hi in [(-99, -4), (-4, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 99)]:
        sel = (xs >= lo) & (xs < hi)
        if sel.sum():
            lines.append(
                f"  llr16 [{lo:+d},{hi:+d}): {100 * eff[sel].mean():+6.2f} pp  helped {helped[sel].mean():.2f}  n={sel.sum()}"
            )
    lines += ["", "Mean effect by session start accuracy (never condition)"]
    xa = np.array([r["start_acc"] for r in rows])
    for lo, hi in [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        sel = (xa >= lo) & (xa < hi)
        if sel.sum():
            lines.append(
                f"  start acc {lo:.1f}-{min(hi, 1):.1f}: {100 * eff[sel].mean():+6.2f} pp  helped {helped[sel].mean():.2f}  n={sel.sum()}"
            )
    lines += ["", "Mean effect by position in the session (run)"]
    for rr in range(1, 8):
        sel = np.array([int(r["t"]) // 40 + 1 == rr for r in rows])
        if sel.sum():
            lines.append(
                f"  run {rr}: {100 * eff[sel].mean():+6.2f} pp  helped {helped[sel].mean():.2f}  n={sel.sum()}"
            )
    text = "\n".join(lines)
    open(os.path.join(args.outdir or args.sim, "benefit_summary.txt"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
