# python python/analysis/gate_study/round10_decision.py --ps <round2plus7plus10/per_session.csv> \
#        --claims <round10_sprtkb_hi_20261125/_binding_round2_orig> --out <txt>
"""
Round 10 (sprtkb, SPRT 0.75/0.95/0.05/0.2) decision rules and comparisons, as written in round10_sprtkb_hi_prereg.txt.
Unit = subject (seeds averaged per session, sessions averaged per subject). Two sets:
  all     every session (the 24 sessions bound to ws-4090-01 in round 2 ran on ws-4090-02 in round 10: cross-hardware)
  clean   only the sessions bound to ws-5090-01 (round 2, 7 and 10 on the same GPU); a subject counts if it has >= 1
D1 residual vs the clock curve > 0 (Wilcoxon, Holm over k); D2 lower bound > -1 pp vs block (95% and Holm-strict 98.3%)
with >= 30% fewer finetunes; D3 beats block (Holm). C1 same cadence (k=4 vs block, k=8 vs every:8, k=12 vs every:12);
C4 vs round 7 sprtk at the same k (Holm over k).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HI = "0.75_0.95_0.05_0.2"
KS = (4, 8, 12)
CLOCK = ["never", "every_20", "every_12", "every_8", "every_6", "always"]


def holm(ps):
    ps = np.asarray(ps, float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def boot(x, lv=95, n=10000, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [(100 - lv) / 2, 100 - (100 - lv) / 2])


def subject_table(ps):
    s = ps.groupby(["condition", "subject", "session"])[["acc", "n_ft"]].mean()
    return s.groupby(["condition", "subject"]).mean()


def residual(sub, c):
    out = {}
    for s in sub.loc[c].index:
        pts = sorted((sub.loc[(p, s), "n_ft"], sub.loc[(p, s), "acc"]) for p in CLOCK)
        x, y = np.array(pts).T
        out[s] = sub.loc[(c, s), "acc"] - np.interp(sub.loc[(c, s), "n_ft"], x, y)
    return pd.Series(out)


def report(ps, label):
    sub = subject_table(ps)
    acc = sub["acc"].unstack(0)
    nft = sub["n_ft"].unstack(0).mean()
    L = [f"== {label}: n = {len(acc)} subjects, {ps.groupby(['subject', 'session']).ngroups} sessions"]
    kb = {k: f"sprtkb_{k}_{HI}" for k in KS}
    r7 = {k: f"sprtk_{k}_{HI}" for k in KS}
    L.append(f"  {'condition':30s} {'acc':>6s} {'ft/ses':>7s}")
    for c in ["block", "every_12", "every_8", "always"] + [r7[k] for k in KS] + [kb[k] for k in KS]:
        L.append(f"  {c:30s} {100 * acc[c].mean():6.2f} {nft[c]:7.1f}")

    def paired(a, b):
        d = 100 * (acc[a] - acc[b]).dropna()
        return d, d.mean(), boot(d), (wilcoxon(d).pvalue if np.any(d != 0) else 1.0)

    # D1
    L.append("  D1 residual vs clock curve (pp [95% CI], Wilcoxon p, Holm p)")
    res = {k: 100 * residual(sub, kb[k]) for k in KS}
    p1 = [wilcoxon(res[k]).pvalue for k in KS]
    h1 = holm(p1)
    for k, p, h in zip(KS, p1, h1):
        lo, hi = boot(res[k])
        L.append(f"    k={k:<2d} {res[k].mean():+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3g} holm={h:.3g}")
    d1 = [k for k, h in zip(KS, h1) if h < 0.05 and res[k].mean() > 0]
    # D2 / D3 vs block
    L.append("  vs block (pp [95% CI] [98.3% CI], Wilcoxon p, Holm p, finetunes % of block)")
    rows = {k: paired(kb[k], "block") for k in KS}
    h3 = holm([rows[k][3] for k in KS])
    d2, d3 = [], []
    for k, h in zip(KS, h3):
        d, m, ci, p = rows[k]
        ci2 = boot(d, 98.33)
        share = nft[kb[k]] / nft["block"]
        L.append(
            f"    k={k:<2d} {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] [{ci2[0]:+.2f}, {ci2[1]:+.2f}] p={p:.3g} holm={h:.3g} "
            f"ft {100 * share:.0f}%"
        )
        if ci[0] > -1 and share <= 0.7:
            d2.append((k, "95%" + (" and 98.3%" if ci2[0] > -1 else " only")))
        if h < 0.05 and m > 0:
            d3.append(k)
    # C1, C3, C4
    L.append("  C1 same cadence / C3 vs clocks (pp [95% CI], Wilcoxon p)")
    for k, ref in ((4, "block"), (8, "every_8"), (12, "every_12"), (4, "always"), (8, "every_12"), (12, "every_8")):
        d, m, ci, p = paired(kb[k], ref)
        L.append(
            f"    sprtkb_{k:<2d} - {ref:9s} {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.3g}  better in {int((d > 0).sum())} / {len(d)}"
        )
    L.append("  C4 vs round 7 sprtk at the same k (pp [95% CI], Wilcoxon p, Holm p)")
    rows4 = {k: paired(kb[k], r7[k]) for k in KS}
    h4 = holm([rows4[k][3] for k in KS])
    for k, h in zip(KS, h4):
        d, m, ci, p = rows4[k]
        L.append(
            f"    k={k:<2d} {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.3g} holm={h:.3g}  better in {int((d > 0).sum())} / {len(d)}, worse in {int((d < 0).sum())}"
        )
    rec = bool(d3) or (bool(d1) and any(k in d1 for k, _ in d2))
    L.append(
        f"  Decision: D1 {d1 or 'not met'}; D2 {d2 or 'not met'}; D3 {d3 or 'not met'} -> {'RECOMMEND' if rec else 'not recommended'}"
    )
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True)
    ap.add_argument("--claims", required=True, help="round 10 _binding_round2_orig (host__<sub>_<ses> files)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ps = pd.read_csv(args.ps, dtype={"subject": str})
    clean = set()
    for f in os.listdir(args.claims):
        if f.startswith("host__") and open(os.path.join(args.claims, f)).read().strip() == "ws-5090-01":
            sub, ses = f[len("host__") :].split("_")
            clean.add((sub, ses))
    key = list(zip(ps.subject, ps.session))
    L = report(ps, "all sessions (half cross-hardware in round 10)")
    L += [""] + report(ps[[k in clean for k in key]], "clean: sessions bound to ws-5090-01 only")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
