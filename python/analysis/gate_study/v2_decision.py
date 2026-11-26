# python python/analysis/gate_study/v2_decision.py --replica <v2 per_session.csv ...> --ref16 <round per_session.csv ...>
#        --seeds 0,1,2 --out <txt>
"""
Vote check V2, decision rules P1 / P2 of v2_vote_prereg.txt (written before the run).
P1  under the replica of the online vote: always (= every:4) - X for each SPRT variant X; mean over subjects, 95%
    bootstrap CI, Wilcoxon signed-rank, Holm over the variants. Holds if every X has mean > 0 and Holm p < 0.05.
P2  D_X = (always - X | replica) - (always - X | 16-window majority, rounds 2 / 4 / 7 / 10), same seeds.
Unit = subject (n = 24): seeds averaged per session, sessions per subject.
"""

import argparse
import collections
import csv

import numpy as np
from scipy.stats import wilcoxon

VARIANTS = [
    "sprtkb_4_0.75_0.95_0.05_0.2",
    "sprtkb_8_0.75_0.95_0.05_0.2",
    "sprtkb_12_0.75_0.95_0.05_0.2",
    "sprtk_4_0.75_0.95_0.05_0.2",
    "sprtk_8_0.75_0.95_0.05_0.2",
    "sprtk_12_0.75_0.95_0.05_0.2",
    "sprt_0.65_0.85_0.05_0.2",
    "sprt_0.55_0.95_0.05_0.2",
    "sprt_0.75_0.95_0.05_0.2",
    "sprtb_0.65_0.85_0.05_0.2",
]


def subject_table(paths, seeds, key="acc"):
    """{condition: {subject: value}}: seeds averaged per session, sessions per subject"""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["condition"] != "never" and int(r["seed"]) not in seeds:
                continue
            per[r["condition"]][(r["subject"], r["session"])].append(float(r[key]))
    out = {}
    for c, ses in per.items():
        sub = collections.defaultdict(list)
        for (s, _), v in ses.items():
            sub[s].append(np.mean(v))
        out[c] = {s: float(np.mean(v)) for s, v in sub.items()}
    return out


def boot(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(d, (n, len(d))).mean(1), [2.5, 97.5])


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica", nargs="+", required=True)
    ap.add_argument("--ref16", nargs="+", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = {int(s) for s in a.seeds.split(",")}
    R, Q = subject_table(a.replica, seeds), subject_table(a.ref16, seeds)
    RF, QF = subject_table(a.replica, seeds, "n_ft"), subject_table(a.ref16, seeds, "n_ft")
    L = [
        f"V2 decision (v2_vote_prereg.txt), seeds {sorted(seeds)}; unit = subject",
        f"replica files: {' '.join(a.replica)}",
        f"16-window files: {' '.join(a.ref16)}",
        "",
    ]
    subs = sorted(R["always"])
    L.append(f"subjects with always under the replica: {len(subs)}")
    L.append("\ncondition                      replica acc  ft/ses | 16-window acc  ft/ses | replica - 16 (pp)")
    for c in ["never", "always"] + VARIANTS:
        if c not in R:
            L.append(f"  {c:28s} missing under the replica")
            continue
        s = [x for x in subs if x in R[c] and c in Q and x in Q[c]]
        r = np.mean([R[c][x] for x in subs if x in R[c]])
        q = np.mean([Q[c][x] for x in s]) if s else np.nan
        rf = np.mean([RF[c][x] for x in subs if x in RF[c]])
        qf = np.mean([QF[c][x] for x in s]) if s else np.nan
        L.append(f"  {c:28s} {r*100:9.2f} {rf:7.1f} | {q*100:12.2f} {qf:7.1f} | {(r-q)*100:+.2f} (n {len(s)})")
    L.append(
        "\nP1: always - X under the replica (pp [95% CI], Wilcoxon p, Holm p over the variants, subjects where always wins)"
    )
    rows = []
    for X in VARIANTS:
        s = [x for x in subs if x in R.get(X, {})]
        d = np.array([R["always"][x] - R[X][x] for x in s])
        p = wilcoxon(d).pvalue if len(d) and np.any(d != 0) else 1.0
        rows.append((X, d, p))
    adj = holm([r[2] for r in rows]) if rows else []
    verdicts = []
    for (X, d, p), pa in zip(rows, adj):
        lo, hi = boot(d)
        v = "holds" if d.mean() > 0 and pa < 0.05 else ("not shown yet" if d.mean() > 0 else "FAILS")
        verdicts.append(v)
        L.append(
            f"  always - {X:28s} {d.mean()*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]  p={p:.3g}  Holm p={pa:.3g}  "
            f"wins {int((d > 0).sum())}/{len(d)}  -> {v}"
        )
    claim = (
        "HOLDS"
        if verdicts and all(v == "holds" for v in verdicts)
        else ("FAILS" if "FAILS" in verdicts else "NOT SHOWN YET (some variants not significant)")
    )
    L.append(f"  => P1, 'every:4 beats every tested SPRT variant' under the online vote: {claim}")
    L.append("\nP2: D_X = (always - X | replica) - (always - X | 16-window), pp [95% CI] (n subjects with both)")
    for X in VARIANTS:
        s = [x for x in subs if x in R.get(X, {}) and x in Q.get(X, {}) and x in Q.get("always", {})]
        if not s:
            L.append(f"  {X:28s} no 16-window reference")
            continue
        d = np.array([(R["always"][x] - R[X][x]) - (Q["always"][x] - Q[X][x]) for x in s])
        g16 = np.mean([Q["always"][x] - Q[X][x] for x in s])
        lo, hi = boot(d)
        L.append(
            f"  {X:28s} gap 16-window {g16*100:+.2f} -> replica {(g16 + d.mean())*100:+.2f}; "
            f"D {d.mean()*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] (n {len(s)})"
        )
    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
