# python python/analysis/gate_study/oracle_split_report.py --per_session <a/per_session.csv> <b/per_session.csv> ... --out <dir>
"""
Honest ceiling for a finetune gate (round 5): the oracle peeks at the trials it is then scored on; oraclesplit decides on
the odd trials (global index) and is scored on the even ones, which no decision saw. Every condition is scored on the
even trials here (acc_even from analyze_gates.py), so all conditions are compared on the same trials.

Unit = subject (seeds averaged per session, sessions per subject). Paired differences vs --ref on acc_even:
mean over subjects, 95% bootstrap CI, Wilcoxon signed-rank. The oracle's own optimism shows as oracle_n - oraclesplit_n
(both saw the even trials' neighbours, only the oracle saw the even trials themselves).
Writes report.txt and conditions.csv into --out.
"""

import argparse
import collections
import csv
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

KEYS = ("acc", "acc_even", "acc_late_even", "n_ft", "n_train")


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])


def order(c):
    c, _, tag = c.partition("@")
    return _order(c) + (tag,)


def _order(c):
    if c.startswith("oraclesplit"):
        return (2, int(c.split("_")[1]))
    if c.startswith("oracle"):
        return (1, int(c.split("_")[1]) if "_" in c else 48)
    return (0, {"never": 0, "block": 2, "always": 3}.get(c, 1), c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_session", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", default=None, help="one per --per_session; a condition found in several files "
                    "is reported once per file as <condition>@<tag>")
    ap.add_argument(
        "--conditions", default="never,block,every_8,always", help="taken from the files besides all oracle*"
    )
    ap.add_argument("--ref", default="block")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keep = set(args.conditions.split(","))

    tags = args.tags or [str(i) for i in range(len(args.per_session))]
    if len(tags) != len(args.per_session):
        raise SystemExit("--tags needs one tag per --per_session file")
    tables = [list(csv.DictReader(open(p))) for p in args.per_session]
    seen = collections.Counter(c for rows in tables for c in {r["condition"] for r in rows})
    ses = collections.defaultdict(lambda: collections.defaultdict(list))
    source = {}
    for path, tag, rows in zip(args.per_session, tags, tables):
        for r in rows:
            c = r["condition"]
            if c not in keep and not c.startswith("oracle"):
                continue
            if seen[c] > 1:
                c = f"{c}@{tag}"
            source[c] = path
            for k in KEYS:
                ses[(c, r["subject"], r["session"])][k].append(
                    float(r.get(k) or r["n_ft"]) if k == "n_train" else float(r[k])
                )
    subj = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    n_seeds = collections.defaultdict(set)
    for (c, s, _), d in ses.items():
        for k in KEYS:
            subj[c][s][k].append(np.nanmean(d[k]))
        n_seeds[c].add(len(d["acc"]))
    conds = sorted(subj, key=order)
    subjects = sorted({s for c in conds for s in subj[c]})

    def val(c, k):
        return np.array([np.mean(subj[c][s][k]) if s in subj[c] else np.nan for s in subjects])

    lines = [
        "Honest finetune-gate ceiling: every condition scored on the even trials only",
        f"{len(subjects)} subjects; sources: " + "; ".join(sorted(set(source.values()))),
        "",
        f"{'condition':18s} {'seeds':>5s} {'acc all':>8s} {'acc even':>9s} {'even r5-7':>10s} "
        f"{'adopted':>8s} {'trained':>8s}   vs {args.ref} (acc even, pp [95% CI] p)",
    ]
    table = []
    for c in conds:
        acc, even, late = val(c, "acc"), val(c, "acc_even"), val(c, "acc_late_even")
        row = dict(
            condition=c,
            seeds="/".join(str(v) for v in sorted(n_seeds[c])),
            acc=np.nanmean(acc),
            acc_even=np.nanmean(even),
            acc_late_even=np.nanmean(late),
            ft_per_session=np.nanmean(val(c, "n_ft")),
            trainings_per_session=np.nanmean(val(c, "n_train")),
        )
        cmp = ""
        if c != args.ref and args.ref in subj:
            d = even - val(args.ref, "acc_even")
            d = d[~np.isnan(d)]
            lo, hi = boot_ci(d)
            p = wilcoxon(d).pvalue if wilcoxon is not None and np.any(d != 0) else float("nan")
            row.update(vs_ref_pp=100 * d.mean(), ci_lo_pp=100 * lo, ci_hi_pp=100 * hi, p=p, n_subjects=len(d))
            cmp = f"{100 * d.mean():+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] p={p:.3g}"
        table.append(row)
        lines.append(
            f"{c:18s} {row['seeds']:>5s} {row['acc']:8.3f} {row['acc_even']:9.3f} "
            f"{row['acc_late_even']:10.3f} {row['ft_per_session']:8.1f} {row['trainings_per_session']:8.1f}   {cmp}"
        )
    lines += [
        "",
        "oracle* decides on all of the next n trials (even ones included); oraclesplit_n on the odd ones of the next n.",
    ]
    text = "\n".join(lines)
    open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8").write(text + "\n")
    with open(os.path.join(args.out, "conditions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in table for k in r}, key=lambda k: k != "condition"))
        w.writeheader()
        w.writerows(table)
    print(text)


if __name__ == "__main__":
    main()
