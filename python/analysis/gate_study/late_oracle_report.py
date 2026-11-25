# python python/analysis/gate_study/late_oracle_report.py --sim <round 9 folder> --ref_sim <round 2 folder> --out <dir>
"""
Round 9: late-run held-out oracle (oraclelate). Its decisions look at judge set A (half of each class in runs 4-6, see
simulate_gates.late_judge_mask); every condition is scored here on the trials outside A only, so all conditions are
compared on the same trials, none of which a decision saw.

Conditions: everything in --sim (oraclelate_0, oraclelate_0.05, block, never), plus --ref_conditions from --ref_sim
(round 2, tagged @r2; half of its sessions ran on another GPU model: noise, not bias). A is taken from the judge.csv of
--sim and checked against a recomputation from the labels.
Score sets: out = all trials outside A; out_no7 = the same without run 7; r7 = run 7; r13 = runs 1-3; B = runs 4-6 outside A.
Unit = subject (seeds averaged per session, sessions per subject). Paired difference vs --ref (default block of --sim) on
`out`: mean over subjects, 95% bootstrap CI, Wilcoxon signed-rank; the same for out_no7.
Writes report.txt and conditions.csv into --out.
"""

import argparse
import collections
import csv
import glob
import os
import re

import numpy as np
from scipy.stats import wilcoxon

SETS = ("out", "out_no7", "r7", "r13", "B")


def late_judge_mask(sub, ses, labels, run_of):  # = simulate_gates.late_judge_mask
    rng = np.random.default_rng([int(sub), 1 if ses == "s1" else 2, 20261125])
    mask = np.zeros(len(labels), dtype=bool)
    for r in (4, 5, 6):
        for c in (0, 1):
            idx = np.where((run_of == r) & (labels == c))[0]
            mask[rng.choice(idx, len(idx) // 2, replace=False)] = True
    return mask


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])


def read_trials(path):
    rows = list(csv.DictReader(open(path)))
    return (
        np.array([int(r["run"]) for r in rows]),
        np.array([int(r["label"]) for r in rows]),
        np.array([int(r["correct"]) for r in rows], bool),
    )


def n_rows(path):
    return sum(1 for _ in open(path)) - 1 if os.path.exists(path) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--ref_sim", default=None)
    ap.add_argument("--ref_conditions", default="block,always,every_8,oracle,never")
    ap.add_argument("--ref", default="block")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out, "report.txt")):
        raise SystemExit(f"{args.out}/report.txt exists: use another --out")
    os.makedirs(args.out, exist_ok=True)

    # 判斷集 A：讀 judge.csv，並用標籤重算核對
    masks = {}
    for jf in sorted(glob.glob(os.path.join(args.sim, "*_s[12]", "oraclelate_*_seed*", "judge.csv"))):
        sess = os.path.basename(os.path.dirname(os.path.dirname(jf)))
        run_of, labels, _ = read_trials(os.path.join(os.path.dirname(jf), "trials.csv"))
        m = np.zeros(len(labels), bool)
        m[[int(r["t"]) for r in csv.DictReader(open(jf))]] = True
        sub, ses = sess.split("_")
        if not (m == late_judge_mask(sub, ses, labels, run_of)).all():
            raise SystemExit(f"{jf}: judge set differs from the recomputation")
        if sess in masks and not (masks[sess] == m).all():
            raise SystemExit(f"{sess}: judge sets differ between conditions or seeds")
        masks[sess] = m

    sources = [(args.sim, None, "")]
    if args.ref_sim:
        sources.append((args.ref_sim, set(args.ref_conditions.split(",")), "@r2"))
    ses_vals = collections.defaultdict(lambda: collections.defaultdict(list))  # (cond, sub, ses) -> key -> per seed
    for root, keep, tag in sources:
        for d in sorted(glob.glob(os.path.join(root, "*_s[12]", "*_seed*"))):
            sess, name = d.split(os.sep)[-2:]
            cond = re.sub(r"_seed\d+$", "", name)
            if (keep is not None and cond not in keep) or sess not in masks:
                continue
            if not os.path.exists(os.path.join(d, "trials.csv")):
                continue
            run_of, labels, correct = read_trials(os.path.join(d, "trials.csv"))
            a = masks[sess]
            if len(a) != len(labels):
                raise SystemExit(f"{d}: {len(labels)} trials, judge set has {len(a)}")
            sets = dict(
                out=~a,
                out_no7=~a & (run_of != 7),
                r7=run_of == 7,
                r13=run_of <= 3,
                B=~a & (run_of >= 4) & (run_of <= 6),
            )
            v = ses_vals[(cond + tag, *sess.split("_"))]
            for k, s in sets.items():
                v[k].append(correct[s].mean() if s.any() else np.nan)
            v["adopted"].append(n_rows(os.path.join(d, "updates.csv")))
            orc = os.path.join(d, "oracle.csv")
            v["trained"].append(n_rows(orc) if os.path.exists(orc) else n_rows(os.path.join(d, "updates.csv")))

    keys = SETS + ("adopted", "trained")
    subj = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    n_seeds = collections.defaultdict(set)
    for (c, s, _), d in ses_vals.items():
        for k in keys:
            subj[c][s][k].append(np.nanmean(d[k]))
        n_seeds[c].add(len(d["out"]))
    subjects = sorted({s for c in subj for s in subj[c]})

    def val(c, k):
        return np.array([np.nanmean(subj[c][s][k]) if s in subj[c] else np.nan for s in subjects])

    def paired(c, k):
        d = val(c, k) - val(args.ref, k)
        d = d[~np.isnan(d)]
        lo, hi = boot_ci(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else np.nan
        return 100 * d.mean(), 100 * lo, 100 * hi, p, len(d)

    rank = {"never": 0, "block": 1, "every_8": 2, "always": 3, "oracle": 4, "oraclelate_0": 5, "oraclelate_0.05": 6}
    conds = sorted(subj, key=lambda c: (c.endswith("@r2"), rank.get(c.split("@")[0], 9), c))
    lines = [
        "Round 9: late-run held-out oracle; every condition scored on the trials outside judge set A",
        f"{len(subjects)} subjects, {len(masks)} sessions with A (A = {int(np.mean([m.sum() for m in masks.values()]))} "
        f"trials on average); sources: {args.sim}" + (f"; {args.ref_sim} (@r2)" if args.ref_sim else ""),
        "out = outside A; out_no7 = outside A without run 7; r7 = run 7; r13 = runs 1-3; B = runs 4-6 outside A",
        "",
        f"{'condition':22s} {'seeds':>5s} " + " ".join(f"{k:>7s}" for k in SETS) + f" {'adopted':>8s} {'trained':>8s}"
        f"   vs {args.ref} on out (pp [95% CI] p)          on out_no7",
    ]
    table = []
    for c in conds:
        row = dict(condition=c, seeds="/".join(str(v) for v in sorted(n_seeds[c])))
        row.update({k: np.nanmean(val(c, k)) for k in keys})
        cmp = ""
        if c != args.ref:
            for k in ("out", "out_no7"):
                m, lo, hi, p, n = paired(c, k)
                row.update({f"{k}_vs_ref_pp": m, f"{k}_ci_lo": lo, f"{k}_ci_hi": hi, f"{k}_p": p, "n_subjects": n})
                cmp += f"{m:+6.2f} [{lo:+.2f}, {hi:+.2f}] p={p:<8.3g} "
        table.append(row)
        lines.append(
            f"{c:22s} {row['seeds']:>5s} "
            + " ".join(f"{row[k]:7.3f}" for k in SETS)
            + f" {row['adopted']:8.1f} {row['trained']:8.1f}   {cmp}"
        )
    lines += [
        "",
        "Decision rule (round9_oraclelate_plan.txt): worth a deployable version if oraclelate_0 or oraclelate_0.05 beats "
        f"{args.ref} on out with the lower 95% CI bound above 0 (p between 0.025 and 0.05 = weak, two conditions tested).",
    ]
    text = "\n".join(lines)
    open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8").write(text + "\n")
    with open(os.path.join(args.out, "conditions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
        w.writeheader()
        w.writerows(table)
    print(text)


if __name__ == "__main__":
    main()
