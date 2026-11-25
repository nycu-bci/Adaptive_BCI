# python python/analysis/gate_study/balance_diagnostic.py --never <round2 dir> --sim <dir> [<dir> ...] --out <txt>
"""
Exploratory (written after round 8, not pre-registered). Two checks:
1. Label structure of the recorded order: share of class-balanced (2 left / 2 right) blocks among the aligned 4-trial
   blocks of each run, and the same for aligned 2- and 3-trial blocks; P(label == previous label).
2. Gain after a finetune by the class balance of the new data: for each finetune, |#left - #right| over the trials
   added since the previous finetune (same run), and the mean gain over the next <= 4 trials before the next
   finetune (accuracy of this simulation - never seed 0 on the same trials).
"""

import argparse
import collections
import csv
import glob
import os

import numpy as np


def load(f):
    return [
        (int(x["run"]), int(x["label"]), int(x["correct"]), int(x["finetune"] or 0)) for x in csv.DictReader(open(f))
    ]


def label_structure(never):
    bal = {2: [], 3: [], 4: []}
    same = []
    for T in never.values():
        runs = collections.defaultdict(list)
        for r, lab, _, _ in T:
            runs[r].append(lab)
        for labs in runs.values():
            for k in bal:
                for i in range(0, len(labs) - k + 1, k):
                    bal[k].append(abs(sum(labs[i : i + k]) - k / 2) <= 0.5)
            same += [labs[i] == labs[i - 1] for i in range(1, len(labs))]
    return bal, same


def gain_by_balance(files, never, horizon=4):
    g, n = collections.defaultdict(list), collections.Counter()
    for f in files:
        ses = f.split(os.sep)[-3]
        T, N, last = load(f), never[ses], -1
        for t in range(len(T) - 1):
            r, _, _, ft = T[t]
            if not ft:
                continue
            new = [2 * T[i][1] - 1 for i in range(last + 1, t + 1) if T[i][0] == r]
            last = t
            ex = min(abs(sum(new)), 2)
            nxt = []
            for i in range(t + 1, min(t + 1 + horizon, len(T))):
                if T[i][0] != r:
                    break
                nxt.append(i)
                if T[i][3]:
                    break
            if nxt:
                g[ex].append(np.mean([T[i][2] - N[i][2] for i in nxt]))
                n[ex] += 1
    return g, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--never", required=True, help="round 2 simulate_gates.py out dir (never_seed0)")
    ap.add_argument("--sim", nargs="+", required=True)
    ap.add_argument(
        "--conditions", default="always,every_8,every_6,random_0.15,sprt_0.65_0.85_0.05_0.2,every_1,every_2,every_3"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    never = {
        f.split(os.sep)[-3]: load(f) for f in glob.glob(os.path.join(args.never, "*", "never_seed0", "trials.csv"))
    }
    lines = []
    bal, same = label_structure(never)
    lines.append("1. Label structure of the recorded order (48 sessions)")
    for k, chance in [(2, 0.5), (3, 0.75), (4, 0.375)]:
        lines.append(
            f"   aligned {k}-trial blocks balanced: {100 * np.mean(bal[k]):.1f}% (random labels: {100 * chance:.1f}%)"
        )
    lines.append(f"   P(label == previous label): {100 * np.mean(same):.1f}% (random labels: 50%)")
    lines.append("")
    lines.append("2. Gain over never on the next <= 4 trials after a finetune, by |#left - #right| of the new data")
    for c in args.conditions.split(","):
        files = [f for d in args.sim for f in glob.glob(os.path.join(d, "*", f"{c}_seed*", "trials.csv"))]
        if not files:
            continue
        g, n = gain_by_balance(files, never)
        tot = sum(n.values())
        parts = [
            f"|L-R|={k}{'+' if k == 2 else ''}: {100 * n[k] / tot:3.0f}% of finetunes, gain {100 * np.mean(g[k]):+5.1f} pp"
            for k in sorted(g)
        ]
        lines.append(f"   {c:24s} " + "   ".join(parts))
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
