# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/cut_compare/step0_vote_sensitivity.py
"""
How many vote mismatches would the ~3% window-end jitter cause by chance? (context for the 98% vote gate of step 0a)

For each trial and model, the logged 100 predictions before AT are re-voted after flipping each prediction independently with
probability q (q = the model's measured prediction disagreement from step0a_summary.txt, default 0.031). The number of changed
votes over the 384 trials is simulated; the real mismatch count of step0a_votes.csv is compared with it. Also lists the logged margin
of the mismatched trials (how close to a tie they were).

Outputs in --out: step0a_vote_sensitivity.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.RESULTS_DIR)
    ap.add_argument("--q", type=float, default=0.031)
    ap.add_argument("--n_sim", type=int, default=2000)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    runs = [C.Run(r, load_eeg=False) for r in C.RUNS]
    votes = pd.read_csv(os.path.join(args.out, "step0a_votes.csv"))
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"flip probability per prediction q = {args.q}; {args.n_sim} simulations per model")
    for m in "AB":
        trial_preds = []
        for r in runs:
            P = r.preds[m]
            for t in r.trials:
                idx = C.last_before(P[:, 0], t["at"])
                trial_preds.append((r.name, t["idx"], P[idx, 1].astype(int), t["logged"][m]))
        base = [C.unity_vote(p) for _, _, p, _ in trial_preds]
        sims = np.zeros(args.n_sim, int)
        for s in range(args.n_sim):
            n = 0
            for (_, _, p, _), b in zip(trial_preds, base):
                flip = rng.random(len(p)) < args.q
                n += C.unity_vote(np.where(flip, 1 - p, p)) != b
            sims[s] = n
        real = int(((votes.model == m) & (votes.vote_offline != votes.vote_logged)).sum())
        say(f"{m}: real mismatches {real}/384 ({real / 384:.4f}); simulated under pure flip noise: mean {sims.mean():.1f}, "
            f"5-95% [{np.percentile(sims, 5):.0f}, {np.percentile(sims, 95):.0f}], P(>= real) = {(sims >= real).mean():.3f}; "
            f"the 98% gate allows at most 7")
        mm = votes[(votes.model == m) & (votes.vote_offline != votes.vote_logged)]
        for _, row in mm.iterrows():
            r = next(x for x in runs if x.name == row.run)
            P = r.preds[m]
            t = next(t for t in r.trials if t["idx"] == row.trial and abs(t["at"] - row["at"]) < 1e-3)
            p = P[C.last_before(P[:, 0], t["at"]), 1].astype(int)
            w = 1 + 4 * np.arange(len(p)) / (len(p) - 1)
            say(f"   {row.run} trial {row.trial}: logged vote {row.vote_logged}, offline {row.vote_offline}, {row.n_pred_diff} of 100 predictions differ; "
                f"weighted share of class 1 in the logged predictions {(w * p).sum() / w.sum():.3f} (tie = 0.5)")
    with open(os.path.join(args.out, "step0a_vote_sensitivity.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
