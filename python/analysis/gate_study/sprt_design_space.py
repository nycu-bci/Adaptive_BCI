# python python/analysis/gate_study/sprt_design_space.py --out <dir>
"""
SPRT parameter space, exact (no model): what a gate does when the model's per-trial accuracy is p (i.i.d.).
For every (p0, p1, alpha, beta) in a grid:
  - steps win = ln(p1/p0), loss = ln((1-p1)/(1-p0)); lines lower = ln(beta/(1-alpha)), upper = ln((1-beta)/alpha)
  - break-even p* (expected step 0), wrongs in a row to fire from 0, corrects in a row to "good" from 0
  - expected finetunes per 40-trial run at accuracy p (score reset at run start, after a fire and after "good"),
    by dynamic programming over the score distribution
  - expected trials to the first fire from a fresh start (detection delay if bad, false-alarm delay if good), within 400 trials
  - behaviour signature: fire positions on fixed random outcome sequences. Configurations with the same signature
    make the same decisions on every sequence tested, so only one per class needs a GPU run.
The block gate (>= 1 wrong in the 4-trial block and run accuracy < thr), always (every block) and every-k are given for comparison.
Writes grid.csv, classes.csv, curves.csv into --out.
"""

import argparse
import csv
import hashlib
import itertools
import math
import os

import numpy as np

P_GRID = np.round(np.arange(0.40, 0.951, 0.05), 2)
RUN = 40
BLOCK = 4


def lines(p0, p1, a, b):
    return math.log(p1 / p0), math.log((1 - p1) / (1 - p0)), math.log(b / (1 - a)), math.log((1 - b) / a)


def sprt_fires_per_run(params, p, n=RUN):
    """expected number of fires in one run of n trials (reset at start, after fire, after good)"""
    win, loss, lo, up = lines(*params)
    dist = {0.0: 1.0}
    fires = 0.0
    for _ in range(n):
        new = {}
        reset = 0.0
        for s, m in dist.items():
            for st, q in ((win, p), (loss, 1 - p)):
                v = round(s + st, 9)
                if v <= lo:
                    fires += m * q
                    reset += m * q
                elif v >= up:
                    reset += m * q
                else:
                    new[v] = new.get(v, 0.0) + m * q
        if reset:
            new[0.0] = new.get(0.0, 0.0) + reset
        dist = new
    return fires


def sprt_first_fire(params, p, horizon=400):
    """expected trials to the first fire from score 0 (good resets keep testing), capped at horizon"""
    win, loss, lo, up = lines(*params)
    dist = {0.0: 1.0}
    exp_t, alive = 0.0, 1.0
    for t in range(1, horizon + 1):
        new = {}
        for s, m in dist.items():
            for st, q in ((win, p), (loss, 1 - p)):
                v = round(s + st, 9)
                if v <= lo:
                    exp_t += t * m * q
                    alive -= m * q
                elif v >= up:
                    new[0.0] = new.get(0.0, 0.0) + m * q
                else:
                    new[v] = new.get(v, 0.0) + m * q
        dist = new
    return exp_t + alive * horizon


def block_fires_per_run(p, thr=0.75, n=RUN):
    """block gate: at each block end, fire if >= 1 wrong in the block and run accuracy < thr"""
    dist = {0: 1.0}  # correct so far -> prob
    fires = 0.0
    for b in range(n // BLOCK):
        new = {}
        for c, m in dist.items():
            for k in range(BLOCK + 1):  # corrects in this block
                q = math.comb(BLOCK, k) * p**k * (1 - p) ** (BLOCK - k)
                cc = c + k
                if k < BLOCK and cc / ((b + 1) * BLOCK) < thr:
                    fires += m * q
                new[cc] = new.get(cc, 0.0) + m * q
        dist = new
    return fires


def signature(params, seqs):
    win, loss, lo, up = lines(*params)
    out = []
    for seq in seqs:
        s = 0.0
        for i, ok in enumerate(seq):
            s += win if ok else loss
            if s <= lo + 1e-12:
                out.append(i)
                s = 0.0
            elif s >= up - 1e-12:
                s = 0.0
        out.append(-1)
    return hashlib.md5(np.array(out, dtype=np.int16).tobytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(0)
    seqs = [rng.random(RUN) < rng.uniform(0.35, 0.97) for _ in range(3000)]

    grid = []
    for p0, p1, a, b in itertools.product(
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        [0.01, 0.05, 0.1, 0.2, 0.3],
        [0.05, 0.1, 0.2, 0.3, 0.4],
    ):
        if p1 < p0 + 0.1 - 1e-9 or a + b >= 1:
            continue
        win, loss, lo, up = lines(p0, p1, a, b)
        grid.append(
            dict(
                p0=p0,
                p1=p1,
                alpha=a,
                beta=b,
                win=round(win, 4),
                loss=round(loss, 4),
                lower=round(lo, 4),
                upper=round(up, 4),
                p_star=round(-loss / (win - loss), 4),
                wrongs_to_fire=math.ceil(-lo / -loss - 1e-9),
                corrects_per_wrong=round(-loss / win, 3),
                corrects_to_good=math.ceil(up / win - 1e-9),
                sig=signature((p0, p1, a, b), seqs),
            )
        )
    # 行為等價類：每類算一次曲線
    classes = {}
    for g in grid:
        classes.setdefault(g["sig"], []).append(g)
    class_rows, curves = [], []
    for k, (sig, members) in enumerate(sorted(classes.items(), key=lambda kv: kv[1][0]["p_star"])):
        rep = min(members, key=lambda g: (abs(g["alpha"] - 0.05) + abs(g["beta"] - 0.2), g["p0"], g["p1"]))
        params = (rep["p0"], rep["p1"], rep["alpha"], rep["beta"])
        fpr = {p: sprt_fires_per_run(params, p) for p in P_GRID}
        row = dict(
            sig=sig,
            n_members=len(members),
            rep=f"{rep['p0']:g},{rep['p1']:g},{rep['alpha']:g},{rep['beta']:g}",
            p_star_range=f"{min(g['p_star'] for g in members):.3f}-{max(g['p_star'] for g in members):.3f}",
            wrongs_to_fire=rep["wrongs_to_fire"],
            corrects_per_wrong=rep["corrects_per_wrong"],
            corrects_to_good=rep["corrects_to_good"],
            first_fire_p50=round(sprt_first_fire(params, 0.5), 2),
            first_fire_p65=round(sprt_first_fire(params, 0.65), 2),
            first_fire_p85=round(sprt_first_fire(params, 0.85), 2),
            **{f"fires_run_p{int(round(p * 100))}": round(v, 3) for p, v in fpr.items()},
        )
        class_rows.append(row)
        for g in members:
            g["class_rep"] = row["rep"]
        for p, v in fpr.items():
            curves.append(dict(gate=f"sprt {row['rep']}", p=p, fires_per_run=round(v, 4)))
    for thr in (0.65, 0.75, 0.85):
        for p in P_GRID:
            curves.append(dict(gate=f"block {thr:g}", p=p, fires_per_run=round(block_fires_per_run(p, thr), 4)))
    for k in (4, 6, 8, 12, 20):
        for p in P_GRID:
            curves.append(dict(gate=f"every {k}", p=p, fires_per_run=RUN // k))

    for name, rows in (("grid.csv", grid), ("classes.csv", class_rows), ("curves.csv", curves)):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(grid)} configurations, {len(class_rows)} behaviour classes")
    show = [
        "rep",
        "n_members",
        "p_star_range",
        "wrongs_to_fire",
        "corrects_per_wrong",
        "corrects_to_good",
        "fires_run_p50",
        "fires_run_p65",
        "fires_run_p75",
        "fires_run_p85",
        "first_fire_p85",
    ]
    print(" | ".join(show))
    for r in class_rows:
        print(" | ".join(str(r[k]) for k in show))
    print("block 0.75 fires/run:", {p: round(block_fires_per_run(p), 2) for p in (0.5, 0.65, 0.75, 0.85)})


if __name__ == "__main__":
    main()
