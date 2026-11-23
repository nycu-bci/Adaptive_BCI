# python python/analysis/gate_study/e1_replay_original.py --e0 python/analysis/results/gate_study/e0_20261123
"""E1: replay the block gate and the SPRT gate on the ORIGINAL online outcomes of subjects 35-70 (trial correct = 5 CUTs),
and compare with the finetunes the original system really ran (session log.txt: calibration_start).
The models are as recorded (the replay cannot show what other finetunes would have changed)."""
import argparse, collections, csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import bd_stub  # noqa: E402
bd_stub.install()
from main.Utils.sprt_gate import SprtGate  # noqa: E402
CONDITION_AB = {"35": "BA", "37": "AB", "38": "BA", "40": "BA", "41": "BA", "42": "AB", "43": "AB", "44": "BA", "45": "BA",
                "47": "BA", "48": "BA", "50": "AB", "51": "BA", "52": "BA", "54": "BA", "55": "AB", "57": "BA", "58": "AB",
                "63": "AB", "64": "AB", "65": "AB", "68": "AB", "69": "AB", "70": "AB"}

def block_fires(c, thr=0.75):
    fires, rc = [], 0
    for k in range(4, len(c) + 1, 4):
        rc = c[:k].sum()
        fires.append(int((~c[k - 4:k]).sum() >= 1 and rc / k < thr))
    return fires

def sprt_fires(c, params):
    g, out = SprtGate(*params), []
    for ok in c:
        out.append(g.update(bool(ok)) == "finetune")
    return np.array(out)

ap = argparse.ArgumentParser(); ap.add_argument("--e0", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or a.e0.replace("e0_", "e1_")
os.makedirs(out, exist_ok=True)
trials = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(a.e0, "trials.csv"))):
    trials[(r["subject"], r["session"], int(r["run"]))].append((int(r["trial"]), 1 - int(r["fail"])))
ups = collections.Counter()
for r in csv.DictReader(open(os.path.join(a.e0, "updates.csv"))):
    if r["run"] != "" and r["done"] == "1":
        ups[(r["subject"], r["session"], int(r["run"]))] += 1
grid = {"sprt_0.65_0.85_0.05_0.2": (0.65, 0.85, 0.05, 0.2), "sprt_0.5_0.75_0.05_0.2": (0.5, 0.75, 0.05, 0.2),
        "sprt_0.6_0.8_0.05_0.2": (0.6, 0.8, 0.05, 0.2), "sprt_0.7_0.9_0.05_0.2": (0.7, 0.9, 0.05, 0.2),
        "sprt_0.65_0.85_0.05_0.1": (0.65, 0.85, 0.05, 0.1), "sprt_0.65_0.85_0.05_0.3": (0.65, 0.85, 0.05, 0.3)}
rows = []
for (s, ses, run), lst in sorted(trials.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
    c = np.array([ok for _, ok in sorted(lst)], dtype=bool)
    cond = CONDITION_AB[s][0 if ses == "s1" else 1]
    adaptive = cond == "B" or run <= 3
    row = dict(subject=s, session=ses, run=run, condition=cond, adaptive_original=int(adaptive), n=len(c), acc=round(c.mean(), 4),
               original_updates=ups[(s, ses, run)], block=sum(block_fires(c)), always=len(c) // 4)
    for k, p in grid.items():
        row[k] = int(sprt_fires(c, p).sum())
    rows.append(row)
with open(os.path.join(out, "per_run.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def summ(sel, label):
    acc = np.array([r["acc"] for r in sel])
    print(f"\n== {label}: {len(sel)} runs, run accuracy mean {acc.mean():.3f}, median {np.median(acc):.3f}, "
          f"quartiles {np.percentile(acc,25):.3f}/{np.percentile(acc,75):.3f}; share of runs below p*=0.76: {(acc<0.76).mean():.2f}")
    for k in ["original_updates", "always", "block"] + list(grid):
        v = np.array([r[k] for r in sel]); print(f"   {k:28s} finetunes per 40-trial run: {v.mean():5.2f}   total {v.sum()}")
summ(rows, "all runs")
summ([r for r in rows if r["adaptive_original"]], "runs that were adaptive in the original study")
summ([r for r in rows if not r["adaptive_original"]], "runs that were static in the original study")
# original updates vs block-rule replay, adaptive runs
ad = [r for r in rows if r["adaptive_original"]]
print("\noriginal updates vs replayed block gate (adaptive runs): exact match in", sum(r["original_updates"] == r["block"] for r in ad), "of", len(ad))
print("original updates == 10 (every block) in", sum(r["original_updates"] == 10 for r in ad), "runs; == 0 in", sum(r["original_updates"] == 0 for r in ad))
# per-subject accuracy
subj = collections.defaultdict(list)
for r in rows: subj[r["subject"]].append(r["acc"])
sa = sorted((np.mean(v), k) for k, v in subj.items())
print("\nper-subject mean run accuracy (original online):", " ".join(f"{k}:{m:.2f}" for m, k in sa))
