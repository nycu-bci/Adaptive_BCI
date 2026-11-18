# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/sprt_gate_replay.py --sessions sub01/ses1,sub03/ses1
"""
Replay the old block gate and the SPRT finetune gate on recorded sessions, per model (A, B).

Input per run{N}/: log_*.txt END lines 'LABEL: y MODEL_PREDICTION: A=a B=b' (per-trial correctness of both models, the
Unity-style replica vote) and ab_blocks.csv (block sizes, and the finetunes that really happened: ft_A / ft_B).
  block gate: per block, finetune if >= 1 wrong trial and run accuracy so far (block included) < 0.75
  SPRT gate:  per trial; finetune when the score crosses the lower line; score reset after a decision and per run
The models are taken as they were recorded: a replay can't show how they would have changed after other finetunes.
Online, the live model is scored by Unity's CUT / WRONG_CUT, not the replica vote, so the replayed block gate can
differ slightly from the logged ft_A / ft_B.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wald_sprt_explainer import load_session
from main.Utils.sprt_gate import SprtGate

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PY_DIR, "main", "real_time_data")
MODELS = ("A", "B")


def block_gate(trials, blocks, model, threshold=0.75):
    """-> trial counts (since run start) at which the block gate would finetune `model`."""
    k = MODELS.index(model)
    out, i, rc, rs = [], 0, 0, 0
    for size in blocks:
        ok = [t[k] for t in trials[i:i + size]]
        i += size
        rc, rs = rc + sum(ok), rs + len(ok)
        if len(ok) - sum(ok) >= 1 and rs and rc / rs < threshold:
            out.append(i)
    return out


def sprt_gate(trials, model, p0, p1, alpha, beta):
    """-> (finetune trial counts, good trial counts) for `model`; trial count = trials since run start, 1-based."""
    k = MODELS.index(model)
    g = SprtGate(p0, p1, alpha, beta)
    ft, good = [], []
    for n, t in enumerate(trials, 1):
        d = g.update(t[k])
        if d == "finetune":
            ft.append(n)
        elif d == "good":
            good.append(n)
    return ft, good


def logged_finetunes(run_dir, model):
    with open(os.path.join(run_dir, "ab_blocks.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out, n = [], 0
    for r in rows:
        n += int(r["n_trials"])
        if r.get(f"ft_{model}") == "1":
            out.append(n)
    return out


def config_sprt():
    """SPRT parameters of python/config.json (online_adaptation), so the replay defaults follow the live gate."""
    with open(os.path.join(PY_DIR, "config.json"), encoding="utf-8") as f:
        adapt = json.load(f).get("online_adaptation", {})
    return {k: float(adapt.get(f"sprt_{k}", d)) for k, d in (("p0", 0.65), ("p1", 0.85), ("alpha", 0.05), ("beta", 0.2))}


def main():
    cfg = config_sprt()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="", help="comma list like sub01/ses1; default: every session with ab_blocks.csv")
    ap.add_argument("--p0", type=float, default=cfg["p0"])
    ap.add_argument("--p1", type=float, default=cfg["p1"])
    ap.add_argument("--alpha", type=float, default=cfg["alpha"])
    ap.add_argument("--beta", type=float, default=cfg["beta"])
    ap.add_argument("--out", default=os.path.join(PY_DIR, "analysis", "results", "sprt_gate_replay"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        found = glob.glob(os.path.join(DATA, "sub*", "ses*", "run*", "ab_blocks.csv"))
        sessions = sorted({os.path.relpath(os.path.dirname(os.path.dirname(p)), DATA).replace(os.sep, "/") for p in found})

    g = SprtGate(args.p0, args.p1, args.alpha, args.beta)
    lines = [f"SPRT p0={args.p0:g} p1={args.p1:g} alpha={args.alpha:g} beta={args.beta:g}: "
             f"win {g.win:+.3f} loss {g.loss:+.3f} upper {g.upper:.3f} lower {g.lower:.3f}", ""]
    rows = []
    totals = {m: {"trials": 0, "logged": 0, "block": 0, "sprt": 0, "good": 0} for m in MODELS}
    for sess in sessions:
        sess_dir = os.path.join(DATA, *sess.split("/"))
        try:
            runs = load_session(sess_dir)
        except ValueError as e:
            lines.append(f"{sess}: skipped ({e})")
            continue
        if not runs:
            lines.append(f"{sess}: skipped (no run with both a log and ab_blocks.csv)")
            continue
        lines.append(f"{sess}")
        for run in runs:
            run_dir = os.path.join(sess_dir, run["name"])
            for m in MODELS:
                acc = sum(t[MODELS.index(m)] for t in run["trials"]) / len(run["trials"])
                logged = logged_finetunes(run_dir, m)
                blk = block_gate(run["trials"], run["blocks"], m)
                ft, good = sprt_gate(run["trials"], m, args.p0, args.p1, args.alpha, args.beta)
                rows.append([sess, run["name"], m, len(run["trials"]), f"{acc:.3f}", len(logged), len(blk), len(ft), len(good),
                             " ".join(map(str, blk)), " ".join(map(str, ft)), " ".join(map(str, good))])
                t = totals[m]
                t["trials"] += len(run["trials"]); t["logged"] += len(logged); t["block"] += len(blk)
                t["sprt"] += len(ft); t["good"] += len(good)
                lines.append(f"  {run['name']:5s} {m}  n={len(run['trials']):3d} acc={acc:.2f}  finetunes: logged {len(logged):2d}, "
                             f"block gate {len(blk):2d}, SPRT {len(ft):2d} at {ft}  (SPRT good at {good})")
    lines.append("")
    lines.append("totals (all sessions): model, trials, logged finetunes, block-gate finetunes, SPRT finetunes, SPRT good")
    for m in MODELS:
        t = totals[m]
        lines.append(f"  {m}: {t['trials']} trials, logged {t['logged']}, block gate {t['block']}, SPRT {t['sprt']}, SPRT good {t['good']}")

    with open(os.path.join(args.out, "replay.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "run", "model", "n_trials", "acc", "ft_logged", "ft_block_gate", "ft_sprt", "sprt_good",
                    "block_gate_at", "sprt_finetune_at", "sprt_good_at"])
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
