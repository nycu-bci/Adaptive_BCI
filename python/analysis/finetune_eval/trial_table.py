# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/trial_table.py --time_one
# full run: C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/trial_table.py   (--subs sub01,sub03,... to choose sessions)
"""
Step 2: trial x checkpoint table. For every session, model and checkpoint of the model's chain, every usable trial of the session is
scored: Unity's vote recomputed with that checkpoint on the 1 s windows ending at the model's own logged prediction times in the last
100 before the vote time AT (run's step-0a lag, 50-sample chunks), then vote_ok / margin against the label.

Windows are preprocessed once per trial and model (they do not depend on the checkpoint) and predicted on the GPU for all checkpoints.
Also the live-path check: the checkpoint live at each trial's AT vs the vote in the END line of the log.

Outputs in results/finetune_eval/: trial_checkpoint.csv.gz (all rows), live_path_check.txt
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_common as F
C = F.C

COLUMNS = ["session", "model", "checkpoint", "ckpt_index", "run", "trial_seq", "trial_idx", "label", "at", "live_model", "vote_ok", "margin",
           "in_sample", "vote", "on_chain", "logged_vote", "n_preds"]


def passed_sessions():
    """Sessions that pass the plan's step-0a criterion: pooled prediction agreement >= 95% for both models, read from the gate lines of the
    step-0a summary (sub01: the cut_compare summary, whose RESULT line also includes the vote gate that the plan accepted as missed)."""
    out = []
    for sub in F.SESSIONS:
        p = os.path.join(C.RESULTS_DIR, "step0a_summary.txt") if sub == "sub01" else os.path.join(F.OUT_DIR, f"step0a_{sub}", "step0a_summary.txt")
        if os.path.exists(p):
            gate = re.findall(r"^\s+([AB]): predictions ([\d.]+) (OK|FAIL)", open(p, encoding="utf-8").read(), flags=re.M)
            if len(gate) == 2 and all(g[2] == "OK" for g in gate):
                out.append(sub)
    return out


@torch.no_grad()
def predict_all(model, x, chunk=8192):
    return torch.cat([torch.argmax(model(x[i:i + chunk]), 1) for i in range(0, len(x), chunk)]).cpu().numpy()


def run_windows(sess, run, model, lag, pos):
    """Preprocessed windows of every trial of the run for `model` -> (tensor (N, ch, 500) on the device, per-trial counts, invalid flags)."""
    P = run.preds[model]
    xs, counts, bad = [], [], []
    for t in run.trials:
        rows = C.last_before(P[:, 0], t["at"])
        ends = run.window_ends(P[rows, 0], lag)
        ok = (ends >= C.config.BUFFER_SIZE) & (ends <= len(run.ts))
        bad.append(not ok.all())
        ends = np.where(ok, ends, C.config.BUFFER_SIZE)
        xs.append(C.preprocess_windows(C.gather_windows(run.eeg, ends), pos))
        counts.append(len(rows))
    return torch.from_numpy(np.concatenate(xs)).to(C.DEVICE), counts, bad


def score_session(sub, log, max_ckpts=None):
    sess = F.Session(sub)
    lags = sess.load_lags()
    rows, checks = [], []
    for m in "AB":
        files = sess.ckpt_files(m)[:max_ckpts]
        models = [C.load_model(p) for p in files]
        pos = models[0][1]
        assert all(list(mm[1]) == list(pos) for mm in models), f"{sub} {m}: checkpoints differ in channels"
        chain = sess.chain[m]
        for run in sess.runs:
            t0 = time.time()
            x, counts, bad = run_windows(sess, run, m, lags[run.name], pos)
            off = np.concatenate([[0], np.cumsum(counts)])
            preds = [predict_all(mm[0], x) for mm in models]
            for t_i, t in enumerate(run.trials):
                live_k = F.live_index(chain, t["at"])
                for k in range(len(files)):
                    p = preds[k][off[t_i]:off[t_i + 1]]
                    vote = None if bad[t_i] else C.unity_vote(p)
                    sw = chain[k]["time"]
                    rows.append([sub, m, F.ckpt_name(chain[k]["path"]), k, run.run, t["seq"], t["idx"], t["label"], t["at"], t["live"],
                                 np.nan if vote is None else float(vote == t["label"]),
                                 np.nan if bad[t_i] else C.vote_margin(p, t["label"]),
                                 F.in_sample(t["at"], sw), np.nan if vote is None else vote, k == live_k,
                                 np.nan if t["logged"][m] is None else t["logged"][m], len(p)])
            log(f"  {sub} {m} {run.name}: {len(run.trials)} trials x {len(files)} checkpoints, {len(x)} windows ({time.time() - t0:.0f} s)")
            del x
            torch.cuda.empty_cache()
    return sess, pd.DataFrame(rows, columns=COLUMNS)


def live_path_check(df, sess, log):
    """Checkpoint live at the trial's AT: its single-checkpoint vote vs the logged END-line vote."""
    d = df[df.on_chain & df.logged_vote.notna() & df.vote.notna()]
    out = {}
    for m in "AB":
        x = d[d.model == m]
        out[m] = (int((x.vote == x.logged_vote).sum()), len(x))
        mism = x[x.vote != x.logged_vote]
        # trials where a swap of this model lies in the 100 predictions before AT (the live path would mix checkpoints there)
        swaps = np.array([c["time"] for c in sess.chain[m][1:]])
        near = 0
        for _, r in mism.iterrows():
            P = [run for run in sess.runs if run.run == r["run"]][0].preds[m]
            idx = C.last_before(P[:, 0], r["at"])
            near += int(((swaps > P[idx[0], 0]) & (swaps <= r["at"])).any()) if len(idx) else 0
        log(f"  {sess.sub} {m}: live-path vote agrees with END line on {out[m][0]}/{out[m][1]} = {out[m][0] / out[m][1]:.4f}; "
            f"{len(mism)} mismatches, {near} of them with a swap of {m} inside the last-100-prediction window")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=None, help="default: sessions whose step-0a result is PASS")
    ap.add_argument("--out", default=F.OUT_DIR)
    ap.add_argument("--name", default="trial_checkpoint")
    ap.add_argument("--time_one", action="store_true", help="time the smallest session (sub02, first 4 checkpoints) and exit")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    t_start = time.time()
    if args.time_one:
        sess, df = score_session("sub02", log, max_ckpts=4)
        print(f"{len(df)} rows, {time.time() - t_start:.0f} s")
        return True
    subs = args.subs.split(",") if args.subs else passed_sessions()
    log(f"device {C.DEVICE}; sessions: {subs}")
    parts = []
    for sub in subs:
        sess, df = score_session(sub, log)
        parts.append(df)
        log(f"{sub}: {len(df)} rows, elapsed {time.time() - t_start:.0f} s")
        live_path_check(df, sess, log)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(args.out, args.name + ".csv.gz"), index=False, compression="gzip")
    log(f"saved {len(df)} rows; {int(df.vote_ok.isna().sum())} rows without a vote (fewer than 2 predictions or windows outside the recording)")
    with open(os.path.join(args.out, args.name.replace("trial_checkpoint", "live_path_check") + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"elapsed {time.time() - t_start:.0f} s")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
