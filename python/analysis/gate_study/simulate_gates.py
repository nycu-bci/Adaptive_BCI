# python python/analysis/gate_study/simulate_gates.py --start_ckpts <folds.json> --out <dir> --conditions never,block,always --seeds 0,1,2,3,4 --jobs 35:s1,35:s2
"""
Offline replay of the online finetune loop on the old Cygnus sessions (subjects 35-70), to compare finetune gates.

For one session (7 runs x 40 trials, in recorded order) and one gate:
  every trial is voted by the model in use -> correct / wrong -> the gate decides -> maybe finetune -> next trial.
Gates: never | always (every block end) | block (EEG_Train._gates: >= 1 wrong in the block and run accuracy < 0.75)
       | sprt:p0,p1,alpha,beta (main/Utils/sprt_gate.py: per trial, finetune at once on the lower line).
Training is the online code itself: main/EEG/MI_train.ReplayBuffer + OnlineCalibrationTrainer with python/config.json
(buffer 384 windows / class, 4 epochs, lr, batch 8, failed trials x2). As online, the trials since the last add go into the
buffer at every finetune and at every 4-trial block end, whatever the gate; the global RNG is re-seeded before each finetune.

Data: run{N}/mi_22.pt is stored in arrange_by_label order (0,1,0,1 ...). It is put back in recorded order with the labels
of the run's log (checked against CSV re-cuts: identical). Windows: 16 per trial (500 samples, stride 100), CAR then
1-40 Hz band-pass, as prepare_calibration_dataset. Trial vote: majority of the 16 window predictions, a tie is wrong.

Outputs in --out/<subject>_<session>/<condition>_seed<k>/: trials.csv, updates.csv, matrix.npz
(every model version's correct / margin on every trial of the session, for parent-vs-child analyses).
"""

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import random
import re
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset

PY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PY_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bd_stub  # noqa: E402

bd_stub.install()
import main.Utils.config as config  # noqa: E402
from main.Utils import preprocess  # noqa: E402
from main.Utils.sprt_gate import SprtGate  # noqa: E402
from main.EEG.MI_train import ReplayBuffer, OnlineCalibrationTrainer  # noqa: E402
from main.EEG.models import SCCNet  # noqa: E402

DATA_DIR = "/mnt/project/MIEXP/DATA_Cygnus"
SUBJECTS = [
    "35",
    "37",
    "38",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "47",
    "48",
    "50",
    "51",
    "52",
    "54",
    "55",
    "57",
    "58",
    "63",
    "64",
    "65",
    "68",
    "69",
    "70",
]
# 原始 crossover 條件 (python/offline_simulation/online_simulation.py)：A = run 1-3 自適應、之後固定；B = 全部自適應
CONDITION_AB = {
    "35": ("B", "A"),
    "37": ("A", "B"),
    "38": ("B", "A"),
    "40": ("B", "A"),
    "41": ("B", "A"),
    "42": ("A", "B"),
    "43": ("A", "B"),
    "44": ("B", "A"),
    "45": ("B", "A"),
    "47": ("B", "A"),
    "48": ("B", "A"),
    "50": ("A", "B"),
    "51": ("B", "A"),
    "52": ("B", "A"),
    "54": ("B", "A"),
    "55": ("A", "B"),
    "57": ("B", "A"),
    "58": ("A", "B"),
    "63": ("A", "B"),
    "64": ("A", "B"),
    "65": ("A", "B"),
    "68": ("A", "B"),
    "69": ("A", "B"),
    "70": ("A", "B"),
}
BLOCK = 4
SEGMENT, STRIDE, N_WIN_SAMPLES = 500, 100, 2000
GLOBAL_SEED, GEN_SEED, SEED_STEP = 42, 42, 1000


# ---------------- data ----------------
def arrange_idx(y):
    """common_utils.arrange_by_label 的排列：x_stored[j] = x_recorded[idx[j]]"""
    l0, l1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(l0), len(l1))
    p0, p1 = l0[::-1][:n][::-1], l1[::-1][:n][::-1]
    inter = np.empty(2 * n, dtype=int)
    inter[0::2], inter[1::2] = p0, p1
    rem = np.concatenate((l0[::-1][n:][::-1], l1[::-1][n:][::-1]))
    return np.concatenate((rem, inter)).astype(int)


def log_labels(log_path):
    trials = {}
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        m = re.match(r"Trial\s+(\d+)\s+(START|END):\s*[\d.]+(?:\s+LABEL:\s*(\d+))?", line.strip())
        if m:
            t = trials.setdefault(int(m[1]), {})
            t[m[2].lower()] = True
            if m[3] is not None:
                t["label"] = int(m[3])
    ids = sorted(i for i in trials if {"start", "end", "label"} <= set(trials[i]))
    return np.array([trials[i]["label"] for i in ids], dtype=int)


def windows_of(x_trials):
    """(trial, ch, sample) -> (trial, 16, 1, ch, 500) float32，CAR 後帶通 (= prepare_calibration_dataset)"""
    out = []
    for tr in x_trials:
        s = tr[:, :N_WIN_SAMPLES].T  # (sample, ch)
        ws = []
        for start in range(0, s.shape[0] - SEGMENT + 1, STRIDE):
            d = s[start : start + SEGMENT]
            ws.append(preprocess.bandpass(d - np.mean(d, axis=1, keepdims=True)).T)
        out.append(np.stack(ws)[:, None])
    return np.stack(out).astype(np.float32)


def load_session(sub, ses, data_dir):
    runs = []
    for r in range(1, 9):
        rdir = os.path.join(data_dir, sub, ses, f"run{r}")
        logs = sorted(glob.glob(os.path.join(rdir, "log_*.txt")))
        pt = os.path.join(rdir, "mi_22.pt")
        if not logs or not os.path.exists(pt):
            continue
        y_log = log_labels(logs[0])
        d = torch.load(pt, weights_only=False, map_location="cpu")
        x, y = np.asarray(d["x_data"]), np.asarray(d["y_data"]).astype(int)
        if len(y_log) != len(y) or len(y) == 0:
            raise RuntimeError(f"{sub}/{ses}/run{r}: log has {len(y_log)} trials, mi_22 has {len(y)}")
        perm = arrange_idx(y_log)
        if not (y_log[perm] == y).all():
            raise RuntimeError(f"{sub}/{ses}/run{r}: mi_22 labels are not the arrange_by_label order of the log")
        x_rec = np.empty_like(x)
        x_rec[perm] = x
        runs.append(dict(run=r, x=x_rec, y=y_log))
    return runs


# ---------------- model ----------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def vote_all(model, W, labels, device, bs=4096):
    """每個 trial：16 個 window 的預測多數決 (平手算錯)；margin = 預測為正確類別的 window 比例"""
    model.eval()
    flat = torch.from_numpy(W.reshape(-1, *W.shape[2:]))
    preds = []
    for i in range(0, len(flat), bs):
        preds.append(model(flat[i : i + bs].to(device).squeeze(1)).argmax(1).cpu())
    p = torch.cat(preds).numpy().reshape(W.shape[0], W.shape[1])
    hits = (p == labels[:, None]).sum(1)
    n = W.shape[1]
    correct = hits * 2 > n
    return correct, hits / n


def to_dataset(W, labels, fails):
    X = torch.from_numpy(W.reshape(-1, *W.shape[2:]))
    y = torch.from_numpy(np.repeat(labels, W.shape[1])).long()
    f = torch.from_numpy(np.repeat(fails.astype(np.float32), W.shape[1]))
    return TensorDataset(X, Fnn.one_hot(y, 2).float(), f)


def parse_condition(c):
    if c.startswith("sprt"):
        vals = (
            [float(v) for v in c.split(":")[1].split(",")]
            if ":" in c
            else [config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA]
        )
        return "sprt", vals
    if c not in ("never", "always", "block"):
        raise ValueError(f"unknown condition {c}")
    return c, None


def cond_name(c):
    kind, vals = parse_condition(c)
    return kind if vals is None else "sprt_" + "_".join(f"{v:g}" for v in vals)


# ---------------- one simulation ----------------
def simulate(sess, start_ckpt, cond, seed, adaptive_runs, device, threshold):
    kind, vals = parse_condition(cond)
    W = sess["W"]
    labels = sess["labels"]
    run_of, pos_in_run = sess["run_of"], sess["pos_in_run"]
    n_trials = len(labels)

    ck = torch.load(start_ckpt, map_location=device, weights_only=False)
    n_chans = int(ck.get("n_chans", W.shape[3]))
    trainer = OnlineCalibrationTrainer(
        model_class=SCCNet,
        model_kwargs=dict(config.LOAD_MODEL_PARAM, channels=n_chans),
        batch_size=config.adaption_batch_size,
        num_epochs=config.adaption_epochs,
        lr=config.adaption_learning_rate,
        device=device,
        ft=True,
        tag="sim",
        seed=GEN_SEED + SEED_STEP * seed,
    )
    trainer.model.load_state_dict(ck["model_state_dict"])
    model = trainer.model
    buffer = ReplayBuffer(limit=config.REPLAY_BUFFER_LIMIT, use_val=config.adaption_use_val)

    cur_correct, cur_margin = vote_all(model, W, labels, device)
    versions_correct, versions_margin = [cur_correct.copy()], [cur_margin.astype(np.float16)]
    version = 0
    gate = SprtGate(*vals) if kind == "sprt" else None
    trial_rows, update_rows = [], []
    run_correct = run_scored = 0
    block_wrong = 0
    last_added = 0  # 下一個還沒加進 buffer 的 trial
    sim_correct = np.zeros(n_trials, dtype=bool)

    for t in range(n_trials):
        if pos_in_run[t] == 0:  # 新 run：run 計數與 SPRT 分數歸零
            run_correct = run_scored = 0
            block_wrong = 0
            if gate is not None:
                gate.reset()
        adaptive = run_of[t] in adaptive_runs
        ok = bool(cur_correct[t])
        sim_correct[t] = ok
        run_correct += int(ok)
        run_scored += 1
        block_wrong += int(not ok)
        block_end = (pos_in_run[t] + 1) % BLOCK == 0
        decision, score = "", ""
        do_ft = False
        if adaptive and kind == "sprt":
            d = gate.update(ok)
            score = f"{gate.last_score:.3f}"
            decision = d or ""
            do_ft = d == "finetune"
        if adaptive and block_end:
            if kind == "always":
                do_ft = True
            elif kind == "block":
                do_ft = block_wrong >= 1 and run_correct / run_scored < threshold
        trial_rows.append(
            [
                run_of[t],
                pos_in_run[t],
                t,
                int(labels[t]),
                version,
                int(ok),
                f"{cur_margin[t]:.4f}",
                score,
                decision,
                int(do_ft),
            ]
        )
        if block_end:
            block_wrong = 0
        if adaptive and (do_ft or block_end) and t + 1 > last_added:
            idx = np.arange(last_added, t + 1)
            buffer.add(to_dataset(W[idx], labels[idx], ~sim_correct[idx]))
            last_added = t + 1
        if do_ft:
            t0 = time.perf_counter()
            seed_all(GLOBAL_SEED + SEED_STEP * seed)  # 線上 _calibrate 每次微調前都重設全域種子
            with contextlib.redirect_stdout(io.StringIO()):
                loss = trainer.train(buffer.train_set(), buffer.val_set())
            secs = time.perf_counter() - t0
            version += 1
            cur_correct, cur_margin = vote_all(model, W, labels, device)
            versions_correct.append(cur_correct.copy())
            versions_margin.append(cur_margin.astype(np.float16))
            tr_sizes, val_sizes = buffer.sizes()
            update_rows.append(
                [
                    version,
                    t,
                    run_of[t],
                    pos_in_run[t],
                    f"{loss:.4f}",
                    f"{secs:.3f}",
                    sum(tr_sizes.values()),
                    sum(val_sizes.values()),
                ]
            )
            if gate is not None:
                gate.reset()
    return trial_rows, update_rows, np.array(versions_correct), np.array(versions_margin)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--start_ckpts", required=True, help="json: {subject: checkpoint path}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default="all", help="sub:ses,sub:ses ... or all")
    ap.add_argument("--conditions", default="never,block,always")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument(
        "--adaptive",
        default="all",
        choices=["all", "crossover"],
        help="all = every run may finetune; crossover = the original condition (A: runs 1-3 only)",
    )
    ap.add_argument("--threshold", type=float, default=config.FINETUNE_ACC_THRESHOLD)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.deterministic = True  # 同一個 seed 重跑結果相同
    torch.backends.cudnn.benchmark = False
    config.EEG_CHECKPOINT_TMP_BASE_FILE = (
        tempfile.mkdtemp(prefix="gate_sim_ckpt_") + os.sep
    )  # 本機暫存，平行執行不互相覆蓋
    config.verbose = False
    starts = json.load(open(args.start_ckpts))
    jobs = (
        [(s, ses) for s in SUBJECTS for ses in ("s1", "s2")]
        if args.jobs == "all"
        else [tuple(j.split(":")) for j in args.jobs.split(",")]
    )
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    print(
        f"device={device} buffer={config.REPLAY_BUFFER_LIMIT} epochs={config.adaption_epochs} "
        f"lr={config.adaption_learning_rate} batch={config.adaption_batch_size} use_val={config.adaption_use_val}",
        flush=True,
    )

    for sub, ses in jobs:
        t_load = time.perf_counter()
        runs = load_session(sub, ses, args.data_dir)
        W = np.concatenate([windows_of(r["x"]) for r in runs])
        sess = dict(
            W=W,
            labels=np.concatenate([r["y"] for r in runs]),
            run_of=np.concatenate([[r["run"]] * len(r["y"]) for r in runs]),
            pos_in_run=np.concatenate([np.arange(len(r["y"])) for r in runs]),
        )
        cond_ab = CONDITION_AB[sub][0 if ses == "s1" else 1]
        all_runs = [r["run"] for r in runs]
        adaptive_runs = set(all_runs if args.adaptive == "all" or cond_ab == "B" else all_runs[:3])
        print(
            f"{sub}/{ses}: runs {all_runs}, {len(sess['labels'])} trials, load {time.perf_counter() - t_load:.1f}s",
            flush=True,
        )
        for cond in conditions:
            for seed in (seeds[:1] if cond == "never" else seeds):
                odir = os.path.join(args.out, f"{sub}_{ses}", f"{cond_name(cond)}_seed{seed}")
                if os.path.exists(os.path.join(odir, "trials.csv")):
                    print(f"  skip {odir} (exists)", flush=True)
                    continue
                os.makedirs(odir, exist_ok=True)
                t0 = time.perf_counter()
                trial_rows, update_rows, vc, vm = simulate(
                    sess, starts[sub], cond, seed, adaptive_runs, device, args.threshold
                )
                write_csv(
                    os.path.join(odir, "updates.csv"),
                    ["version", "after_trial", "run", "pos_in_run", "train_loss", "seconds", "n_train", "n_val"],
                    update_rows,
                )
                np.savez_compressed(os.path.join(odir, "matrix.npz"), correct=vc, margin=vm)
                write_csv(
                    os.path.join(odir, "trials.csv"),
                    [
                        "run",
                        "pos_in_run",
                        "t",
                        "label",
                        "version",
                        "correct",
                        "margin",
                        "sprt_score",
                        "sprt_decision",
                        "finetune",
                    ],
                    trial_rows,
                )
                acc = np.mean([r[5] for r in trial_rows])
                print(
                    f"  {cond_name(cond)} seed{seed}: acc {acc:.3f}, finetunes {len(update_rows)}, "
                    f"{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
