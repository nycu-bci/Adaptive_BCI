# python python/analysis/gate_study/simulate_gates.py --start_ckpts <folds.json> --out <dir> --conditions never,block,always --seeds 0,1,2,3,4 --jobs 35:s1,35:s2
"""
Offline replay of the online finetune loop on the old Cygnus sessions (subjects 35-70), to compare finetune gates.

For one session (7 runs x 40 trials, in recorded order) and one gate:
  every trial is voted by the model in use -> correct / wrong -> the gate decides -> maybe finetune -> next trial.
Gates: never | always (every block end) | block[:thr] (EEG_Train._gates: >= 1 wrong in the block and run accuracy < thr,
       default --threshold = 0.75) | sprt[:p0,p1,alpha,beta] (main/Utils/sprt_gate.py: per trial, finetune at once on the
       lower line) | every:k (after every k-th trial of a run) | random:q (after any trial with probability q, own RNG)
       | oracle[:n] (cheating ceiling: train at every block end, keep the new weights only if they beat the old ones on the
       next n = 48 trials, else go back to the old weights)
       | oraclesplit[:n] (honest ceiling: as oracle, but the decision looks only at the odd trials (global index) of the
       next n = 96 trials; score such a run on the even trials only (analyze_gates.py acc_even), which no decision saw)
       | shadow[:k,m] (causal: train at a block end, let the new model vote in the background on the next k trials, then
       use it only if it got at least m more of them right than the model in use, else go back; one candidate at a time,
       the next one is trained at the next block end; default k = 8, m = 0)
       | sprtb[:p0,p1,alpha,beta] (diagnostic: the SPRT decides per trial, but the finetune waits for the block end).
       | sprtk:k[,p0,p1,alpha,beta] (SPRT with a minimum gap: when the SPRT says finetune but fewer than k trials came
       since the previous finetune (or the session start), the request is kept and the finetune runs at the first
       trial that makes the gap k. A kept request is not cancelled by a later "good"; the SPRT keeps scoring meanwhile
       and is reset when the new weights go in. sprtk:1 = sprt.)
--delay d: new weights are used from the d-th trial after the next one (online a finetune takes 2.5-14 s, a trial ~6.8 s:
       d = 1-2). The d trials in between are voted by the old model; the SPRT does not score them ("stale", as online),
       and no other finetune starts while one is pending.
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
import copy
import csv
import glob
import io
import json
import os
import random
import re
import socket
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
    kind, _, arg = c.partition(":")
    if kind == "sprt":
        vals = (
            [float(v) for v in arg.split(",")]
            if arg
            else [config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA]
        )
        return "sprt", vals
    if kind in ("never", "always") and not arg:
        return kind, None
    if kind == "block":
        return "block", ([float(arg)] if arg else None)
    if kind == "every" and arg:
        return "every", [int(arg)]
    if kind == "random" and arg:
        return "random", [float(arg)]
    if kind == "oracle":
        return "oracle", [int(arg) if arg else 48]
    if kind == "oraclesplit":
        return "oraclesplit", [int(arg) if arg else 96]
    if kind == "shadow":
        k, m = ([int(v) for v in arg.split(",")] + [0])[:2] if arg else (8, 0)
        return "shadow", [k, m]
    if kind == "sprtk" and arg:
        k, _, rest = arg.partition(",")
        vals = (
            [float(v) for v in rest.split(",")]
            if rest
            else [config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA]
        )
        return "sprtk", [int(k)] + vals
    if kind == "sprtb":
        vals = (
            [float(v) for v in arg.split(",")]
            if arg
            else [config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA]
        )
        return "sprtb", vals
    raise ValueError(f"unknown condition {c}")


def split_conditions(s):
    """'never,sprt:0.65,0.85,0.05,0.2,block' -> ['never', 'sprt:0.65,0.85,0.05,0.2', 'block']"""
    out = []
    for tok in s.split(","):
        if out and re.fullmatch(r"[0-9.]+", tok) and out[-1].split(":")[0] in ("sprt", "sprtb", "sprtk", "shadow") and ":" in out[-1]:
            out[-1] += "," + tok
        else:
            out.append(tok)
    return out


def cond_name(c):
    kind, vals = parse_condition(c)
    if vals is None or (kind == "oracle" and vals == [48]):
        return kind
    return kind + "_" + "_".join(f"{v:g}" for v in vals)


# ---------------- one simulation ----------------
def simulate(sess, start_ckpt, cond, seed, adaptive_runs, device, threshold, delay=0):
    kind, vals = parse_condition(cond)
    W = sess["W"]
    labels = sess["labels"]
    run_of, pos_in_run = sess["run_of"], sess["pos_in_run"]
    n_trials = len(labels)
    thr = vals[0] if kind == "block" and vals else threshold
    rng = np.random.default_rng([seed, 20261123])  # 只給 random gate；不受每次微調前的全域種子重設影響

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
    version = 0  # matrix 的列號 = 目前在用的模型
    pending = None  # (load_trial, version, correct, margin)：已訓練、還沒換上的權重
    gate = SprtGate(*vals) if kind in ("sprt", "sprtb") else (SprtGate(*vals[1:]) if kind == "sprtk" else None)
    deferred = False  # sprtb：SPRT 已判定要微調，等 block 結束；sprtk：等到距上次微調滿 k 個 trial
    last_ft = -1  # sprtk：上一次微調的 trial (session 開始 = -1)
    shadow = None  # shadow：背景中的候選模型 dict
    trial_rows, update_rows, oracle_rows = [], [], []
    run_correct = run_scored = 0
    block_wrong = 0
    last_added = 0  # 下一個還沒加進 buffer 的 trial
    sim_correct = np.zeros(n_trials, dtype=bool)

    def resolve_shadow(sh, t_now, n_total):
        """t_now = first trial after the background vote window; returns None (no candidate left)"""
        nonlocal version, cur_correct, cur_margin, pending
        w = slice(sh["t"] + 1, min(sh["t_dec"] + 1, n_total))
        n_next = w.stop - w.start
        gain = int(sh["correct"][w].sum()) - int(cur_correct[w].sum())
        accepted = n_next == vals[0] and gain >= vals[1]
        oracle_rows.append(
            [sh["t"], run_of[sh["t"]], pos_in_run[sh["t"]], n_next, f"{cur_correct[w].mean() if n_next else np.nan:.4f}",
             f"{sh['correct'][w].mean() if n_next else np.nan:.4f}", int(accepted), f"{sh['secs']:.3f}"]
        )
        if not accepted:
            model.load_state_dict(sh["parent"])
            return None
        versions_correct.append(sh["correct"].copy())
        versions_margin.append(sh["margin"].astype(np.float16))
        new_version = len(versions_correct) - 1
        tr_sizes, val_sizes = sh["sizes"]
        load_trial = t_now + delay
        update_rows.append(
            [new_version, sh["t"], run_of[sh["t"]], pos_in_run[sh["t"]], f"{sh['loss']:.4f}", f"{sh['secs']:.3f}",
             sum(tr_sizes.values()), sum(val_sizes.values()), load_trial]
        )
        if delay == 0:
            version, cur_correct, cur_margin = new_version, sh["correct"], sh["margin"]
        else:
            pending = (load_trial, new_version, sh["correct"], sh["margin"])
        return None

    for t in range(n_trials):
        if pending is not None and t >= pending[0]:  # 新權重從這個 trial 開始用
            _, version, cur_correct, cur_margin = pending
            pending = None
            if gate is not None:
                gate.reset()
        if shadow is not None and t == shadow["t_dec"] + 1:  # 候選模型在背景投完 k 個 trial：決定換不換
            shadow = resolve_shadow(shadow, t, n_trials)
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
        u = rng.random()
        decision, score = "", ""
        do_ft = False
        if adaptive and kind in ("sprt", "sprtb", "sprtk"):
            if pending is not None:
                decision, score = "stale", f"{gate.score:.3f}"
            else:
                d = gate.update(ok)
                score = f"{gate.last_score:.3f}"
                decision = d or ""
                do_ft = d == "finetune"
                if kind in ("sprtb", "sprtk"):
                    deferred = deferred or do_ft
                    do_ft = False
        if adaptive and kind == "sprtk":
            do_ft = deferred and t - last_ft >= vals[0]  # t - last_ft = 上次微調之後的新 trial 數
        if adaptive and block_end:
            if kind in ("always", "oracle", "oraclesplit"):
                do_ft = True
            elif kind == "block":
                do_ft = block_wrong >= 1 and run_correct / run_scored < thr
            elif kind == "sprtb":
                do_ft, deferred = deferred, False
            elif kind == "shadow":
                do_ft = shadow is None
        if adaptive and kind == "every":
            do_ft = (pos_in_run[t] + 1) % vals[0] == 0
        if adaptive and kind == "random":
            do_ft = u < vals[0]
        if pending is not None:  # 上一次微調還沒換上：不開始新的
            do_ft = False
        if do_ft and kind == "sprtk":
            deferred, last_ft = False, t
            if delay == 0:  # 新權重從下一個 trial 開始用 (延遲 > 0 時在換上權重時歸零)
                gate.reset()
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
            parent_state = copy.deepcopy(model.state_dict()) if kind in ("oracle", "oraclesplit", "shadow") else None
            t0 = time.perf_counter()
            seed_all(GLOBAL_SEED + SEED_STEP * seed)  # 線上 _calibrate 每次微調前都重設全域種子
            with contextlib.redirect_stdout(io.StringIO()):
                loss = trainer.train(buffer.train_set(), buffer.val_set())
            secs = time.perf_counter() - t0
            new_correct, new_margin = vote_all(model, W, labels, device)
            load_trial = t + 1 + delay
            if kind == "shadow":
                shadow = dict(t=t, t_dec=t + vals[0], correct=new_correct, margin=new_margin, parent=parent_state, loss=loss,
                              secs=secs, sizes=buffer.sizes())
                continue
            if kind in ("oracle", "oraclesplit"):
                w = np.arange(load_trial, min(load_trial + vals[0], n_trials))
                if kind == "oraclesplit":  # 只看奇數 trial 決定；偶數 trial 留給評分
                    w = w[w % 2 == 1]
                n_next = len(w)
                parent_acc = cur_correct[w].mean() if n_next > 0 else np.nan
                child_acc = new_correct[w].mean() if n_next > 0 else np.nan
                accepted = n_next > 0 and child_acc > parent_acc
                oracle_rows.append(
                    [t, run_of[t], pos_in_run[t], n_next, f"{parent_acc:.4f}", f"{child_acc:.4f}", int(accepted), f"{secs:.3f}"]
                )
                if not accepted:
                    model.load_state_dict(parent_state)
                    trial_rows[-1][-1] = 0
                    continue
            versions_correct.append(new_correct.copy())
            versions_margin.append(new_margin.astype(np.float16))
            new_version = len(versions_correct) - 1
            tr_sizes, val_sizes = buffer.sizes()
            update_rows.append(
                [
                    new_version,
                    t,
                    run_of[t],
                    pos_in_run[t],
                    f"{loss:.4f}",
                    f"{secs:.3f}",
                    sum(tr_sizes.values()),
                    sum(val_sizes.values()),
                    load_trial,
                ]
            )
            if delay == 0:
                version, cur_correct, cur_margin = new_version, new_correct, new_margin
                if gate is not None:
                    gate.reset()
            else:
                pending = (load_trial, new_version, new_correct, new_margin)
    if shadow is not None:  # session 結束時還在背景的候選：算一次訓練、沒採用
        shadow = resolve_shadow(shadow, n_trials, n_trials)
    return trial_rows, update_rows, oracle_rows, np.array(versions_correct), np.array(versions_margin)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_session(sub, ses, data_dir):
    t_load = time.perf_counter()
    runs = load_session(sub, ses, data_dir)
    W = np.concatenate([windows_of(r["x"]) for r in runs])
    sess = dict(
        W=W,
        labels=np.concatenate([r["y"] for r in runs]),
        run_of=np.concatenate([[r["run"]] * len(r["y"]) for r in runs]),
        pos_in_run=np.concatenate([np.arange(len(r["y"])) for r in runs]),
        runs=[r["run"] for r in runs],
    )
    print(
        f"{sub}/{ses}: runs {sess['runs']}, {len(sess['labels'])} trials, load {time.perf_counter() - t_load:.1f}s",
        flush=True,
    )
    return sess


def run_condition(args, starts, device, sub, ses, sess, cond, seeds):
    cond_ab = CONDITION_AB[sub][0 if ses == "s1" else 1]
    all_runs = sess["runs"]
    adaptive_runs = set(all_runs if args.adaptive == "all" or cond_ab == "B" else all_runs[:3])
    for seed in seeds[:1] if cond == "never" else seeds:
        odir = os.path.join(args.out, f"{sub}_{ses}", f"{cond_name(cond)}_seed{seed}")
        if os.path.exists(os.path.join(odir, "trials.csv")):
            print(f"  skip {odir} (exists)", flush=True)
            continue
        os.makedirs(odir, exist_ok=True)
        t0 = time.perf_counter()
        trial_rows, update_rows, oracle_rows, vc, vm = simulate(
            sess, starts[sub], cond, seed, adaptive_runs, device, args.threshold, args.delay
        )
        write_csv(
            os.path.join(odir, "updates.csv"),
            ["version", "after_trial", "run", "pos_in_run", "train_loss", "seconds", "n_train", "n_val", "load_trial"],
            update_rows,
        )
        if oracle_rows:
            write_csv(
                os.path.join(odir, "oracle.csv"),
                ["after_trial", "run", "pos_in_run", "n_next", "parent_acc", "child_acc", "accepted", "seconds"],
                oracle_rows,
            )
        np.savez_compressed(os.path.join(odir, "matrix.npz"), correct=vc, margin=vm)
        write_csv(  # trials.csv 最後寫：它存在 = 這個模擬完成
            os.path.join(odir, "trials.csv"),
            ["run", "pos_in_run", "t", "label", "version", "correct", "margin", "sprt_score", "sprt_decision", "finetune"],
            trial_rows,
        )
        acc = np.mean([r[5] for r in trial_rows])
        n_train = len(oracle_rows) if oracle_rows else len(update_rows)
        print(
            f"  {sub}_{ses} {cond_name(cond)} seed{seed}: acc {acc:.3f}, finetunes {len(update_rows)} "
            f"(trainings {n_train}), {time.perf_counter() - t0:.1f}s",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--start_ckpts", required=True, help="json: {subject: checkpoint path}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default="all", help="sub:ses,sub:ses ... or all")
    ap.add_argument("--conditions", default="never,block,always")
    ap.add_argument(
        "--queue",
        default=None,
        help="file with one 'sub:ses:condition[@seeds]' per line; several workers (shared --out) take items in order, "
        "each item claimed once by an atomic mkdir in --out/_claims, each session bound to one host. "
        "Overrides --jobs / --conditions.",
    )
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument(
        "--adaptive",
        default="all",
        choices=["all", "crossover"],
        help="all = every run may finetune; crossover = the original condition (A: runs 1-3 only)",
    )
    ap.add_argument("--threshold", type=float, default=config.FINETUNE_ACC_THRESHOLD)
    ap.add_argument("--delay", type=int, default=0, help="trials between a finetune decision and the new weights")
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
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    # 同一個 --out 裡所有結果的設定必須一致
    sim_cfg = dict(adaptive=args.adaptive, threshold=args.threshold, delay=args.delay, start_ckpts=args.start_ckpts)
    cfg_path = os.path.join(args.out, "sim_config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if old != sim_cfg:
            raise SystemExit(f"{cfg_path} has {old}, this call {sim_cfg}: use another --out")
    else:
        json.dump(sim_cfg, open(cfg_path, "w"), indent=1)
    print(
        f"device={device} buffer={config.REPLAY_BUFFER_LIMIT} epochs={config.adaption_epochs} "
        f"lr={config.adaption_learning_rate} batch={config.adaption_batch_size} use_val={config.adaption_use_val} "
        f"delay={args.delay}",
        flush=True,
    )

    if args.queue:
        # 不同型號的 GPU (4090 / 5090) 同一個 seed 的結果不完全相同：一個 session 的所有條件都在同一台主機上跑，
        # 條件之間的比較才不混進硬體差異。第一個領到該 session 的主機把它綁定 (_claims/host__<session>)。
        host = socket.gethostname()
        items = []
        for line in open(args.queue):
            if line.strip() and not line.startswith("#"):
                sub, ses, cond = line.strip().split(":", 2)
                cond, _, item_seeds = cond.partition("@")
                parse_condition(cond)
                items.append((sub, ses, cond, [int(s) for s in item_seeds.split(",")] if item_seeds else seeds))
        claims = os.path.join(args.out, "_claims")
        os.makedirs(claims, exist_ok=True)
        cache = {}
        for sub, ses, cond, item_seeds in items:
            bind = os.path.join(claims, f"host__{sub}_{ses}")
            if not os.path.exists(bind):
                tmp = f"{bind}.{host}.{os.getpid()}"
                with open(tmp, "w") as f:
                    f.write(host)
                try:
                    os.link(tmp, bind)  # 原子操作：只有一台主機成功
                except FileExistsError:
                    pass
                os.remove(tmp)
            if open(bind).read().strip() != host:
                continue
            try:
                os.mkdir(os.path.join(claims, f"{sub}_{ses}__{cond_name(cond)}"))
            except FileExistsError:
                continue
            if (sub, ses) not in cache:
                if len(cache) >= 2:
                    cache.pop(next(iter(cache)))
                cache[(sub, ses)] = build_session(sub, ses, args.data_dir)
            run_condition(args, starts, device, sub, ses, cache[(sub, ses)], cond, item_seeds)
        print("queue done", flush=True)
        return

    jobs = (
        [(s, ses) for s in SUBJECTS for ses in ("s1", "s2")]
        if args.jobs == "all"
        else [tuple(j.split(":")) for j in args.jobs.split(",")]
    )
    conditions = split_conditions(args.conditions)
    for c in conditions:
        parse_condition(c)
    for sub, ses in jobs:
        sess = build_session(sub, ses, args.data_dir)
        for cond in conditions:
            run_condition(args, starts, device, sub, ses, sess, cond, seeds)


if __name__ == "__main__":
    main()
