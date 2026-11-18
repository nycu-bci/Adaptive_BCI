# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/cut_compare/step1_cut_compare.py --time_one
# full run: C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/cut_compare/step1_cut_compare.py   (resumable; then step1_analyze.py)
"""
Step 1: does training on the 4 s ending at the vote (cut 2) beat the first 4 s from START (cut 1)?

Leave-one-run-out over runs 1-7. For each held-out run, model (A: cA_000 base, 22 ch; B: cB_000 base, 32 ch), cut and seed, the
base checkpoint is finetuned with OnlineCalibrationTrainer (config batch size / lr / epochs, val_set=None) on all trials of the
other six runs (16 windows of 500 samples, stride 100, prepare_calibration_dataset; failed trials weight 2.0), then tested on the
held-out run with Unity's vote: the offline predictions of the windows ending at the logged prediction times of that model's rows
in the last 100 before AT, per-run lag from step 0a. Secondary: window accuracy on the held-out trial's own 16 windows.

Cut 1 = [START, START + 4 s) in file coordinates (= data_*.pt x_data). Cut 2 = the 2000 samples before the vote time AT, with the
step-0a lag applied (--cut2_align wall, default: no EEG that the predictor could not have had at AT) or without it (file).
Cut 3 = the 2000 samples recorded from START on (lag applied): the cut the code intended, and what it gives after the H4 fix.

Outputs in --out: per_trial.csv (one row per trial, model, cut, seed; appended as it goes), per_trial_base.csv, step1_meta.json
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from main.EEG.EEG_Train import prepare_calibration_dataset
from main.EEG.MI_train import OnlineCalibrationTrainer
from torch.utils.data import TensorDataset

FAIL_WEIGHT = 2.0
SEEDS_DEFAULT = 5
CUTS = (1, 2, 3)  # 1 online as is (file clock, = data_*.pt), 2 the 4 s before the vote, 3 intended START..START+4 s
COLUMNS = ["run", "trial", "label", "model", "cut", "seed", "vote_ok", "margin", "window_acc", "trial_seq", "vote"]


def trial_cuts(run, cut, lags, align):
    """(n_trials, 32, 2000) raw EEG of every trial of the run for the chosen cut."""
    if cut == 1:
        segs = [run.cut1(t) for t in run.trials]
    elif cut == 3:
        segs = [run.cut3(t, lags[run.name]) for t in run.trials]
    else:
        lag = lags[run.name] if align == "wall" else 0.0
        segs = [run.cut2(t, lag) for t in run.trials]
    assert all(len(s) == C.SEG for s in segs), f"{run.name} cut {cut}: short segment"
    return np.transpose(np.stack(segs), (0, 2, 1))


def window_dataset(run, cut, model, lags, align, positions):
    """TensorDataset(X (N,1,C,500), onehot, failure) from prepare_calibration_dataset for one run."""
    x = trial_cuts(run, cut, lags, align)[:, positions, :]
    y = np.array([t["label"] for t in run.trials])
    f = np.array([t["failed"] for t in run.trials])
    return prepare_calibration_dataset([x], [y], [f])


def test_windows(run, model, timeline_unused, lag, positions):
    """Preprocessed 1 s windows (n_trials, 100, C, 500) at the logged prediction times of `model` before each trial's AT."""
    P = run.preds[model]
    out = []
    for t in run.trials:
        rows = C.last_before(P[:, 0], t["at"])
        ends = run.window_ends(P[rows, 0], lag)
        assert (ends >= C.config.BUFFER_SIZE).all() and (ends <= len(run.ts)).all()
        out.append(C.preprocess_windows(C.gather_windows(run.eeg, ends), positions))
    counts = [len(o) for o in out]
    return torch.from_numpy(np.concatenate(out)), counts


@torch.no_grad()
def predict_tensor(model, x, chunk=4096):
    model.eval()
    return np.concatenate([torch.argmax(model(x[i:i + chunk].to(C.DEVICE).float()), 1).cpu().numpy() for i in range(0, len(x), chunk)])


def evaluate(model, tw, counts, own_ds, run):
    """-> per-trial arrays: vote, margin, window_acc."""
    pred = predict_tensor(model, tw)
    own = predict_tensor(model, own_ds.tensors[0].squeeze(1)).reshape(len(run.trials), -1)
    own_y = own_ds.tensors[1].argmax(1).numpy().reshape(len(run.trials), -1)
    votes, margins, wacc, k = [], [], [], 0
    for i, t in enumerate(run.trials):
        p = pred[k:k + counts[i]]
        k += counts[i]
        votes.append(C.unity_vote(p))
        margins.append(C.vote_margin(p, t["label"]))
        wacc.append(float((own[i] == own_y[i]).mean()))
    return np.array(votes), np.array(margins), np.array(wacc)


def train_model(model_name, n_chans, train_set, seed):
    """Finetune the base checkpoint exactly like _ModelSlot.finetune does (tag cutcmp, own temp dir for the per-epoch files)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = OnlineCalibrationTrainer(
        model_class=C.config.USE_MODEL, model_kwargs=dict(C.config.LOAD_MODEL_PARAM, channels=n_chans),
        batch_size=C.config.adaption_batch_size, num_epochs=C.config.adaption_epochs, lr=C.config.adaption_learning_rate,
        ft=True, tag="cutcmp", seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.load_checkpoint(C.BASE_CKPT[model_name])
        trainer.train(train_set, None)
    return trainer.model


def concat_sets(sets):
    X = torch.cat([s.tensors[0] for s in sets])
    y = torch.cat([s.tensors[1] for s in sets])
    w = 1.0 + (FAIL_WEIGHT - 1.0) * torch.cat([s.tensors[2] for s in sets])
    return TensorDataset(X, y, w)


def read_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done[(r["run"], r["model"], int(r["cut"]), int(r["seed"]))] = done.get((r["run"], r["model"], int(r["cut"]), int(r["seed"])), 0) + 1
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=C.SESSION_DIR)
    ap.add_argument("--out", default=C.RESULTS_DIR)
    ap.add_argument("--lags", default=os.path.join(C.RESULTS_DIR, "step0a_lags.json"))
    ap.add_argument("--cut2_align", choices=["wall", "file"], default="wall")
    ap.add_argument("--seeds", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--time_one", action="store_true", help="time one (fold, cut, model, seed) and estimate the full run, then stop")
    ap.add_argument("--force", action="store_true", help="run although step 0a did not PASS")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="cutcmp_ckpt_")
    C.config.EEG_CHECKPOINT_TMP_BASE_FILE = tmp + os.sep  # the trainer writes per-epoch files there, not under python/main

    summary0 = os.path.join(C.RESULTS_DIR, "step0a_summary.txt")
    if not args.force:
        if not (os.path.exists(summary0) and "RESULT: PASS" in open(summary0, encoding="utf-8").read()):
            sys.exit("step 0a has not passed (no RESULT: PASS in step0a_summary.txt); not running the cut comparison")
    lags = json.load(open(args.lags))
    t0 = time.time()
    runs = [C.Run(r, args.session) for r in C.RUNS]
    models = {}
    for m in "AB":
        ck = torch.load(C.BASE_CKPT[m], map_location="cpu", weights_only=False)
        n = int(ck.get("n_chans", C.config.N_CHANNELS))
        models[m] = (n, C.config.channel_positions(n))
    print(f"loaded runs in {time.time() - t0:.0f} s; device {C.DEVICE}; cut2 alignment: {args.cut2_align}; lags {lags}", flush=True)

    t0 = time.time()
    ds = {(r.name, cut, m): window_dataset(r, cut, m, lags, args.cut2_align, models[m][1]) for r in runs for cut in CUTS for m in "AB"}
    print(f"built training windows in {time.time() - t0:.0f} s", flush=True)

    meta = {"cut2_align": args.cut2_align, "lags": lags, "seeds": args.seeds, "fail_weight": FAIL_WEIGHT, "epochs": C.config.adaption_epochs,
            "batch_size": C.config.adaption_batch_size, "lr": C.config.adaption_learning_rate, "device": str(C.DEVICE)}
    csv_path = os.path.join(args.out, "per_trial.csv")
    base_path = os.path.join(args.out, "per_trial_base.csv")
    done = read_done(csv_path)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

    def write(path, rows):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    def rows_for(run, m, cut, seed, votes, margins, wacc):
        return [[run.name, t["idx"], t["label"], m, cut, seed, int(votes[i] == t["label"]), f"{margins[i]:.5f}", f"{wacc[i]:.5f}", i, votes[i]]
                for i, t in enumerate(run.trials)]

    test_cache = {}

    def get_test(run, m):
        if (run.name, m) not in test_cache:
            test_cache.clear()
            test_cache[(run.name, m)] = test_windows(run, m, None, lags[run.name], models[m][1])
        return test_cache[(run.name, m)]

    if args.time_one:
        run = next(r for r in runs if r.name == "run4")
        m, cut, seed = "A", 1, 0
        t1 = time.time()
        train_set = concat_sets([ds[(r.name, cut, m)] for r in runs if r is not run])
        t_cat = time.time() - t1
        t1 = time.time()
        model = train_model(m, models[m][0], train_set, seed)
        if C.DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t1
        t1 = time.time()
        tw, counts = get_test(run, m)
        t_testprep = time.time() - t1
        t1 = time.time()
        votes, margins, wacc = evaluate(model, tw, counts, ds[(run.name, cut, m)], run)
        t_eval = time.time() - t1
        n_train = len(train_set)
        total = args.seeds * 2 * 2 * len(runs) * (t_train + t_eval) + len(runs) * 2 * t_testprep + 60
        msg = (f"timing run4 held out, A, cut1, seed0: train windows {n_train}, concat {t_cat:.1f} s, train {t_train:.1f} s, test-window prep {t_testprep:.1f} s, "
               f"eval {t_eval:.1f} s; vote acc {np.mean(votes == np.array([t['label'] for t in run.trials])):.3f}\n"
               f"estimated full run ({args.seeds} seeds x 2 cuts x 2 models x 7 folds = {args.seeds * 28} trainings): {total / 60:.0f} min "
               f"(B has 32 channels instead of 22, so it can be a little slower; run1 as test fold has 2x the test windows)")
        print(msg)
        with open(os.path.join(args.out, "step1_timing.txt"), "w") as f:
            f.write(msg + "\n")
        return

    base_done = set()
    if os.path.exists(base_path):
        with open(base_path, newline="") as f:
            base_done = {(r["run"], r["model"]) for r in csv.DictReader(f)}
    else:
        write(base_path, [COLUMNS])

    n_total = args.seeds * len(CUTS) * 2 * len(runs)
    n_done = sum(1 for k in done if k[3] < args.seeds)
    t_run = time.time()
    for run in runs:
        for m in "AB":
            todo = [(seed, cut) for seed in range(args.seeds) for cut in CUTS if done.get((run.name, m, cut, seed)) != len(run.trials)]
            if not todo and (run.name, m) in base_done:
                continue
            tw, counts = get_test(run, m)
            if (run.name, m) not in base_done:
                model, _ = C.load_model(C.BASE_CKPT[m])
                for cut in CUTS:
                    votes, margins, wacc = evaluate(model, tw, counts, ds[(run.name, cut, m)], run)
                    write(base_path, rows_for(run, m, cut, -1, votes, margins, wacc))
            train_sets = {}
            for seed, cut in todo:
                if cut not in train_sets:
                    train_sets[cut] = concat_sets([ds[(r.name, cut, m)] for r in runs if r is not run])
                t1 = time.time()
                model = train_model(m, models[m][0], train_sets[cut], seed)
                votes, margins, wacc = evaluate(model, tw, counts, ds[(run.name, cut, m)], run)
                write(csv_path, rows_for(run, m, cut, seed, votes, margins, wacc))
                n_done += 1
                acc = np.mean([votes[i] == t["label"] for i, t in enumerate(run.trials)])
                print(f"[{n_done}/{n_total}] held-out {run.name} {m} cut{cut} seed {seed}: vote acc {acc:.3f} window acc {wacc.mean():.3f} "
                      f"({time.time() - t1:.0f} s, elapsed {(time.time() - t_run) / 60:.1f} min)", flush=True)
    meta["elapsed_min_this_session"] = (time.time() - t_run) / 60
    with open(os.path.join(args.out, "step1_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
