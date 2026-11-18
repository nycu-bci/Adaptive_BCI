# Shared helpers for the cut comparison: C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/cut_compare/step0_reproduce.py
"""
Shared code for step 0a (offline predictor reproduces the online predictions) and step 1 (0-4 s vs vote-ending cut).

Loads a run (EEG + sample times, trials from the log, predictions.csv, ab_blocks.csv), builds models from checkpoints,
predicts 1 s windows exactly like EEGPredictor.predict_loop (bandpass -> subtract channel mean -> model), and replicates
the Unity vote. Nothing here writes under python/main (temp checkpoints of the trainer are redirected by the caller).
"""
import csv
import glob
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.dirname(ANALYSIS_DIR)
for _p in (PY_DIR, ANALYSIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main.Utils.config as config  # noqa: E402
import main.Utils.preprocess as preprocess  # noqa: E402
from main.Utils.trial_scoring import HISTORY_LENGTH  # noqa: E402
from main.EEG.data_process_np import EEGDataLoader  # noqa: E402

SESSION_DIR = os.path.join(PY_DIR, "main", "real_time_data", "sub01", "ses1")
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results", "cut_compare")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "cut_compare_cache")
BASE_CKPT = {"A": os.path.join(config.EEG_CHECKPOINT_MAIN_BASE_FILE, "cA_000.pth"),
             "B": os.path.join(config.EEG_CHECKPOINT_MAIN_BASE_FILE, "cB_000.pth")}
RUNS = list(range(1, 8))
FS = config.SAMPLE_RATE
SEG = 2000  # samples per trial cut (4 s)
CHUNK = 50  # LSL delivers the Cygnus stream in 100 ms chunks: the live buffer only ever ends at a chunk boundary (see step 0a)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_RE = re.compile(r"Trial (\d+) START: ([\d.]+) LABEL: (\d)")
CUT_RE = re.compile(r"Trial (\d+) CUT: ([\d.]+)")
WRONG_RE = re.compile(r"Trial (\d+) WRONG_CUT: ([\d.]+)")
END_RE = re.compile(r"Trial (\d+) END: ([\d.]+) LABEL: (\d)(?: MODEL_PREDICTION: A=(\w+) B=(\w+) LIVE=(\w) AT: ([\d.]+))?")


# ---------------- votes ----------------
def unity_vote(preds):
    """Replica of trial_scoring.unity_vote: last 100 predictions, weights 1 -> 5, class-1 count adds round(weight)."""
    preds = list(preds)[-HISTORY_LENGTH:]
    n = len(preds)
    if n < 2:
        return None
    total, count1 = 0.0, 0
    for i, p in enumerate(preds):
        w = 1.0 + 4.0 * i / (n - 1)
        total += w
        if p == 1:
            count1 += round(w)
    return 1 if count1 >= total / 2.0 else 0


def vote_margin(preds, label):
    """Weighted share (plain weights 1 -> 5) of the last 100 predictions equal to the label; nan if fewer than 2."""
    preds = np.asarray(preds)[-HISTORY_LENGTH:]
    n = len(preds)
    if n < 2:
        return float("nan")
    w = 1.0 + 4.0 * np.arange(n) / (n - 1)
    return float((w * (preds == label)).sum() / w.sum())


# ---------------- run loading ----------------
def run_dir(run, session_dir=SESSION_DIR):
    return os.path.join(session_dir, f"run{run}")


def parse_log(path):
    """Trials in file order (START/END paired sequentially, since indices restart inside run1)."""
    trials, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = START_RE.match(line)
            if m:
                cur = {"idx": int(m.group(1)), "start": float(m.group(2)), "label": int(m.group(3)), "cuts": [], "wrong": None,
                       "end": None, "at": None, "logged": None, "live": None}
                trials.append(cur)
                continue
            if cur is None:
                continue
            m = CUT_RE.match(line)
            if m and int(m.group(1)) == cur["idx"]:
                cur["cuts"].append(float(m.group(2)))
                continue
            m = WRONG_RE.match(line)
            if m and int(m.group(1)) == cur["idx"]:
                cur["wrong"] = float(m.group(2))
                continue
            m = END_RE.match(line)
            if m and int(m.group(1)) == cur["idx"]:
                cur["end"] = float(m.group(2))
                if m.group(7):
                    cur["at"] = float(m.group(7))
                    cur["live"] = m.group(6)
                    cur["logged"] = {k: (None if v == "NA" else int(v)) for k, v in zip("AB", (m.group(4), m.group(5)))}
                cur = None
    for t in trials:
        t["failed"] = t["wrong"] is not None or len(t["cuts"]) < config.SUCCESS_CUT_COUNT
    return [t for t in trials if t["end"] is not None]


def read_blocks(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_trials"] = int(r["n_trials"])
        r["swap_time"] = float(r["swap_time"]) if r["swap_time"] else None
    return rows


def read_predictions(path):
    """-> {'A': array (n, 3) [time, pred, prob1] sorted by time, 'B': ...}"""
    df = pd.read_csv(path)
    out = {}
    for m in ("A", "B"):
        a = df[df["model"] == m][["time", "pred", "prob1"]].to_numpy(float)
        out[m] = a[np.argsort(a[:, 0], kind="stable")]
    return out


def read_eeg(path, channel_index=None):
    """Same rows and values as EEGDataLoader._read_eeg_csv (skips 11 lines, drops short lines), but fast."""
    channel_index = list(config.channel_index if channel_index is None else channel_index)
    cols = [0] + channel_index
    df = pd.read_csv(path, skiprows=11, header=None, usecols=cols, float_precision="round_trip", engine="c", on_bad_lines="skip")
    df = df.dropna()
    arr = df[cols].to_numpy(np.float64)
    return arr[:, 0].copy(), np.ascontiguousarray(arr[:, 1:])


def record_epoch(path):
    return EEGDataLoader([], [], config.channel_index)._extract_record_epoch(path)


class Run:
    """One run: EEG (n, 32) at 500 Hz with file timestamps, trials, predictions, blocks."""

    def __init__(self, run, session_dir=SESSION_DIR, cache_dir=CACHE_DIR, load_eeg=True):
        self.run = run
        d = run_dir(run, session_dir)
        csv_path = glob.glob(os.path.join(d, "eeg_record_*.csv"))[0]
        self.name = f"run{run}"
        self.trials = parse_log(glob.glob(os.path.join(d, "log_*.txt"))[0])
        self.blocks = read_blocks(os.path.join(d, "ab_blocks.csv"))
        for b in self.blocks:  # newer ab_blocks.csv store checkpoint names only (EEG_Train._log_block)
            for key in ("ckpt_A", "ckpt_B"):
                if b.get(key) and not os.path.isabs(b[key]):
                    b[key] = os.path.join(session_dir, "checkpoints", b[key])
        self.preds = read_predictions(os.path.join(d, "predictions.csv"))
        self.rec_epoch = record_epoch(csv_path)
        self.data_files = sorted(glob.glob(os.path.join(d, "data_*.pt")))
        self.ts = self.eeg = None
        if load_eeg:
            os.makedirs(cache_dir, exist_ok=True)
            key = os.path.join(cache_dir, f"{self.name}_{os.path.getsize(csv_path)}")
            if os.path.exists(key + "_eeg.npy"):
                self.ts, self.eeg = np.load(key + "_ts.npy"), np.load(key + "_eeg.npy")
            else:
                self.ts, self.eeg = read_eeg(csv_path)
                np.save(key + "_ts.npy", self.ts)
                np.save(key + "_eeg.npy", self.eeg)
        if sum(b["n_trials"] for b in self.blocks) != len(self.trials):
            raise ValueError(f"{self.name}: blocks sum to {sum(b['n_trials'] for b in self.blocks)}, log has {len(self.trials)} trials")
        i = 0
        for bi, b in enumerate(self.blocks):
            for t in self.trials[i:i + b["n_trials"]]:
                t["block"] = bi
                t["run"] = run
            i += b["n_trials"]

    def start_index(self, t_abs):
        """First sample index at/after absolute time t_abs, as data_process_np does for START."""
        return int(np.searchsorted(self.ts, t_abs - self.rec_epoch, side="left"))

    def cut1(self, trial):
        """Online cut: [START, START + 4 s) = 2000 samples from the first sample at/after START (data_process_np)."""
        i0 = self.start_index(trial["start"])
        return self.eeg[i0:i0 + SEG]

    def cut3(self, trial, lag):
        """Intended cut: the 2000 samples recorded from START on (wall clock, lag from step 0a applied)."""
        i0 = self.start_index(trial["start"] - lag)
        return self.eeg[i0:i0 + SEG]

    def cut2_indices(self, trial, lag=0.0):
        """The 2000 samples that end right before the vote time AT (strictly earlier than AT): (i0, i1)."""
        i1 = int(np.searchsorted(self.ts, trial["at"] - self.rec_epoch - lag, side="left"))
        return i1 - SEG, i1

    def cut2(self, trial, lag=0.0):
        i0, i1 = self.cut2_indices(trial, lag)
        return self.eeg[i0:i1]

    def window_ends(self, times, lag, chunk=CHUNK):
        """For prediction times (time.time() domain) -> index one past the last sample in the live buffer: the samples with
        file time <= T - rec_epoch - lag, cut back to a whole number of chunks (chunk=1: no chunking)."""
        n = np.searchsorted(self.ts, np.asarray(times) - self.rec_epoch - lag, side="right")
        return (n // chunk) * chunk


# ---------------- checkpoints ----------------
def checkpoint_timeline(model, runs):
    """Chronological [(t_from, path)] of the weights live in `model` ('A'/'B'): base checkpoint, then each block's
    ckpt_<model> from its swap_time (rows without a swap_time load nothing)."""
    events = [(-np.inf, BASE_CKPT[model])]
    for r in runs:
        for b in r.blocks:
            if b["swap_time"] is not None:
                events.append((b["swap_time"], b[f"ckpt_{model}"]))
    events.sort(key=lambda e: e[0])
    return events


def ckpt_at(timeline, t):
    path = timeline[0][1]
    for ts, p in timeline:
        if ts <= t:
            path = p
    return path


_MODEL_CACHE = {}


def load_model(path, device=DEVICE):
    """Build the model exactly like EEGPredictor.init_model; returns (model, channel_positions)."""
    path = os.path.normpath(path)
    if path not in _MODEL_CACHE:
        ck = torch.load(path, map_location=device, weights_only=False)
        n_chans = int(ck.get("n_chans", config.N_CHANNELS))
        model = config.USE_MODEL(**dict(config.LOAD_MODEL_PARAM, channels=n_chans)).to(device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        _MODEL_CACHE[path] = (model, config.channel_positions(n_chans))
    return _MODEL_CACHE[path]


# ---------------- prediction ----------------
def preprocess_windows(windows, positions):
    """windows: (N, 32, S) raw -> (N, len(positions), S) float32 as in predict_loop: bandpass on the model's channels, minus channel mean."""
    data = preprocess.bandpass(windows[:, positions, :], axis=-1)
    return (data - np.mean(data, axis=1, keepdims=True)).astype(np.float32)


@torch.no_grad()
def predict_windows(model, positions, windows):
    """windows (N, 32, S) float64 raw EEG -> (pred int array, prob1 array)."""
    x = torch.from_numpy(preprocess_windows(windows, positions)).to(DEVICE)
    out = model(x)
    return torch.argmax(out, dim=1).cpu().numpy(), torch.softmax(out, dim=1)[:, 1].cpu().numpy().astype(float)


def gather_windows(eeg, ends, length=config.BUFFER_SIZE):
    """eeg (n, 32); ends: sample counts (window = eeg[end-length:end]) -> (N, 32, length)."""
    idx = np.asarray(ends)[:, None] - length + np.arange(length)[None, :]
    return np.transpose(eeg[idx], (0, 2, 1))


def predict_ends(model, positions, eeg, ends, batch=1024):
    """Predict the 500-sample windows eeg[end-500:end] for each end; ends outside the recording give pred -1."""
    ends = np.asarray(ends)
    ok = (ends >= config.BUFFER_SIZE) & (ends <= len(eeg))
    pred, prob = np.full(len(ends), -1), np.full(len(ends), np.nan)
    sel = np.flatnonzero(ok)
    for i in range(0, len(sel), batch):
        j = sel[i:i + batch]
        pred[j], prob[j] = predict_windows(model, positions, gather_windows(eeg, ends[j]))
    return pred, prob


def predict_at(run, ckpt_path, times, lag, chunk=CHUNK):
    """Offline prediction for the 1 s windows a predictor saw at the given time.time() stamps."""
    model, pos = load_model(ckpt_path)
    return predict_ends(model, pos, run.eeg, run.window_ends(times, lag, chunk))


def predict_rows(run, timeline, name, rows, lag, chunk=CHUNK):
    """Offline pred/prob for rows of run.preds[name], each with the checkpoint that was live at its time."""
    t = run.preds[name][rows, 0]
    ev_t = np.array([e[0] for e in timeline])
    k = np.searchsorted(ev_t, t, side="right") - 1
    pred, prob = np.full(len(rows), -1), np.full(len(rows), np.nan)
    for ki in np.unique(k):
        sel = np.flatnonzero(k == ki)
        pred[sel], prob[sel] = predict_at(run, timeline[ki][1], t[sel], lag, chunk)
    return pred, prob


def last_before(times, at, n=HISTORY_LENGTH):
    """Row indices of the last n predictions with time <= at (times sorted)."""
    k = int(np.searchsorted(times, at, side="right"))
    return np.arange(max(0, k - n), k)
