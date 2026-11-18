# Shared helpers for the finetune evaluation (imported by the scripts in this folder, no run command of its own)
"""
Sessions, checkpoint chains, finetune lists and F1 windows for docs/plan_finetune_evaluation.md.

A session = all usable runs (those with predictions.csv) of python/main/real_time_data/<sub>/ses1. A run without ab_blocks.csv
(sub05 run 2, Static) is loaded as test-only: no swaps, display blocks of 4 trials. The chain of a model is its base checkpoint
c?_000 plus the checkpoint of every ab_blocks row with ft_<model> == 1, live from that row's swap_time.
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cut_compare"))
import common as C  # noqa: E402

DATA_ROOT = os.path.join(C.PY_DIR, "main", "real_time_data")
OUT_DIR = os.path.join(C.ANALYSIS_DIR, "results", "finetune_eval")
CACHE_ROOT = os.path.join(tempfile.gettempdir(), "finetune_eval_cache")
SESSIONS = {"sub01": [1, 2, 3, 4, 5, 6, 7], "sub02": [1], "sub03": [1, 2, 3], "sub04": [1, 2], "sub05": [1, 2]}
SUB01_LAGS = os.path.join(C.RESULTS_DIR, "step0a_lags.json")
W2_TRIALS = 48
PRE_TRIALS = 12
DISPLAY_BLOCK = 4


def session_dir(sub):
    return os.path.join(DATA_ROOT, sub, "ses1")


def lags_path(sub):
    return SUB01_LAGS if sub == "sub01" else os.path.join(OUT_DIR, f"step0a_{sub}", "step0a_lags.json")


def local_ckpt(sub, path):
    """ab_blocks stores absolute paths of the recording machine: resolve by file name (c?_000 -> pretrained base)."""
    name = os.path.basename(str(path).replace("\\", "/"))
    if name.endswith("_000.pth"):
        return C.BASE_CKPT[name[1]]
    return os.path.join(session_dir(sub), "checkpoints", name)


def ckpt_name(path):
    return os.path.basename(str(path).replace("\\", "/"))[:-4]


# ---------------- run loading ----------------
class TRun(C.Run):
    """C.Run that also loads a run without ab_blocks.csv as test-only (no swaps; display blocks of 4 trials)."""

    def __init__(self, run, session_dir_, cache_dir, load_eeg=True):
        d = C.run_dir(run, session_dir_)
        blocks_path = os.path.join(d, "ab_blocks.csv")
        self.test_only = not os.path.exists(blocks_path)
        if not self.test_only:
            super().__init__(run, session_dir_, cache_dir, load_eeg)
            return
        self.run = run
        self.name = f"run{run}"
        csv_path = glob.glob(os.path.join(d, "eeg_record_*.csv"))[0]
        self.trials = C.parse_log(glob.glob(os.path.join(d, "log_*.txt"))[0])
        self.preds = C.read_predictions(os.path.join(d, "predictions.csv"))
        self.rec_epoch = C.record_epoch(csv_path)
        self.data_files = sorted(glob.glob(os.path.join(d, "data_*.pt")))
        self.ts = self.eeg = None
        if load_eeg:
            self.ts, self.eeg = cached_eeg(csv_path, self.name, cache_dir)
        n = len(self.trials)
        self.blocks = [{"block": i + 1, "n_trials": min(DISPLAY_BLOCK, n - i * DISPLAY_BLOCK), "swap_time": None, "ft_A": "0", "ft_B": "0",
                        "ckpt_A": None, "ckpt_B": None} for i in range(-(-n // DISPLAY_BLOCK))]
        for bi in range(len(self.blocks)):
            for t in self.trials[bi * DISPLAY_BLOCK:(bi + 1) * DISPLAY_BLOCK]:
                t["block"] = bi
                t["run"] = run


def cached_eeg(csv_path, name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key = os.path.join(cache_dir, f"{name}_{os.path.getsize(csv_path)}")
    if os.path.exists(key + "_eeg.npy"):
        return np.load(key + "_ts.npy"), np.load(key + "_eeg.npy")
    ts, eeg = C.read_eeg(csv_path)
    np.save(key + "_ts.npy", ts)
    np.save(key + "_eeg.npy", eeg)
    return ts, eeg


# ---------------- chains, finetunes ----------------
def build_chain(blocks_per_run, model, base_path):
    """Chain of `model` from ab_blocks rows (list of block lists, one per run, in run order): [{'path','time'}], base first (time None).
    A row with ft_<model> == 1 starts a new checkpoint at its swap_time; rows without a finetune repeat the previous checkpoint and are skipped."""
    chain = [{"path": base_path, "time": None}]
    for blocks in blocks_per_run:
        for b in blocks:
            if str(b[f"ft_{model}"]) == "1":
                if b["swap_time"] is None:
                    raise ValueError(f"block {b['block']}: ft_{model}=1 without swap_time")
                if ckpt_name(b[f"ckpt_{model}"]) == ckpt_name(chain[-1]["path"]):
                    raise ValueError(f"block {b['block']}: finetune of {model} repeats {ckpt_name(chain[-1]['path'])}")
                chain.append({"path": b[f"ckpt_{model}"], "time": b["swap_time"]})
    return chain


def finetune_list(chain, model):
    """One entry per finetune: parent = the chain checkpoint before the child."""
    return [{"model": model, "k": i, "parent_index": i - 1, "child_index": i, "parent": ckpt_name(chain[i - 1]["path"]),
             "child": ckpt_name(chain[i]["path"]), "swap_time": chain[i]["time"]} for i in range(1, len(chain))]


def live_index(chain, at):
    """Chain index of the checkpoint live at time `at` (a swap at time t is live for at >= t)."""
    k = 0
    for i, c in enumerate(chain):
        if c["time"] is not None and c["time"] <= at:
            k = i
    return k


def in_sample(at, live_time):
    """The trial's vote time is before the checkpoint went live (c?_000: never in-sample)."""
    return live_time is not None and at < live_time


def windows(ats, swap_time, next_swap_time=None, n2=W2_TRIALS):
    """Indices (into the time-sorted `ats`) of W1 = swap_time < AT <= next swap (or end) and W2 = the first n2 trials with AT > swap_time."""
    ats = np.asarray(ats, float)
    after = np.flatnonzero(ats > swap_time)
    end = np.inf if next_swap_time is None else next_swap_time
    return after[ats[after] <= end], after[:n2]


def live_tag(live, model):
    """True where `model` was the live model (LIVE= of the log) on the trial."""
    return np.asarray([x == model for x in live], bool)


def pre_trials(ats, swap_time, after_time, n=PRE_TRIALS):
    """The last n trials with AT <= swap_time that are also after `after_time` (the parent's own swap; None: any)."""
    ats = np.asarray(ats, float)
    ok = ats <= swap_time
    if after_time is not None:
        ok &= ats > after_time
    return np.flatnonzero(ok)[-n:]


# ---------------- session ----------------
class Session:
    def __init__(self, sub, load_eeg=True, runs=None):
        self.sub = sub
        self.dir = session_dir(sub)
        cache = os.path.join(CACHE_ROOT, sub)
        self.runs = [TRun(r, self.dir, cache, load_eeg) for r in (runs or SESSIONS[sub])]
        self.trials = []
        for r in self.runs:
            for t in r.trials:
                if t["at"] is None:
                    raise ValueError(f"{sub} {r.name} trial {t['idx']}: no vote time AT")
                self.trials.append(t)
        for i, t in enumerate(self.trials):
            t["seq"] = i
        self.ats = np.array([t["at"] for t in self.trials])
        if not np.all(np.diff(self.ats) > 0):
            raise ValueError(f"{sub}: vote times are not increasing across trials")
        self.labels = np.array([t["label"] for t in self.trials])
        self.live = np.array([t["live"] for t in self.trials])
        self.chain, self.finetunes = {}, []
        for m in "AB":
            self.chain[m] = build_chain([r.blocks for r in self.runs if not r.test_only], m, C.BASE_CKPT[m])
            self.finetunes += finetune_list(self.chain[m], m)
        self.lags = None

    def load_lags(self):
        with open(lags_path(self.sub)) as f:
            self.lags = json.load(f)
        return self.lags

    def timeline(self, model):
        """[(t_from, path)] for C.ckpt_at-style lookups (local paths)."""
        return [(-np.inf if c["time"] is None else c["time"], local_ckpt(self.sub, c["path"])) for c in self.chain[model]]

    def ckpt_files(self, model):
        return [local_ckpt(self.sub, c["path"]) for c in self.chain[model]]
