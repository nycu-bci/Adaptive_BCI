"""
Replica of the original online trial vote (gate study V0-V2), as opposed to the simulator's 16-window majority.

Online (VR_BCI handover code on the NAS, = the code that recorded subjects 35-70):
  - Python predicts continuously: each prediction uses the last 500 samples received (1 s), band-pass 1-40 Hz
    (preprocess.bandpass, filtfilt), then CAR over the model's channels; about 70 predictions per second
    (comment in the old config.py: PREDICTION_INTERVAL 0.01 s, "in practice about 70 per second").
  - Unity (AutoSaber.cs) keeps the last 100 predictions and, when it swings at the 5th (brain-controlled) note, votes
    with linear weights 1 (oldest) -> 5 (newest): count1 += round(weight), class 1 if count1 >= total weight / 2.
    The trial is a success (5th CUT logged) if that class is the note's side.
Here:
  - the EEG stream is the run's eeg_record_*.csv; its sample times are relative to the header's Record datetime
    (local time, UTC+8). With that, the [START, START + 2000 samples) cut equals mi_22.pt exactly (checked).
  - vote time = reference - lead; reference = the 5th CUT; a failed trial (4 CUTs) has no 5th CUT, so its reference is
    the 4th CUT + the run's median 4th -> 5th CUT gap (fallback: START + the median START -> 5th CUT gap).
  - prediction j (j = 0 oldest .. 99 newest) ends at vote time - (99 - j) / rate.
"""

import datetime
import glob
import os
import re

import numpy as np

from main.Utils import preprocess

HIST = 100
WIN = 500
TZ = datetime.timezone(datetime.timedelta(hours=8))
C13 = [
    7,
    8,
    9,
    12,
    13,
    14,
    17,
    18,
    19,
    21,
    22,
    23,
    28,
]  # CSV columns of the old 13-channel model (= mi.pt, checked equal)


def unity_weights(n=HIST):
    w = 1.0 + 4.0 * np.arange(n) / (n - 1)
    return w, np.round(w)  # np.round halves to even, like C# Mathf.RoundToInt


def unity_vote(P):
    """P: (..., n) 0/1 predictions, oldest first. Returns the voted class (0/1), shape P.shape[:-1]."""
    w, rw = unity_weights(P.shape[-1])
    return ((P * rw).sum(-1) >= w.sum() / 2.0).astype(int)


def unity_correct(P, labels):
    """correct (bool) and margin = weight share of the predictions that equal the label"""
    w, _ = unity_weights(P.shape[-1])
    correct = unity_vote(P) == labels
    margin = ((P == labels[..., None]) * w).sum(-1) / w.sum()
    return correct, margin


def parse_log(path):
    """trials in id order that have START, END and LABEL: dicts start, end, label, cut (list), wrong_cut"""
    T = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.match(r"Trial\s+(\d+)\s+(START|END|CUT|WRONG_CUT):\s*([\d.]+)(?:\s+LABEL:\s*(\d+))?", line.strip())
        if not m:
            continue
        t = T.setdefault(int(m[1]), {"id": int(m[1]), "cut": []})
        if m[2] == "CUT":
            t["cut"].append(float(m[3]))
        else:
            t[m[2].lower()] = float(m[3])
        if m[4] is not None:
            t["label"] = int(m[4])
    return [T[i] for i in sorted(T) if {"start", "end", "label"} <= set(T[i])]


def read_stream(rdir):
    """-> (ts relative to the record start, X (34, n) float64 indexed by CSV column, rec epoch, trials)"""
    import pandas as pd

    csvp = sorted(glob.glob(os.path.join(rdir, "eeg_record_*.csv")))[0]
    with open(csvp, encoding="utf-8", errors="ignore") as f:
        head = [next(f) for _ in range(11)]
    dt = next(h for h in head if "Record datetime" in h).split(":", 1)[1].strip()
    rec = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=TZ).timestamp()
    arr = pd.read_csv(csvp, skiprows=11, header=None, usecols=range(34), engine="c").to_numpy(np.float64)
    arr = arr[np.isfinite(arr).all(1)]
    trials = parse_log(sorted(glob.glob(os.path.join(rdir, "log_*.txt")))[0])
    return arr[:, 0], arr.T, rec, trials


def vote_refs(trials, fallback=None):
    """Reference time of the brain-controlled (5th) note and kind:
    5 = its CUT is logged (the brain vote was right), 4 = the 4th note's CUT is the last one: 4th CUT + the run's median
    4th -> 5th gap, 0 = START + the run's median START -> 5th note offset. Notes are told apart by time, not by count:
    the non-brain notes are also missed now and then (4-6 % each, CUT not logged).
    fallback: the dict this returns for another run of the session (used when a run has no full trial)."""
    full = [t for t in trials if len(t["cut"]) == 5]
    med = lambda v, k: float(np.median(v)) if v else (fallback or {}).get(k)  # noqa: E731
    o4 = med([t["cut"][3] - t["start"] for t in full], "o4")
    o5 = med([t["cut"][4] - t["start"] for t in full], "o5")
    o3 = med([t["cut"][2] - t["start"] for t in full], "o3")
    g45 = med([t["cut"][4] - t["cut"][3] for t in full], "g45")
    if None in (o3, o4, o5, g45):  # no full trial anywhere: the grand medians of the 13,198 logged trials
        short = np.median([t["end"] - t["start"] for t in trials]) < 4.2
        o3, o4, o5, g45 = (1.90, 2.61, 3.33, 0.72) if short else (2.50, 3.50, 4.50, 1.00)
    refs, kinds = [], []
    for t in trials:
        last = t["cut"][-1] - t["start"] if t["cut"] else -1.0
        if t["cut"] and last > (o4 + o5) / 2:
            refs.append(t["cut"][-1]), kinds.append(5)
        elif t["cut"] and last > (o3 + o4) / 2:
            refs.append(t["cut"][-1] + g45), kinds.append(4)
        else:
            refs.append(t["start"] + o5), kinds.append(0)
    return np.array(refs), np.array(kinds), dict(o3=o3, o4=o4, o5=o5, g45=g45)


def end_index(ts, rec, t):
    """number of samples recorded up to wall time t (the window ends there, exclusive)"""
    return np.searchsorted(ts, np.asarray(t) - rec, side="right")


def prep_windows(X, ends, chans, car=True, chunk=4096):
    """windows X[chans, e-500:e] for each end e -> band-pass, then CAR (online order). (n, ch, 500) float32"""
    ends = np.asarray(ends).reshape(-1)
    out = np.empty((len(ends), len(chans), WIN), dtype=np.float32)
    rows = np.asarray(chans)
    for i in range(0, len(ends), chunk):
        e = ends[i : i + chunk]
        idx = e[:, None] - WIN + np.arange(WIN)[None, :]  # (m, 500)
        w = X[rows][:, idx].transpose(1, 0, 2)  # (m, ch, 500)
        w = preprocess.bandpass(w, axis=-1)
        if car:
            w = w - w.mean(axis=1, keepdims=True)
        out[i : i + len(e)] = w
    return out


def vote_ends(ts, rec, refs, lead, rate):
    """(trials, 100) window end indices, oldest first"""
    j = np.arange(HIST)
    times = refs[:, None] - lead - (HIST - 1 - j)[None, :] / rate
    return end_index(ts, rec, times)


def epoch_from_stream(ts, X, rec, trials, chans, n=2000):
    """[START, START + n) cut, as saveDataAsPtFromPointer (searchsorted left): (trial, ch, n)"""
    i0 = np.searchsorted(ts, np.array([t["start"] for t in trials]) - rec, side="left")
    return np.stack([X[np.asarray(chans)][:, a : a + n] for a in i0])
