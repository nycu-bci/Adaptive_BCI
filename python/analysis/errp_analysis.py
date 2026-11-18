# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/errp_analysis.py --session python/main/real_time_data/sub01/ses1 --out python/analysis/results/errp
"""
Is there an error-related potential (ErrP) after wrong brain-control votes (WRONG_CUT) vs correct ones (CUT)?
Plan: docs/plan_errp_analysis.md. One subject, one session; inference is over trials.

Steps: load + sanity + alignment check -> filter/epoch around the vote -> ERP + cluster/window statistics ->
confound controls -> leave-one-run-out single-trial decoding.

Outputs in --out: summary.txt, epochs_meta.csv, errp-epo.fif, figures e1-e7.
"""
import argparse
import datetime
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wald_sprt_explainer import (SURFACE, INK, INK2, MUTED, GRID, AXIS, MODEL_COLOR, RULE_COLOR, NEUTRAL_DARK,
                                 style, header, PY_DIR)

FS = 500
CH = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8", "T7", "C3", "Cz", "C4",
      "T8", "TP7", "CP3", "CPz", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
FC7 = ["Fz", "FC3", "FCz", "FC4", "Cz", "CPz", "Pz"]
ERP_CH = ["Fz", "FCz", "Cz", "CPz", "Pz", "O1", "Oz", "O2"]
OCC = ["O1", "Oz", "O2"]
FRONTOCENTRAL = ["Fz", "F3", "F4", "FC3", "FCz", "FC4", "Cz", "C3", "C4"]
OCCIPITOPARIETAL = ["O1", "Oz", "O2", "P7", "P8"]
EXPECTED = {"run1": (65, 31), "run2": (38, 10), "run3": (36, 12), "run4": (36, 12), "run5": (38, 10),
            "run6": (35, 13), "run7": (40, 8)}
TMIN, TMAX = -1.0, 1.0
BASE = (-0.2, 0.0)
REJECT_UV = 150.0
DEC = 5
C_OK, C_BAD, C_DIFF = MODEL_COLOR["A"], MODEL_COLOR["B"], INK

TRIAL_RE = re.compile(r"Trial\s+(\d+)\s+(START|CUT|WRONG_CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?", re.IGNORECASE)
END_RE = re.compile(r"Trial (\d+) END: [\d.]+ LABEL: (\d) MODEL_PREDICTION: A=(\w+) B=(\w+) LIVE=(\w) AT: ([\d.]+)")


# ---------------- loading ----------------
def parse_record_epoch(path):
    """Same logic as EEGDataLoader._extract_record_epoch (python/main/EEG/data_process_np.py)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(10):
            line = f.readline()
            if "Record datetime" in line:
                m = re.search(r"Record datetime:\s*([0-9\- :\.]+)", line)
                tz = datetime.timezone(datetime.timedelta(seconds=time.localtime().tm_gmtoff))
                dt = datetime.datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=tz)
                return dt.timestamp()
    raise ValueError("Record datetime not found")


def parse_log(path):
    """-> list of trial dicts (log order) with start, vote time, outcome, label, A/B votes.
    A new trial opens at every START (trial numbers restart inside a log, e.g. run1 has 0-47 twice)."""
    raw, cur = [], None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = TRIAL_RE.match(line)
            if not m:
                continue
            typ, ts = m.group(2).upper(), float(m.group(3))
            if typ == "START":
                cur = {"trial": int(m.group(1))}
                raw.append(cur)
            if cur is None:
                continue
            cur[typ] = ts
            if m.group(4) is not None:
                cur["label"] = int(m.group(4))
            m2 = END_RE.search(line)
            if m2:
                cur["A"], cur["B"] = m2.group(3), m2.group(4)
    out = []
    for e in raw:
        if "CUT" not in e and "WRONG_CUT" not in e:
            continue
        wrong = "WRONG_CUT" in e
        out.append({"trial": e["trial"], "label": e.get("label", -1), "start": e["START"], "vote": e["WRONG_CUT" if wrong else "CUT"],
                    "outcome": "WRONG_CUT" if wrong else "CUT", "A": e.get("A", ""), "B": e.get("B", "")})
    return out


def load_run(run_dir, cache_dir=None):
    """-> dict(ts, data (n, 32) float64 in uV, rec_epoch, name, trials). Parsed CSV is cached as .npy next to --out."""
    csvs = sorted(glob.glob(os.path.join(run_dir, "eeg_record_*.csv")))
    logs = sorted(glob.glob(os.path.join(run_dir, "log_*.txt")))
    if not csvs or not logs:
        return None
    trials = [t for lg in logs for t in parse_log(lg)]
    if not trials:
        return None
    path = csvs[0]
    name = os.path.basename(run_dir.rstrip("/\\"))
    st = os.stat(path)
    key = f"{st.st_size}_{int(st.st_mtime)}"
    cache = os.path.join(cache_dir, f"{name}_{key}.npy") if cache_dir else None
    if cache and os.path.exists(cache):
        arr = np.load(cache)
    else:
        df = pd.read_csv(path, skiprows=10, header=0)
        arr = df[["Timestamp"] + CH].to_numpy(dtype=np.float64)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            for old in glob.glob(os.path.join(cache_dir, f"{name}_*.npy")):
                os.remove(old)
            np.save(cache, arr)
    return {"name": name, "ts": arr[:, 0], "data": arr[:, 1:], "rec_epoch": parse_record_epoch(path), "trials": trials,
            "serial_gaps": None}


# ---------------- preprocessing / epoching ----------------
def preprocess(x, fs, l_freq, h_freq):
    """x (n, ch) -> (ch, n): linear detrend, zero-phase FIR band-pass (MNE), common average reference."""
    import mne
    from scipy import signal
    y = signal.detrend(x, axis=0, type="linear")
    y = mne.filter.filter_data(y.T, fs, l_freq, h_freq, method="fir", phase="zero", fir_design="firwin", verbose=False)
    return y - y.mean(axis=0, keepdims=True)


def extract_epochs(y, ts, times, fs, tmin, tmax):
    """Epochs (n_ev, ch, n_t) around EEG-relative event times (s), centred on the nearest sample. -> (epochs, valid)."""
    k0, k1 = int(round(tmin * fs)), int(round(tmax * fs))
    n_t = k1 - k0 + 1
    times = np.asarray(times, float)
    i = np.clip(np.searchsorted(ts, times), 1, len(ts) - 1)
    c = np.where(times - ts[i - 1] <= ts[i] - times, i - 1, i)
    valid = (times >= ts[0]) & (times <= ts[-1]) & (c + k0 >= 0) & (c + k1 < y.shape[1])
    out = np.full((len(times), y.shape[0], n_t), np.nan, dtype=np.float32)
    for j in np.where(valid)[0]:
        out[j] = y[:, c[j] + k0:c[j] + k1 + 1]
    return out, valid


def baseline(ep, t, win=BASE):
    m = (t >= win[0] - 1e-9) & (t <= win[1] + 1e-9)
    return ep - ep[..., m].mean(axis=-1, keepdims=True)


def sanity_run(run):
    from scipy import signal
    ts, x = run["ts"], run["data"]
    dt = np.diff(ts) * 1000.0
    expected = (ts[-1] - ts[0]) * FS + 1
    f, P = signal.welch(signal.detrend(x, axis=0), FS, nperseg=2048, axis=0)
    return {"n": len(ts), "dur": ts[-1] - ts[0], "dt": dt, "missing": expected - len(ts), "f": f, "P": P.T,
            "dc": np.abs(x.mean(axis=0)).max()}


def spectral_checks(f, P):
    """P (ch, F). Per channel: log-log slope 2-40 Hz without alpha/line bands; alpha peak ratio; 50/60 Hz line ratio."""
    base = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14)) & ~((f >= 48) & (f <= 52)) & ~((f >= 58) & (f <= 62))
    slope = np.array([np.polyfit(np.log10(f[base]), np.log10(p[base]), 1)[0] for p in P])
    fit = (((f >= 3) & (f <= 6)) | ((f >= 15) & (f <= 20)))
    al = (f >= 7) & (f <= 13)
    ratio, pk = [], []
    for p in P:
        c = np.polyfit(np.log10(f[fit]), np.log10(p[fit]), 1)
        pred = 10 ** np.polyval(c, np.log10(f[al]))
        r = p[al] / pred
        ratio.append(r.max())
        pk.append(f[al][r.argmax()])
    line = {}
    for fl in (50, 60):
        pk_ = (f >= fl - 0.5) & (f <= fl + 0.5)
        fl_ = ((f >= fl - 4) & (f <= fl - 2)) | ((f >= fl + 2) & (f <= fl + 4))
        line[fl] = np.array([p[pk_].max() / p[fl_].mean() for p in P])
    return slope, np.array(ratio), np.array(pk), line


def process_run(run, lag, fs=FS):
    """Epochs for one run. lag (s) is added to every event time (alignment fix). Returns arrays keyed by band."""
    ts, x, rec = run["ts"], run["data"], run["rec_epoch"]
    tr = run["trials"]
    vote = np.array([t["vote"] for t in tr]) - rec + lag
    start = np.array([t["start"] for t in tr]) - rec
    yb = preprocess(x, fs, 1.0, 10.0)
    xb, valid = extract_epochs(yb, ts, vote, fs, TMIN, TMAX)
    del yb
    yf = preprocess(x, fs, 0.1, 30.0)
    xf, _ = extract_epochs(yf, ts, vote, fs, TMIN, TMAX)
    sf, svalid = extract_epochs(yf, ts, start, fs, -0.5, 1.5)
    del yf
    t = np.arange(round(TMIN * fs), round(TMAX * fs) + 1) / fs
    ts_ax = np.arange(round(-0.5 * fs), round(1.5 * fs) + 1) / fs
    return {"xb": xb, "xf": baseline(xf, t), "valid": valid, "start_f": baseline(sf, ts_ax), "start_valid": svalid,
            "t": t, "t_start": ts_ax, "vote_rel": vote, "start_rel": start}


def p2p(ep):
    return (np.nanmax(ep, axis=-1) - np.nanmin(ep, axis=-1)).max(axis=-1)


def peak_latency(erp, t, lo=0.05, hi=0.6):
    m = (t >= lo) & (t <= hi)
    return t[m][np.argmax(np.abs(erp[m]))]


def estimate_lags(erps, t, max_lag=0.3, win=(0.0, 0.8), fs=FS, n_iter=5):
    """erps: {run: (ch, T) mean START ERP, n}. Iteratively: template = mean of the other runs shifted by their current lags;
    each run's lag = shift s maximising corr(template[t], run[t + s]) over all channels, t in win. Lags are re-centred to
    mean 0 (only relative offsets are identifiable). Positive lag = the run's response comes later than the log implies.
    -> {run: (lag_s, r)}"""
    names = [n for n in erps if erps[n][1] > 0]
    k = np.arange(-int(round(max_lag * fs)), int(round(max_lag * fs)) + 1)
    m = np.where((t >= win[0]) & (t <= win[1]))[0]
    lag = {n: 0 for n in names}
    corr = {n: np.nan for n in names}
    for _ in range(n_iter if len(names) > 1 else 0):
        new = {}
        for r in names:
            oth = [o for o in names if o != r]
            w = np.array([erps[o][1] for o in oth], float)
            tpl = sum(erps[o][0][:, m + lag[o]] * wi for o, wi in zip(oth, w)) / w.sum()
            a = tpl.ravel() - tpl.mean()
            best = (-2, 0)
            for sh in k:
                if m[0] + sh < 0 or m[-1] + sh >= erps[r][0].shape[1]:
                    continue
                bvec = erps[r][0][:, m + sh].ravel()
                bvec = bvec - bvec.mean()
                c = (a @ bvec) / (np.linalg.norm(a) * np.linalg.norm(bvec) + 1e-30)
                if c > best[0]:
                    best = (c, sh)
            new[r], corr[r] = best[1], best[0]
        mean = int(round(np.mean(list(new.values()))))
        new = {r: v - mean for r, v in new.items()}
        if new == lag:
            break
        lag = new
    return {n: ((lag[n] / fs, corr[n]) if n in lag else (0.0, np.nan)) for n in erps}


def build_dataset(session, cache_dir, align="none", log=print):
    """Loads every run with a non-empty log, epochs the vote. -> dict with meta, epochs, sanity, alignment info."""
    run_dirs = sorted(glob.glob(os.path.join(session, "run*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
    res, sanity, skipped = {}, {}, []
    for d in run_dirs:
        run = load_run(d, cache_dir)
        if run is None:
            skipped.append(os.path.basename(d))
            continue
        log(f"  {run['name']}: {run['data'].shape[0]} samples, {len(run['trials'])} trials")
        sanity[run["name"]] = sanity_run(run)
        res[run["name"]] = (run, process_run(run, 0.0))
    t, t_start = next(iter(res.values()))[1]["t"], next(iter(res.values()))[1]["t_start"]
    # alignment measured on the unshifted data
    align_info = {}
    erps = {}
    for name, (run, p) in res.items():
        ok_s = p["start_valid"] & (p2p(p["start_f"]) < REJECT_UV)
        ok_c = p["valid"] & (p2p(p["xb"]) < REJECT_UV) & np.array([tr["outcome"] == "CUT" for tr in run["trials"]])
        occ = [CH.index(c) for c in OCC]
        s_erp = np.nanmean(p["start_f"][ok_s], axis=0) if ok_s.any() else np.zeros((len(CH), len(t_start)))
        c_erp = np.nanmean(p["xf"][ok_c], axis=0) if ok_c.any() else np.zeros((len(CH), len(t)))
        erps[name] = (s_erp, int(ok_s.sum()))
        align_info[name] = {"start_erp": s_erp[occ].mean(0), "cut_erp": c_erp[occ].mean(0), "n_start": int(ok_s.sum()),
                            "n_cut": int(ok_c.sum()),
                            "start_peak": peak_latency(s_erp[occ].mean(0), t_start), "cut_peak": peak_latency(c_erp[occ].mean(0), t),
                            "first_start_rel": float(p["start_rel"][0])}
    lags = estimate_lags({n: (e, k) for n, (e, k) in erps.items()}, t_start)
    for n, (lag, r) in lags.items():
        align_info[n]["lag"], align_info[n]["r"] = lag, r
    applied = {n: 0.0 for n in res}
    if align == "xcorr":
        for n, (run, _) in list(res.items()):
            applied[n] = lags[n][0]
            if abs(lags[n][0]) > 0:
                res[n] = (run, process_run(run, lags[n][0]))
    # assemble
    rows, xb, xf = [], [], []
    for name, (run, p) in res.items():
        pp = np.where(p["valid"], p2p(np.nan_to_num(p["xb"], nan=0.0)), np.nan)
        for j, tr in enumerate(run["trials"]):
            rows.append({"run": name, "trial": tr["trial"], "label": tr["label"], "outcome": tr["outcome"],
                         "correct": tr["outcome"] == "CUT", "A_vote": tr["A"], "B_vote": tr["B"],
                         "vote_minus_start": tr["vote"] - tr["start"], "start_rel": p["start_rel"][j],
                         "vote_rel": p["vote_rel"][j], "lag_applied": applied[name], "p2p_uv": pp[j],
                         "rejected": (not p["valid"][j]) or pp[j] > REJECT_UV,
                         "reason": "out_of_range" if not p["valid"][j] else ("p2p" if pp[j] > REJECT_UV else "")})
        xb.append(p["xb"])
        xf.append(p["xf"])
    meta = pd.DataFrame(rows)
    xb, xf = np.concatenate(xb), np.concatenate(xf)
    keep = ~meta["rejected"].to_numpy()
    xb_raw = xb[keep]
    xb = baseline(xb_raw, t)
    return {"meta": meta, "keep": keep, "xb": xb.astype(np.float32), "xb_nobase": xb_raw, "xf": xf[keep], "t": t,
            "sanity": sanity, "align": align_info, "applied": applied, "skipped": skipped, "res_names": list(res)}


# ---------------- statistics ----------------
def welch_t(a, b):
    na, nb = len(a), len(b)
    return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / na + b.var(0, ddof=1) / nb + 1e-30)


def perm_welch(D, y, strata, n_perm, rng):
    """Welch t of y==1 minus y==0 per column of D (n, F); labels permuted globally or within strata.
    -> t_obs, p per column (two-sided), p family-wise (max |t|)."""
    D = np.asarray(D, float)
    y = np.asarray(y, bool)
    D2 = D ** 2
    tot, tot2 = D.sum(0), D2.sum(0)
    n = len(y)
    n1 = int(y.sum())
    n0 = n - n1

    def tstat(m):
        s1, q1 = m @ D, m @ D2
        s0, q0 = tot - s1, tot2 - q1
        m1, m0 = s1 / n1, s0 / n0
        return (m1 - m0) / np.sqrt(np.maximum((q1 - n1 * m1 ** 2) / (n1 - 1), 0) / n1 +
                                   np.maximum((q0 - n0 * m0 ** 2) / (n0 - 1), 0) / n0 + 1e-30)

    obs = tstat(y.astype(float))
    groups = [np.arange(n)] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    cnt = np.zeros(D.shape[1])
    mx = np.empty(n_perm)
    yf = y.astype(float)
    for i in range(n_perm):
        m = yf.copy()
        for g in groups:
            m[g] = yf[rng.permutation(g)]
        tt = np.abs(tstat(m))
        cnt += tt >= np.abs(obs)
        mx[i] = tt.max()
    p = (cnt + 1) / (n_perm + 1)
    pfw = ((mx[None, :] >= np.abs(obs)[:, None]).sum(1) + 1) / (n_perm + 1)
    return obs, p, pfw


def window_means(xb, t, ch, win):
    m = (t >= win[0] - 1e-9) & (t <= win[1] + 1e-9)
    return xb[:, CH.index(ch), :][:, m].mean(1)


def window_tests(xb, t, y, runs, n_perm, rng):
    """Primary hypothesis windows: FCz 200-350 ms (negativity), Cz 300-600 ms (positivity); plus Fp1/Fp2 (EOG) and
    occipital mean as the visual check. y = wrong. Differences are wrong - correct."""
    specs = [("FCz 200-350", "FCz", (0.2, 0.35)), ("Cz 300-600", "Cz", (0.3, 0.6)), ("Fp1 200-600", "Fp1", (0.2, 0.6)),
             ("Fp2 200-600", "Fp2", (0.2, 0.6))]
    cols = [window_means(xb, t, c, w) for _, c, w in specs]
    names = [s[0] for s in specs]
    occ = np.mean([window_means(xb, t, c, (0.15, 0.7)) for c in OCC], axis=0)
    cols.append(occ)
    names.append("O1/Oz/O2 mean 150-700")
    D = np.stack(cols, 1)
    y = np.asarray(y, bool)
    t_obs, p_glob, _ = perm_welch(D, y, None, n_perm, rng)
    _, p_str, _ = perm_welch(D, y, runs, n_perm, rng)
    from scipy import stats
    rows = []
    for k, nm in enumerate(names):
        a, b = D[y, k], D[~y, k]
        rows.append({"name": nm, "diff": a.mean() - b.mean(), "t": t_obs[k], "p_welch": stats.ttest_ind(a, b, equal_var=False).pvalue,
                     "p_perm": p_glob[k], "p_perm_within_run": p_str[k], "n_wrong": len(a), "n_correct": len(b)})
    return rows


def welch_stat_fun(*groups):
    return welch_t(groups[0], groups[1])


def cluster_test(X, y, info, n_perm, seed, thr_p=0.05):
    """Spatio-temporal cluster permutation test over trials. X (n, times, ch); y True = wrong. t > 0: wrong more positive."""
    import mne
    from scipy import stats
    adj, _ = mne.channels.find_ch_adjacency(info, "eeg")
    n1, n0 = int(np.sum(y)), int(np.sum(~y))
    thr = stats.t.ppf(1 - thr_p / 2, n1 + n0 - 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, clusters, pv, _ = mne.stats.spatio_temporal_cluster_test(
            [X[y], X[~y]], threshold=thr, n_permutations=n_perm, tail=0, stat_fun=welch_stat_fun, adjacency=adj,
            n_jobs=1, seed=seed, out_type="mask", verbose=False)
    T = np.asarray(T).reshape(X.shape[1], X.shape[2])
    out = []
    for cl, p in zip(clusters, pv):
        cl = np.asarray(cl).reshape(X.shape[1], X.shape[2])
        out.append({"p": float(p), "mask": cl, "mass": float(T[cl].sum()), "sign": 1 if T[cl].mean() > 0 else -1,
                    "tidx": np.where(cl.any(1))[0], "cidx": np.where(cl.any(0))[0]})
    out.sort(key=lambda c: c["p"])
    return {"T": T, "clusters": out}


def describe_cluster(c, tvec, T):
    chs = [CH[i] for i in c["cidx"]]
    mass = np.abs(T) * c["mask"]
    fc = mass[:, [CH.index(x) for x in FRONTOCENTRAL]].sum() / mass.sum()
    oc = mass[:, [CH.index(x) for x in OCCIPITOPARIETAL]].sum() / mass.sum()
    return (f"p={c['p']:.4f}  {tvec[c['tidx'][0]] * 1000:.0f}-{tvec[c['tidx'][-1]] * 1000:.0f} ms  "
            f"{'wrong>correct' if c['sign'] > 0 else 'wrong<correct'}  mass={c['mass']:+.1f}  "
            f"share of |t| in frontocentral={fc:.2f}, occipito-parietal={oc:.2f}  channels({len(chs)}): {','.join(chs)}")


def stat_arrays(xb, t, t0, t1):
    idx = np.where((t >= t0 - 1e-9) & (t <= t1 + 1e-9))[0][::DEC]
    return np.transpose(xb[:, :, idx], (0, 2, 1)).astype(np.float64), t[idx]


def loro_folds(runs):
    runs = np.asarray(runs)
    return [(np.where(runs != r)[0], np.where(runs == r)[0]) for r in sorted(set(runs))]


def permute_within(y, runs, rng):
    out = np.asarray(y).copy()
    for r in np.unique(runs):
        g = np.where(runs == r)[0]
        out[g] = y[rng.permutation(g)]
    return out


# ---------------- decoding ----------------
def bin_means(X, t, edges):
    """X (n, c, T) -> (n, c, n_bins) mean in [edges[i], edges[i+1])."""
    return np.stack([X[..., (t >= a - 1e-9) & (t < b - 1e-9)].mean(-1) for a, b in zip(edges[:-1], edges[1:])], -1)


def xdawn_filters(X, y, n_comp):
    """xDAWN spatial filters: per class, generalised eigenvectors of (class-evoked cov, OAS-shrunk signal cov)."""
    from scipy.linalg import eigh
    from sklearn.covariance import OAS
    sig = OAS().fit(np.concatenate(list(X), axis=1).T).covariance_
    W = []
    for k in (0, 1):
        E = X[y == k].mean(0)
        w, v = eigh(E @ E.T, sig)
        W.append(v[:, np.argsort(w)[::-1][:n_comp]])
    return np.concatenate(W, axis=1)


def cv_decode(feat_fn, y, runs):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import roc_auc_score
    scores = np.full(len(y), np.nan)
    aucs = []
    for tr, te in loro_folds(runs):
        Ftr, Fte = feat_fn(tr, te, y)
        clf = LDA(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(Ftr, y[tr])
        scores[te] = clf.decision_function(Fte)
        aucs.append(roc_auc_score(y[te], scores[te]) if len(set(y[te])) == 2 else np.nan)
    return np.array(aucs), scores


def decode_with_null(feat_fn, y, runs, n_perm, rng):
    from sklearn.metrics import roc_auc_score
    aucs, scores = cv_decode(feat_fn, y, runs)
    obs = np.nanmean(aucs)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.nanmean(cv_decode(feat_fn, permute_within(y, runs, rng), runs)[0])
    pred = scores > 0
    tpr, tnr = pred[y == 1].mean(), (~pred[y == 0]).mean()
    return {"aucs": aucs, "mean_auc": obs, "pooled_auc": roc_auc_score(y, scores), "bacc": (tpr + tnr) / 2, "tpr": tpr,
            "tnr": tnr, "null": null, "p": (1 + np.sum(null >= obs)) / (1 + n_perm), "scores": scores}


def make_feat_fns(xb, xb_nobase, t, y, n_comp):
    """-> {name: feat_fn}. Post-vote features are baseline-corrected, the pre-vote control is not."""
    post = np.arange(0.15, 0.7001, 0.05)
    pre = np.arange(-0.55, 0.0001, 0.05)
    fns = {}
    for tag, X, edges, win in (("post", xb, post, (0.0, 1.0)), ("pre", xb_nobase, pre, (-1.0, 0.0))):
        idx7 = [CH.index(c) for c in FC7]
        F7 = bin_means(X[:, idx7, :], t, edges).reshape(len(X), -1)
        fns[f"{tag}_7ch"] = (lambda F: lambda tr, te, yy: (F[tr], F[te]))(F7)
        m = np.where((t >= win[0] - 1e-9) & (t <= win[1] + 1e-9))[0][::DEC]
        Xd = X[:, :, m].astype(np.float64)
        td = t[m]

        def fn(tr, te, yy, Xd=Xd, td=td, edges=edges):
            W = xdawn_filters(Xd[tr], yy[tr], n_comp)
            Z = np.einsum("cf,nct->nft", W, Xd)
            F = bin_means(Z, td, edges).reshape(len(Xd), -1)
            return F[tr], F[te]
        fns[f"{tag}_xdawn"] = fn
    return fns


# ---------------- figures ----------------
def make_info():
    import mne
    info = mne.create_info(CH, FS, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="raise", verbose=False)
    return info


def f_sanity(S, out, summary_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    f = next(iter(S.values()))["f"]
    P = np.mean([s["P"] for s in S.values()], axis=0)
    ax = axes[0]
    for c, col in zip(["O1", "Oz", "O2", "FCz", "Cz"], [RULE_COLOR[0], RULE_COLOR[1], RULE_COLOR[2], RULE_COLOR[3], INK2]):
        ax.loglog(f[1:], P[CH.index(c)][1:], color=col, lw=1.2, label=c)
    for fl in (10, 50, 60):
        ax.axvline(fl, color=AXIS, lw=0.8, ls="--")
    ax.set_xlim(0.5, 250)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV^2/Hz), mean over runs")
    ax.set_title("e1a. Channel spectra (raw, detrended)")
    ax.legend(fontsize=8.5, ncol=2)
    ax = axes[1]
    dt = np.concatenate([s["dt"] for s in S.values()])
    ax.hist(np.clip(dt, 0, 10), bins=np.linspace(0, 10, 101), color=NEUTRAL_DARK)
    ax.set_yscale("log")
    ax.set_xlabel("Sample interval (ms), clipped at 10")
    ax.set_ylabel("Samples")
    ax.set_title("e1b. Sample intervals (expected 2 ms)")
    ax = axes[2]
    slope = summary_rows["slope"]
    ax.bar(range(len(CH)), slope, color=[RULE_COLOR[1] if c in OCC else NEUTRAL_DARK for c in CH])
    ax.set_xticks(range(len(CH)), CH, rotation=90, fontsize=7.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Log-log slope of PSD, 2-40 Hz")
    ax.set_title("e1c. Spectral slope (white noise = 0)")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.17, wspace=0.28)
    fig.savefig(os.path.join(out, "e1_sanity.png"), dpi=150)
    plt.close(fig)


def f_alignment(A, t_start, t, out):
    runs = list(A)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    for ax, key, tt, title in ((axes[0], "start_erp", t_start, "e2a. START-locked, mean O1/Oz/O2 (0.1-30 Hz)"),
                               (axes[1], "cut_erp", t, "e2b. Vote-locked, correct trials only, O1/Oz/O2")):
        for r, c in zip(runs, cols):
            ax.plot(tt, A[r][key], color=c, lw=1.1, label=r)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-0.5, 1.0)
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel("uV")
        ax.set_title(title, fontsize=10.5)
    axes[0].legend(fontsize=8, ncol=2)
    ax = axes[2]
    ax.bar(range(len(runs)), [A[r]["lag"] * 1000 for r in runs], color=RULE_COLOR[0])
    ax.set_xticks(range(len(runs)), [r.replace("run", "") for r in runs])
    ax.set_xlabel("Run")
    ax.set_ylabel("Estimated lag (ms)")
    ax.set_title("e2c. Cross-correlation lag vs pooled START ERP", fontsize=10.5)
    ax.axhline(0, color=INK, lw=0.8)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.13, wspace=0.25)
    fig.savefig(os.path.join(out, "e2_alignment.png"), dpi=150)
    plt.close(fig)


def f_erp(xf, t, y, out, rng, n_boot=500):
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    for ax, c in zip(axes.T.ravel(), ERP_CH):
        k = CH.index(c)
        a, b = xf[y][:, k], xf[~y][:, k]
        ma, mb = a.mean(0), b.mean(0)
        boot = np.empty((n_boot, len(t)))
        for i in range(n_boot):
            boot[i] = a[rng.integers(0, len(a), len(a))].mean(0) - b[rng.integers(0, len(b), len(b))].mean(0)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax.plot(t, mb, color=C_OK, lw=1.3, label=f"correct (n={len(b)})")
        ax.plot(t, ma, color=C_BAD, lw=1.3, label=f"wrong (n={len(a)})")
        ax.plot(t, ma - mb, color=C_DIFF, lw=1.4, label="wrong - correct")
        ax.fill_between(t, lo, hi, color=C_DIFF, alpha=0.18, lw=0, label="95% bootstrap CI of difference")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        if c == "FCz":
            ax.axvspan(0.2, 0.35, color=C_BAD, alpha=0.1)
        if c == "Cz":
            ax.axvspan(0.3, 0.6, color=C_BAD, alpha=0.1)
        ax.set_xlim(-0.5, 1.0)
        ax.set_title(c + ("  (visual control)" if c in OCC else ""), fontsize=10.5)
        ax.set_ylabel("uV")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time from vote (s)")
    header(fig, "e3. Vote-locked ERPs, wrong vs correct (0.1-30 Hz, CAR, baseline -0.2-0 s)",
           "Shaded bands on FCz / Cz = primary test windows. Difference = wrong - correct.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.05, hspace=0.3, wspace=0.18)
    fig.savefig(os.path.join(out, "e3_erp.png"), dpi=150)
    plt.close(fig)


def f_topo(xf, t, y, info, out):
    import mne
    wins = [(0.15, 0.25), (0.25, 0.35), (0.35, 0.5), (0.5, 0.7)]
    D = [window_means_all(xf[y], t, w).mean(0) - window_means_all(xf[~y], t, w).mean(0) for w in wins]
    v = max(np.abs(d).max() for d in D)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, d, w in zip(axes, D, wins):
        im, _ = mne.viz.plot_topomap(d, info, axes=ax, show=False, vlim=(-v, v), cmap="RdBu_r", contours=0, sensors=True)
        ax.set_title(f"{int(w[0] * 1000)}-{int(w[1] * 1000)} ms", fontsize=10.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("wrong - correct (uV)")
    header(fig, "e4. Topography of the wrong - correct difference (0.1-30 Hz, CAR)",
           "Front of the head at the top. An ErrP would be fronto-central; a visual / no-motion response occipito-parietal.")
    fig.subplots_adjust(left=0.02, right=0.9, top=0.78, bottom=0.03)
    fig.savefig(os.path.join(out, "e4_topomaps.png"), dpi=150)
    plt.close(fig)


def window_means_all(X, t, w):
    return X[:, :, (t >= w[0] - 1e-9) & (t < w[1] - 1e-9)].mean(-1)


def f_cluster(post, pre, tpost, tpre, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1]})
    v = max(np.abs(post["T"]).max(), np.abs(pre["T"]).max())
    for ax, res, tv, title in ((axes[0], post, tpost, "e5a. Post-vote (0-1 s)"), (axes[1], pre, tpre, "e5b. Pre-vote control (-1-0 s)")):
        im = ax.imshow(res["T"].T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                       extent=[tv[0], tv[-1], len(CH) - 0.5, -0.5], interpolation="nearest")
        for i, c in enumerate(res["clusters"][:3]):
            M = c["mask"].T.astype(float)
            ax.contour(tv, np.arange(len(CH)), M, levels=[0.5], colors=[INK], linewidths=1.6 if c["p"] < 0.05 else 0.8,
                       linestyles="-" if c["p"] < 0.05 else "--")
        best = res["clusters"][0]["p"] if res["clusters"] else float("nan")
        ax.set_yticks(range(len(CH)), CH, fontsize=7.5)
        ax.set_xlabel("Time from vote (s)")
        ax.set_title(f"{title}  best cluster p={best:.3f}", fontsize=10.5)
        ax.grid(False)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.84, bottom=0.07, wspace=0.12)
    cax = fig.add_axes([0.92, 0.2, 0.012, 0.55])
    fig.colorbar(im, cax=cax, label="Welch t (wrong - correct)")
    header(fig, "e5. Spatio-temporal cluster test: t-map, wrong vs correct (1-10 Hz, 100 Hz, CAR)",
           "Solid outline = cluster with p < 0.05, dashed = other clusters (up to 3, by p). Channels ordered front to back.")
    fig.savefig(os.path.join(out, "e5_cluster.png"), dpi=150)
    plt.close(fig)


def f_controls(xb, xb_nb, t, meta, y, null_curves, out):
    label = meta["label"].to_numpy()
    runs = meta["run"].to_numpy()
    k_fcz, k_cz = CH.index("FCz"), CH.index("Cz")

    def diff(sel, k, X=xb):
        return X[sel & y][:, k].mean(0) - X[sel & ~y][:, k].mean(0)
    allm = np.ones(len(y), bool)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    ax = axes[0, 0]
    ax.plot(t, diff(allm, k_fcz), color=C_DIFF, lw=1.6, label="all")
    for l, col in ((0, RULE_COLOR[0]), (1, RULE_COLOR[1])):
        s = label == l
        ax.plot(t, diff(s, k_fcz), color=col, lw=1.1, label=f"label {l} (wrong {int((s & y).sum())}, correct {int((s & ~y).sum())})")
    ax.set_title("e6a. FCz difference per MI label", fontsize=10.5)
    ax = axes[0, 1]
    ax.plot(t, diff(allm, k_fcz), color=C_DIFF, lw=1.6, label="FCz all runs")
    ax.plot(t, diff(runs != "run1", k_fcz), color=RULE_COLOR[2], lw=1.1, label="FCz without run1")
    ax.plot(t, diff(allm, k_cz), color=C_DIFF, lw=1.0, ls="--", label="Cz all runs")
    ax.plot(t, diff(runs != "run1", k_cz), color=RULE_COLOR[2], lw=1.0, ls="--", label="Cz without run1")
    ax.set_title("e6b. Excluding run1 (31 of 96 wrong trials)", fontsize=10.5)
    ax = axes[1, 0]
    ax.plot(t, diff(allm, k_fcz, xb_nb), color=C_BAD, lw=1.3, label="FCz")
    ax.plot(t, diff(allm, k_cz, xb_nb), color=RULE_COLOR[3], lw=1.3, label="Cz")
    ax.set_title("e6c. Pre-vote: no baseline, 1-10 Hz", fontsize=10.5)
    ax = axes[1, 1]
    for c in null_curves:
        ax.plot(t, c, color=AXIS, lw=0.7)
    ax.plot(t, diff(allm, k_fcz), color=C_BAD, lw=1.6, label="real labels")
    ax.plot([], [], color=AXIS, label=f"{len(null_curves)} within-run label shuffles")
    ax.set_title("e6d. Sanity null: FCz difference with shuffled labels", fontsize=10.5)
    for ax in axes.ravel():
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Time from vote (s)")
        ax.set_ylabel("wrong - correct (uV)")
        ax.legend(fontsize=8)
    header(fig, "e6. Controls (1-10 Hz, CAR; a, b, d baseline -0.2-0 s)", "Differences are wrong - correct.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07, hspace=0.32, wspace=0.18)
    fig.savefig(os.path.join(out, "e6_controls.png"), dpi=150)
    plt.close(fig)


def f_decoding(D, y, runs, out):
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    titles = {"post_7ch": "7 fronto-central channels", "post_xdawn": "xDAWN, all 32 channels"}
    for ax, key in zip(axes[0], ("post_7ch", "post_xdawn")):
        r = D[key]
        sc = r["scores"]
        for rr in sorted(set(runs)):
            m = runs == rr
            if len(set(y[m])) == 2:
                fpr, tpr, _ = roc_curve(y[m], sc[m])
                ax.plot(fpr, tpr, color=NEUTRAL_DARK, lw=0.9)
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=C_BAD, lw=1.8, label=f"pooled AUC {r['pooled_auc']:.2f}; mean fold AUC {r['mean_auc']:.2f}")
        ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (wrong detected)")
        ax.set_title(f"e7. Leave-one-run-out ROC: {titles[key]}", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="lower right")
    for ax, tag in zip(axes[1], ("7ch", "xdawn")):
        for pre, col in (("post", C_BAD), ("pre", RULE_COLOR[3])):
            r = D[f"{pre}_{tag}"]
            ax.hist(r["null"], bins=30, color=col, alpha=0.4, label=f"{pre}-vote null")
            ax.axvline(r["mean_auc"], color=col, lw=2, label=f"{pre}-vote observed {r['mean_auc']:.3f} (p={r['p']:.3f})")
        ax.set_xlabel("Mean fold AUC")
        ax.set_ylabel("Permutations")
        ax.set_title(f"e7. Within-run label permutation null: {titles['post_' + tag]}", fontsize=10.5)
        ax.legend(fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07, hspace=0.3, wspace=0.2)
    fig.savefig(os.path.join(out, "e7_decoding.png"), dpi=150)
    plt.close(fig)


# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join(PY_DIR, "main", "real_time_data", "sub01", "ses1"))
    ap.add_argument("--out", default=os.path.join(PY_DIR, "analysis", "results", "errp"))
    ap.add_argument("--align", choices=["none", "xcorr"], default="none")
    ap.add_argument("--n_perm_cluster", type=int, default=5000)
    ap.add_argument("--n_perm_decode", type=int, default=1000)
    ap.add_argument("--n_perm_window", type=int, default=10000)
    ap.add_argument("--n_comp", type=int, default=2, help="xDAWN components per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    style()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    L = []

    def say(s=""):
        L.append(s)
        print(s, flush=True)

    say("ErrP analysis: wrong (WRONG_CUT) vs correct (CUT) votes")
    say(f"session: {args.session}   align: {args.align}   seed: {args.seed}")
    say(f"permutations: cluster {args.n_perm_cluster}, window {args.n_perm_window}, decoding {args.n_perm_decode}")
    say("loading runs:")
    ds = build_dataset(args.session, os.path.join(args.out, "cache"), args.align, log=say)
    meta, t = ds["meta"], ds["t"]
    keep = ds["keep"]
    mk = meta[keep].reset_index(drop=True)
    y = (~mk["correct"]).to_numpy()
    runs = mk["run"].to_numpy()
    xb, xb_nb, xf = ds["xb"], ds["xb_nobase"], ds["xf"]
    meta.to_csv(os.path.join(args.out, "epochs_meta.csv"), index=False)
    say(f"skipped runs (no trials in log): {ds['skipped']}")

    say()
    say("CONFOUNDS (read every result below with these in mind)")
    say("  1. The feedback is asymmetric. A correct vote swings the saber and slices the cube; on a wrong vote AutoSaber.cs")
    say("     breaks out, so there is no swing and the cube flies past. wrong - correct therefore mixes error, 'no motion vs")
    say("     motion' and rarity (25% vs 75%). An ErrP-like effect is fronto-central (FCz/Cz, ~200-350 ms negativity, then")
    say("     positivity); a visual/motion effect is occipito-parietal. The error also becomes evident gradually on wrong")
    say("     trials (cube passes), so any ErrP may be later and smeared.")
    say("  2. rec_epoch ('Record datetime') is taken before the LSL inlet connects, so each run has an unknown constant offset")
    say("     between EEG time and the log clock. Estimated below from the START-locked response; absolute latencies stay uncertain.")

    # ---- step 1 ----
    say()
    say("== 1. Counts, timing, sanity ==")
    cnt = meta.groupby("run")["correct"].agg(["sum", lambda s: (~s).sum()])
    cnt.columns = ["correct", "wrong"]
    ok_all = True
    for r in sorted(cnt.index, key=lambda r: int(r[3:])):
        exp = EXPECTED.get(r)
        flag = "" if exp is None else ("  OK" if exp == (cnt.loc[r, "correct"], cnt.loc[r, "wrong"]) else "  MISMATCH (expected %d/%d)" % exp)
        ok_all &= exp is not None and exp == (cnt.loc[r, "correct"], cnt.loc[r, "wrong"])
        say(f"  {r}: CUT {cnt.loc[r, 'correct']}  WRONG_CUT {cnt.loc[r, 'wrong']}{flag}")
    say(f"  total CUT {int(cnt['correct'].sum())}  WRONG_CUT {int(cnt['wrong'].sum())}  "
        f"(expected 288 / 96: {'OK' if ok_all and cnt['correct'].sum() == 288 and cnt['wrong'].sum() == 96 else 'MISMATCH'})")
    d = meta["vote_minus_start"]
    say(f"  vote - START: median {d.median():.3f} s, min {d.min():.3f}, max {d.max():.3f} (expected ~4.25 s)")
    S = ds["sanity"]
    say("  per run: samples, duration, missing samples vs 500 Hz, first START in EEG time, max |DC offset| (uV)")
    for r, s in S.items():
        say(f"    {r}: n={s['n']}  dur={s['dur']:.1f} s  missing={s['missing']:.0f}  first START at {ds['align'][r]['first_start_rel']:.1f} s  DC<={s['dc']:.0f}")
    dt = np.concatenate([s["dt"] for s in S.values()])
    say(f"  sample interval (ms): median {np.median(dt):.3f}, share within 2.0+-0.5: {np.mean(np.abs(dt - 2) <= 0.5):.5f}, "
        f"intervals > 3 ms: {int((dt > 3).sum())}, max {dt.max():.1f}")
    f = next(iter(S.values()))["f"]
    P = np.mean([s["P"] for s in S.values()], axis=0)
    slope, aratio, apk, line = spectral_checks(f, P)
    oi = [CH.index(c) for c in OCC]
    say(f"  spectral slope 2-40 Hz: median {np.median(slope):.2f} (range {slope.min():.2f} to {slope.max():.2f}); white noise would be ~0, real EEG ~ -1 to -2")
    say("  alpha peak ratio (PSD max in 7-13 Hz / power-law fit from 3-6 and 15-20 Hz): " +
        ", ".join(f"{CH[i]} {aratio[i]:.2f}x at {apk[i]:.1f} Hz" for i in oi) + f"; median over all channels {np.median(aratio):.2f}x")
    from scipy.signal import find_peaks, medfilt
    lp = np.log10(np.median(P, axis=0))
    ex = lp - medfilt(lp, 41)
    pk, _ = find_peaks(ex, height=np.log10(5.0), distance=5)
    pk = pk[(f[pk] >= 2) & (f[pk] <= 120)]
    say("  narrow spectral peaks (median over channels, >5x local level, 2-120 Hz): " + (", ".join(f"{f[i]:.2f} Hz ({10 ** ex[i]:.0f}x)" for i in pk) or "none")
        + "; a comb of equally spaced peaks is a device/periodic artefact, not alpha (the alpha ratio above then measures that peak)")
    say(f"  line noise (peak / flanks): 50 Hz median {np.median(line[50]):.2f}x, 60 Hz median {np.median(line[60]):.2f}x")
    f_sanity(S, args.out, {"slope": slope})

    say()
    say("== 1b. Alignment check (START-locked and vote-locked response per run, occipital O1/Oz/O2, 0.1-30 Hz) ==")
    A = ds["align"]
    say("  run: n_START  START peak latency (ms)  n_CUT(correct)  CUT peak latency (ms)  xcorr lag (ms)  r")
    for r in A:
        a = A[r]
        say(f"    {r}: {a['n_start']:3d}  {a['start_peak'] * 1000:6.0f}  {a['n_cut']:3d}  {a['cut_peak'] * 1000:6.0f}  {a['lag'] * 1000:+6.0f}  {a['r']:.2f}")
    sp = np.array([A[r]["start_peak"] for r in A]) * 1000
    lg = np.array([A[r]["lag"] for r in A]) * 1000
    say(f"  START peak latency range across runs: {sp.max() - sp.min():.0f} ms; xcorr lag range: {lg.max() - lg.min():.0f} ms "
        f"({'> 50 ms: runs disagree' if (sp.max() - sp.min() > 50 or lg.max() - lg.min() > 50) else '<= 50 ms: runs agree'})")
    say(f"  lag applied to vote times: " + (", ".join(f"{r} {v * 1000:+.0f} ms" for r, v in ds["applied"].items()) if args.align == "xcorr" else "none (--align none)"))
    say("  Lag is a relative fix only (relative to the pooled response of the other runs); the absolute offset is unknown.")
    f_alignment(A, np.arange(round(-0.5 * FS), round(1.5 * FS) + 1) / FS, t, args.out)

    # ---- step 2 ----
    say()
    say("== 2. Epochs ==")
    say(f"  filter 1-10 Hz zero-phase FIR (statistics, decoding), 0.1-30 Hz (figures e3/e4), linear detrend, common average reference")
    say(f"  epochs {TMIN} to {TMAX} s around the vote, baseline {BASE[0]} to {BASE[1]} s, reject peak-to-peak > {REJECT_UV:.0f} uV (1-10 Hz, any channel incl. Fp1/Fp2)")
    rej = meta.groupby("correct")["rejected"].agg(["sum", "count"])
    say(f"  rejected: correct {int(rej.loc[True, 'sum'])}/{int(rej.loc[True, 'count'])}, wrong {int(rej.loc[False, 'sum'])}/{int(rej.loc[False, 'count'])}")
    say("  rejected per run (correct / wrong): " + ", ".join(
        f"{r} {int(meta[(meta.run == r) & meta.correct].rejected.sum())}/{int(meta[(meta.run == r) & ~meta.correct].rejected.sum())}" for r in A))
    say(f"  kept: correct {int((~y).sum())}, wrong {int(y.sum())}; kept wrong per run: " +
        ", ".join(f"{r} {int(((runs == r) & y).sum())}" for r in A))
    try:
        import mne
        info = make_info()
        md = mk[["run", "trial", "label", "outcome", "correct", "A_vote", "B_vote"]].copy()
        ev = np.c_[np.arange(len(mk)), np.zeros(len(mk), int), np.where(y, 2, 1)]
        ep = mne.EpochsArray(xb.astype(np.float64) * 1e-6, info, events=ev, tmin=TMIN, event_id={"correct": 1, "wrong": 2},
                             metadata=md, baseline=None, verbose=False)
        ep.save(os.path.join(args.out, "errp-epo.fif"), overwrite=True, fmt="single", verbose=False)
        say("  saved errp-epo.fif (1-10 Hz, baselined) and epochs_meta.csv")
    except Exception as e:
        say(f"  could not save errp-epo.fif: {e!r}")
    info = make_info()

    # ---- step 3 ----
    say()
    say("== 3. ERP statistics (wrong - correct; 1-10 Hz, 100 Hz decimated for cluster tests) ==")
    f_erp(xf, t, y, args.out, rng)
    f_topo(xf, t, y, info, args.out)
    Xpost, tpost = stat_arrays(xb, t, 0.0, 1.0)
    Xpre, tpre = stat_arrays(xb_nb, t, -1.0, 0.0)
    ta = time.time()
    post = cluster_test(Xpost, y, info, args.n_perm_cluster, args.seed)
    say(f"  spatio-temporal cluster test 0-1.0 s, 32 ch, Welch-t threshold p<0.05 two-sided, {args.n_perm_cluster} permutations ({time.time() - ta:.0f} s):")
    say(f"    {len(post['clusters'])} cluster(s) formed")
    for c in post["clusters"][:5]:
        say("    " + describe_cluster(c, tpost, post["T"]))
    say("  window tests (wrong - correct, uV; Welch t; permutation p global and with labels shuffled within run):")
    wt = window_tests(xb, t, y, runs, args.n_perm_window, rng)
    for w in wt:
        say(f"    {w['name']:24s} diff={w['diff']:+.2f}  t={w['t']:+.2f}  p_welch={w['p_welch']:.4f}  p_perm={w['p_perm']:.4f}  "
            f"p_perm_within_run={w['p_perm_within_run']:.4f}  (wrong {w['n_wrong']}, correct {w['n_correct']})")
    say("    expectation if ErrP: FCz diff < 0 (200-350 ms), Cz diff > 0 (300-600 ms). Two-sided p reported.")

    # ---- step 4 ----
    say()
    say("== 4. Controls ==")
    ta = time.time()
    pre = cluster_test(Xpre, y, info, args.n_perm_cluster, args.seed)
    say(f"  pre-vote cluster test -1.0-0 s (no baseline), {args.n_perm_cluster} permutations ({time.time() - ta:.0f} s): {len(pre['clusters'])} cluster(s)")
    for c in pre["clusters"][:3]:
        say("    " + describe_cluster(c, tpre, pre["T"]))
    pw = window_tests(xb_nb, t, y, runs, args.n_perm_window, rng)[:2]
    say("  pre-vote windows (unbaselined 1-10 Hz; FCz 200-350 / Cz 300-600 ms are the post-vote windows, so use the same clock before it):")
    for nm, c, w in (("FCz -0.35..-0.20", "FCz", (-0.35, -0.2)), ("Cz -0.60..-0.30", "Cz", (-0.6, -0.3))):
        D1 = window_means(xb_nb, t, c, w)
        o, p, _ = perm_welch(D1[:, None], y, None, args.n_perm_window, rng)
        say(f"    {nm:20s} diff={D1[y].mean() - D1[~y].mean():+.2f}  t={o[0]:+.2f}  p_perm={p[0]:.4f}")

    m1 = runs != "run1"
    ta = time.time()
    ex1 = cluster_test(Xpost[m1], y[m1], info, args.n_perm_cluster, args.seed)
    say(f"  excluding run1 (wrong {int(y[m1].sum())}, correct {int((~y[m1]).sum())}): {len(ex1['clusters'])} cluster(s) ({time.time() - ta:.0f} s)")
    for c in ex1["clusters"][:3]:
        say("    " + describe_cluster(c, tpost, ex1["T"]))
    for w in window_tests(xb[m1], t, y[m1], runs[m1], args.n_perm_window, rng)[:2]:
        say(f"    {w['name']:24s} diff={w['diff']:+.2f}  t={w['t']:+.2f}  p_perm={w['p_perm']:.4f}  p_perm_within_run={w['p_perm_within_run']:.4f}")

    ta = time.time()
    flat = Xpost.reshape(len(Xpost), -1)
    tobs, pf, pfw = perm_welch(flat, y, runs, args.n_perm_cluster, rng)
    say(f"  run-stratified max-|t| test 0-1.0 s (labels shuffled within run, all time x channel points, {args.n_perm_cluster} perms, {time.time() - ta:.0f} s): "
        f"max |t|={np.abs(tobs).max():.2f}, family-wise p={pfw.min():.4f}")

    lab = mk["label"].to_numpy()
    say("  per MI label (label 0 / label 1; LABEL in the log, side mapping not verified here):")
    for l in (0, 1):
        s = lab == l
        if y[s].sum() > 3 and (~y[s]).sum() > 3:
            for w in window_tests(xb[s], t, y[s], runs[s], args.n_perm_window, rng)[:2]:
                say(f"    label {l} (wrong {int(y[s].sum())}, correct {int((~y[s]).sum())}) {w['name']:14s} diff={w['diff']:+.2f}  t={w['t']:+.2f}  p_perm={w['p_perm']:.4f}")
    say(f"  wrong rate by label: label 0 {y[lab == 0].mean():.2f}, label 1 {y[lab == 1].mean():.2f}")

    # topography check: where is the difference?
    tmap = post["T"]
    win = (tpost >= 0.15) & (tpost <= 0.7)
    sub = np.abs(tmap[win])
    ti, ci = np.unravel_index(sub.argmax(), sub.shape)
    say(f"  peak |t| in 150-700 ms: channel {CH[ci]} at {tpost[win][ti] * 1000:.0f} ms, t={tmap[win][ti, ci]:+.2f}")
    ti2, ci2 = np.unravel_index(np.abs(tmap).argmax(), tmap.shape)
    say(f"  peak |t| in 0-1000 ms: channel {CH[ci2]} at {tpost[ti2] * 1000:.0f} ms, t={tmap[ti2, ci2]:+.2f}")
    mfc = np.abs(tmap[win][:, [CH.index(c) for c in FRONTOCENTRAL]]).mean()
    moc = np.abs(tmap[win][:, [CH.index(c) for c in OCCIPITOPARIETAL]]).mean()
    mfp = np.abs(tmap[win][:, [CH.index(c) for c in ("Fp1", "Fp2")]]).mean()
    say(f"  mean |t| 150-700 ms: frontocentral {mfc:.2f}, occipito-parietal (O1,Oz,O2,P7,P8) {moc:.2f}, Fp1/Fp2 {mfp:.2f}")
    mu = lambda cs: np.abs(window_means_all(xb[y], t, (0.15, 0.7))[:, [CH.index(c) for c in cs]].mean(0) -
                           window_means_all(xb[~y], t, (0.15, 0.7))[:, [CH.index(c) for c in cs]].mean(0)).mean()
    say(f"  mean |difference| 150-700 ms (uV): frontocentral {mu(FRONTOCENTRAL):.2f}, occipito-parietal {mu(OCCIPITOPARIETAL):.2f}, Fp1/Fp2 {mu(['Fp1', 'Fp2']):.2f}")

    # sanity null
    ta = time.time()
    ys = permute_within(y, runs, np.random.default_rng(args.seed + 1))
    nul = cluster_test(Xpost, ys, info, args.n_perm_cluster, args.seed)
    say(f"  sanity null: cluster test with labels shuffled within run: {len(nul['clusters'])} cluster(s), best p="
        f"{nul['clusters'][0]['p'] if nul['clusters'] else float('nan'):.4f} ({time.time() - ta:.0f} s)")
    nrng = np.random.default_rng(args.seed + 2)
    hits, curves = np.zeros(2), []
    from scipy import stats as sps
    D2 = np.stack([window_means(xb, t, "FCz", (0.2, 0.35)), window_means(xb, t, "Cz", (0.3, 0.6))], 1)
    n_sh = 200
    for i in range(n_sh):
        s = permute_within(y, runs, nrng)
        for k in range(2):
            hits[k] += sps.ttest_ind(D2[s, k], D2[~s, k], equal_var=False).pvalue < 0.05
        if i < 20:
            curves.append(xb[s][:, CH.index("FCz")].mean(0) - xb[~s][:, CH.index("FCz")].mean(0))
    say(f"  sanity null: share of {n_sh} within-run label shuffles with Welch p<0.05 in the primary windows: FCz {hits[0] / n_sh:.3f}, Cz {hits[1] / n_sh:.3f} (expect ~0.05)")
    f_cluster(post, pre, tpost, tpre, args.out)
    f_controls(xb, xb_nb, t, mk, y, curves, args.out)

    # ---- step 5 ----
    say()
    say("== 5. Single-trial detection (leave-one-run-out LDA, wrong = positive class) ==")
    say(f"  features: mean amplitude in 50 ms bins, 150-700 ms (11 bins); (7ch) {', '.join(FC7)}; (xdawn) {args.n_comp} components per class, fitted inside each fold")
    say("  shrinkage LDA (lsqr, auto), equal class priors (sklearn LDA has no class_weight); pre-vote control = bins -550..0 ms, no baseline")
    say(f"  folds: {len(loro_folds(runs))}; AUC = mean over folds; permutation p from {args.n_perm_decode} within-run label permutations")
    fns = make_feat_fns(xb, xb_nb, t, y.astype(int), args.n_comp)
    D = {}
    yi = y.astype(int)
    for key, fn in fns.items():
        ta = time.time()
        D[key] = decode_with_null(fn, yi, runs, args.n_perm_decode, rng)
        r = D[key]
        say(f"  {key:11s} mean fold AUC {r['mean_auc']:.3f}  pooled AUC {r['pooled_auc']:.3f}  balanced acc {r['bacc']:.3f} (TPR {r['tpr']:.3f}, TNR {r['tnr']:.3f})  "
            f"permutation p={r['p']:.4f}  null mean {r['null'].mean():.3f} (95th pct {np.percentile(r['null'], 95):.3f})  [{time.time() - ta:.0f} s]")
        say("     fold AUCs by held-out run: " + ", ".join(f"{rn} {a:.2f}" for rn, a in zip(sorted(set(runs)), r['aucs'])))
    f_decoding(D, yi, runs, args.out)

    # ---- verdict ----
    say()
    say("== Reading (rule-based; see CONFOUNDS above) ==")
    best = post["clusters"][0] if post["clusters"] else None
    sig = best is not None and best["p"] < 0.05
    say(f"  post-vote cluster test: {'significant cluster' if sig else 'no significant cluster'}"
        + (f" (p={best['p']:.4f}, {tpost[best['tidx'][0]] * 1000:.0f}-{tpost[best['tidx'][-1]] * 1000:.0f} ms)" if best else ""))
    fcz, cz = wt[0], wt[1]
    say(f"  FCz 200-350 ms: diff {fcz['diff']:+.2f} uV, p_perm={fcz['p_perm']:.4f} ({'negative as an ErrP predicts' if fcz['diff'] < 0 else 'positive, opposite to an ErrP'}); "
        f"Cz 300-600 ms: diff {cz['diff']:+.2f} uV, p_perm={cz['p_perm']:.4f} ({'positive as an ErrP predicts' if cz['diff'] > 0 else 'negative, opposite to an ErrP'})")
    say(f"  location: mean |t| frontocentral {mfc:.2f} vs occipito-parietal {moc:.2f} -> {'fronto-central dominates' if mfc > moc else 'occipito-parietal dominates (visual/motion-like, not ErrP-specific)'}")
    prec = pre["clusters"][0]["p"] if pre["clusters"] else 1.0
    say(f"  pre-vote: best cluster p={prec:.4f} -> {'already differs before the vote (pre-existing state, not a response to the error)' if prec < 0.05 else 'no pre-vote difference detected'}")
    say(f"  decoding: 7ch AUC {D['post_7ch']['mean_auc']:.3f} (p={D['post_7ch']['p']:.3f}), xdawn AUC {D['post_xdawn']['mean_auc']:.3f} (p={D['post_xdawn']['p']:.3f}); "
        f"pre-vote control 7ch {D['pre_7ch']['mean_auc']:.3f} (p={D['pre_7ch']['p']:.3f}), xdawn {D['pre_xdawn']['mean_auc']:.3f} (p={D['pre_xdawn']['p']:.3f})")
    say("  Whatever is found cannot be attributed to an error response alone: wrong and correct votes also differ in visual feedback and frequency.")
    say(f"total runtime {time.time() - t0:.0f} s")
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return ds


if __name__ == "__main__":
    main()
