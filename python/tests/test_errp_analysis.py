"""
Tests for python/analysis/errp_analysis.py on synthetic data (fake 32-ch CSV with the real 11-line header + log).
Injected: a Gaussian negativity at FCz 250 ms after WRONG_CUT only; optional START-locked response to test the alignment.

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_errp_analysis.py   (VRBCIenv has no pytest; pytest also works)
"""
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import tempfile
import functools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import errp_analysis as ea

FS = 500
N_TRIALS = 24
PERIOD = 3.5
VOTE_AFTER_START = 1.2
HEADER = ['"Simulating Cygnus version: 0.28.0.7,File version: 2021.11"', "Operative system: Windows",
          "Record datetime: 2026-10-15 15:18:41.500000", "Simulating Device ID: STEEG_DG329018", "Device verison: ",
          "Device bandwidth: DC to 131 Hz", "Device sampling rate: 500 samples/second", "Data type / unit: EEG / micro-volt (uV)",
          "Bandpass filter: None", "Notch filter: None"]


def rec_epoch_true():
    tz = datetime.timezone(datetime.timedelta(seconds=time.localtime().tm_gmtoff))
    return datetime.datetime(2026, 10, 15, 15, 18, 41, 500000, tzinfo=tz).timestamp()


def bump(t, centre, sigma=0.04):
    return np.exp(-0.5 * ((t - centre) / sigma) ** 2)


def make_session(root, inject=True, noise=8.0, offsets=None, start_response=False, seed=0, n_runs=3):
    """offsets[run]: seconds by which the EEG response comes later than the log implies."""
    rng = np.random.default_rng(seed)
    rec = rec_epoch_true()
    offsets = offsets or {}
    chs = ea.CH
    pattern = rng.normal(size=len(chs))
    pattern -= pattern.mean()
    for r in range(1, n_runs + 1):
        name = f"run{r}"
        d = os.path.join(root, name)
        os.makedirs(d)
        off = offsets.get(name, 0.0)
        n = int((2 + N_TRIALS * PERIOD + 3) * FS)
        ts = np.arange(n) / FS
        x = rng.normal(0, noise, size=(n, len(chs))) + rng.uniform(-3000, 3000, size=len(chs))
        x += 200 * np.sin(2 * np.pi * 0.05 * ts)[:, None]
        wrong = rng.permutation(N_TRIALS) < N_TRIALS // 3
        lines = []
        for k in range(N_TRIALS):
            start_rel = 2.0 + k * PERIOD
            vote_rel = round(start_rel + VOTE_AFTER_START, 3)
            lab = k % 2
            lines.append(f"Trial {k} START: {rec + start_rel:.3f} LABEL: {lab} MODE: Adaptive")
            lines.append(f"Trial {k} {'WRONG_CUT' if wrong[k] else 'CUT'}: {rec + vote_rel:.3f}")
            lines.append(f"Trial {k} END: {rec + vote_rel + 0.7:.3f} LABEL: {lab} MODEL_PREDICTION: A=1 B=0 LIVE=A AT: {rec + vote_rel:.3f}")
            if inject and wrong[k]:
                x[:, chs.index("FCz")] -= 15 * bump(ts, vote_rel + off + 0.25)
            if start_response:
                x += 25 * bump(ts, start_rel + off + 0.15, 0.05)[:, None] * pattern[None, :]
        df = pd.DataFrame(np.c_[ts, np.full(n, 999), x], columns=["Timestamp", "Serial Number"] + chs)
        with open(os.path.join(d, "eeg_record_20261015_151841.csv"), "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(HEADER) + "\n")
            df.to_csv(f, index=False, float_format="%.4f")
        with open(os.path.join(d, "log_20261015_151841.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.join(root, "run9"))  # empty log, like run8 in the real session
    open(os.path.join(root, "run9", "log_x.txt"), "w").close()
    return root


def tmp():
    return tempfile.mkdtemp(prefix="errp_test_")


def approx(a, b, tol):
    return abs(a - b) <= tol


@functools.lru_cache(maxsize=None)
def get_ds(kind):
    root = make_session(tmp(), inject=kind == "inj", seed=1 if kind == "inj" else 2)
    return ea.build_dataset(root, None, "none", log=lambda s: None)


def prep(ds):
    keep = ds["keep"]
    mk = ds["meta"][keep].reset_index(drop=True)
    y = (~mk["correct"]).to_numpy()
    return ds["xb"], ds["t"], y, mk["run"].to_numpy()


def test_parse_log_handles_restarting_trial_numbers():
    path = os.path.join(tmp(), "log.txt")
    with open(path, "w") as f:
        f.write("Trial 0 START: 100.0 LABEL: 1 MODE: Adaptive\nTrial 0 CUT: 104.0\n"
                "Trial 0 END: 105.0 LABEL: 1 MODEL_PREDICTION: A=1 B=0 LIVE=A AT: 104.0\n"
                "Trial 0 START: 200.0 LABEL: 0 MODE: Adaptive\nTrial 0 WRONG_CUT: 204.0\n")
    tr = ea.parse_log(path)
    assert [t["outcome"] for t in tr] == ["CUT", "WRONG_CUT"] and tr[0]["A"] == "1" and tr[1]["start"] == 200.0


def test_empty_log_run_skipped():
    inj = get_ds("inj")
    assert inj["skipped"] == ["run9"]
    assert set(inj["meta"]["run"]) == {"run1", "run2", "run3"}
    assert inj["meta"]["outcome"].isin(["CUT", "WRONG_CUT"]).all()


def test_alignment_within_one_sample():
    root = make_session(tmp(), inject=True, noise=0.0, seed=3)
    ds = ea.build_dataset(root, None, "none", log=lambda s: None)
    keep = ds["keep"]
    y = (~ds["meta"]["correct"].to_numpy())[keep]
    xf, t = ds["xf"], ds["t"]
    fcz = ea.CH.index("FCz")
    d = xf[y][:, fcz].mean(0) - xf[~y][:, fcz].mean(0)
    assert abs(t[np.argmin(d)] - 0.25) <= 1.0 / FS + 1e-9


def test_extract_epochs_nearest_sample():
    n = 5000
    ts = np.arange(n) / FS
    y = np.zeros((2, n))
    y[0, 2000] = 1.0
    for jitter in (-0.0009, 0.0, 0.0007):
        ep, valid = ea.extract_epochs(y, ts, [ts[2000] + jitter], FS, -0.5, 1.0)
        assert valid.all()
        t = np.arange(round(-0.5 * FS), round(1.0 * FS) + 1) / FS
        assert abs(t[np.argmax(ep[0, 0])]) <= 1.0 / FS + 1e-9
    ep, valid = ea.extract_epochs(y, ts, [0.1, ts[-1] + 5], FS, -0.5, 1.0)
    assert not valid.any()


def test_window_and_cluster_detect_injected_negativity():
    inj = get_ds("inj")
    xb, t, y, runs = prep(inj)
    rows = ea.window_tests(xb, t, y, runs, 500, np.random.default_rng(0))
    fcz = rows[0]
    assert fcz["diff"] < -3 and fcz["p_perm"] < 0.01 and fcz["p_perm_within_run"] < 0.01
    X, tv = ea.stat_arrays(xb, t, 0.0, 1.0)
    res = ea.cluster_test(X, y, ea.make_info(), 200, 0)
    best = res["clusters"][0]
    assert best["p"] < 0.05 and best["sign"] < 0
    assert "FCz" in [ea.CH[i] for i in best["cidx"]]
    assert 0.15 <= tv[best["tidx"]].mean() <= 0.4


def test_no_injection_is_not_significant():
    null = get_ds("null")
    xb, t, y, runs = prep(null)
    rows = ea.window_tests(xb, t, y, runs, 500, np.random.default_rng(0))
    assert rows[0]["p_perm"] > 0.05 and rows[1]["p_perm"] > 0.05
    X, _ = ea.stat_arrays(xb, t, 0.0, 1.0)
    res = ea.cluster_test(X, y, ea.make_info(), 200, 0)
    assert not res["clusters"] or res["clusters"][0]["p"] > 0.05


def test_loro_folds_never_mix_runs():
    runs = np.array(["run1"] * 5 + ["run2"] * 4 + ["run3"] * 6)
    folds = ea.loro_folds(runs)
    assert len(folds) == 3
    for tr, te in folds:
        assert len(set(runs[te])) == 1
        assert not set(runs[tr]) & set(runs[te])
        assert len(tr) + len(te) == len(runs) and not set(tr) & set(te)


def test_permute_within_keeps_run_label_counts():
    rng = np.random.default_rng(0)
    runs = np.array(["a"] * 10 + ["b"] * 10)
    y = np.array([1] * 3 + [0] * 7 + [1] * 5 + [0] * 5)
    for _ in range(5):
        p = ea.permute_within(y, runs, rng)
        assert p[:10].sum() == 3 and p[10:].sum() == 5


def test_decoding_detects_injection_and_runs_both_feature_sets():
    for ds, expect in ((get_ds("inj"), True), (get_ds("null"), False)):
        xb, t, y, runs = prep(ds)
        fns = ea.make_feat_fns(xb, ds["xb_nobase"], t, y.astype(int), 2)
        r7 = ea.decode_with_null(fns["post_7ch"], y.astype(int), runs, 5, np.random.default_rng(0))
        rx = ea.decode_with_null(fns["post_xdawn"], y.astype(int), runs, 5, np.random.default_rng(0))
        pre = ea.decode_with_null(fns["pre_7ch"], y.astype(int), runs, 5, np.random.default_rng(0))
        if expect:
            assert r7["mean_auc"] > 0.75 and rx["mean_auc"] > 0.7
            assert r7["mean_auc"] - pre["mean_auc"] > 0.2
        else:
            assert r7["mean_auc"] < 0.8 and rx["mean_auc"] < 0.8
        assert len(r7["aucs"]) == 3


def test_xcorr_recovers_relative_lag():
    root = make_session(tmp(), inject=False, noise=4.0, offsets={"run2": 0.12, "run3": -0.08}, start_response=True, seed=4)
    ds = ea.build_dataset(root, None, "none", log=lambda s: None)
    lag = {r: ds["align"][r]["lag"] for r in ds["align"]}
    assert approx(lag["run2"] - lag["run3"], 0.20, 0.02), lag
    assert lag["run2"] > lag["run1"] > lag["run3"]
    assert approx(lag["run1"] - lag["run3"], 0.08, 0.02), lag
    ds2 = ea.build_dataset(root, None, "xcorr", log=lambda s: None)
    assert approx(ds2["applied"]["run2"], lag["run2"], 1e-9)
    assert approx(ds2["meta"].query("run == 'run2'")["lag_applied"].iloc[0], lag["run2"], 1e-9)


def test_end_to_end_small():
    base = tmp()
    root = make_session(os.path.join(base, "s"), inject=True, seed=5, start_response=True)
    out = os.path.join(base, "out")
    ea.main(["--session", root, "--out", out, "--n_perm_cluster", "20", "--n_perm_decode", "5", "--n_perm_window", "50"])
    text = open(os.path.join(out, "summary.txt"), encoding="utf-8").read()
    for s in ("CONFOUNDS", "== 1b. Alignment", "== 5. Single-trial", "MISMATCH"):
        assert s in text
    for f in ("e1_sanity", "e2_alignment", "e3_erp", "e4_topomaps", "e5_cluster", "e6_controls", "e7_decoding"):
        assert os.path.exists(os.path.join(out, f + ".png"))
    assert os.path.exists(os.path.join(out, "epochs_meta.csv")) and os.path.exists(os.path.join(out, "errp-epo.fif"))


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
