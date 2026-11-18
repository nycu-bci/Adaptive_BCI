"""
Tests for python/analysis/cut_compare (offline predictor + 0-4 s vs vote-ending cut comparison).
Real-data tests are skipped when the sub01/ses1 recordings are not on this machine.

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_cut_compare.py   (VRBCIenv has no pytest; pytest also works)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "cut_compare"))
import common as C
import step1_analyze as S1
from main.EEG.EEG_Train import prepare_calibration_dataset
import main.Utils.trial_scoring as trial_scoring

RUN2_DIR = C.run_dir(2)


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def synthetic_run(n=60000, rec_epoch=1.79e9):
    """Run-like object whose sample value is its own file time, so any cut shows which times it holds."""
    r = object.__new__(C.Run)
    r.ts = np.round(np.arange(n) / C.FS, 3)
    r.eeg = np.tile(r.ts[:, None], (1, 32))
    r.rec_epoch = rec_epoch
    return r


def test_unity_vote_replica_equals_trial_scoring():
    rng = np.random.default_rng(0)
    for n in [0, 1, 2, 3, 5, 9, 17, 33, 99, 100, 101, 250]:
        for _ in range(60):
            preds = list(rng.integers(0, 2, n))
            assert C.unity_vote(preds) == trial_scoring.unity_vote(preds), (n, preds)
    for p in (0.1, 0.5, 0.9):  # biased inputs so the count is often close to half
        for _ in range(200):
            preds = list((rng.random(rng.integers(2, 130)) < p).astype(int))
            assert C.unity_vote(preds) == trial_scoring.unity_vote(preds)
    assert C.unity_vote([1]) is None and C.unity_vote([]) is None


def test_vote_margin_is_weighted_share():
    assert C.vote_margin([1] * 100, 1) == 1.0 and C.vote_margin([1] * 100, 0) == 0.0
    preds = [0] * 50 + [1] * 50
    w = 1 + 4 * np.arange(100) / 99
    assert abs(C.vote_margin(preds, 1) - w[50:].sum() / w.sum()) < 1e-12
    assert np.isnan(C.vote_margin([1], 1))


def test_cut2_never_includes_samples_after_at():
    r = synthetic_run()
    rng = np.random.default_rng(1)
    for lag in (0.0, 0.986, 1.082, -0.3):
        for _ in range(200):
            at = r.rec_epoch + rng.uniform(5, 100) + lag  # AT in time.time() terms
            trial = {"at": at}
            i0, i1 = r.cut2_indices(trial, lag)
            seg = r.cut2(trial, lag)
            assert len(seg) == C.SEG == i1 - i0
            limit = at - r.rec_epoch - lag  # file time of the vote
            assert seg[:, 0].max() < limit, (lag, seg[-1, 0], limit)
            assert r.ts[i1] >= limit  # the next sample is not before the vote, so the cut is the latest one that qualifies
    # AT exactly on a sample: that sample is excluded
    t = {"at": r.rec_epoch + 10.0}
    assert r.cut2(t, 0.0)[-1, 0] == 9.998


def test_window_ends_are_whole_chunks_and_never_after_the_time():
    r = synthetic_run()
    T = r.rec_epoch + np.random.default_rng(2).uniform(5, 100, 500)
    for lag in (0.0, 0.986):
        e = r.window_ends(T, lag)
        assert (e % C.CHUNK == 0).all()
        assert (r.ts[e - 1] <= T - r.rec_epoch - lag + 1e-9).all()
        e1 = r.window_ends(T, lag, chunk=1)
        assert ((e1 - e) >= 0).all() and ((e1 - e) < C.CHUNK).all()


def test_last_before_takes_at_most_100_not_after_at():
    times = np.arange(0.0, 50.0, 0.025)
    idx = C.last_before(times, 30.0)
    assert len(idx) == 100 and times[idx].max() <= 30.0 and times[idx[-1] + 1] > 30.0
    assert len(C.last_before(times, 0.05)) == 3


def test_window_count_is_16_per_2000_sample_trial():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 22, C.SEG))
    ds = prepare_calibration_dataset([x], [np.array([0, 1, 0])], [np.array([False, True, False])])
    X, y, f = ds.tensors
    assert len(ds) == 3 * 16 and tuple(X.shape) == (48, 1, 22, 500)
    assert f.numpy().reshape(3, 16).sum(1).tolist() == [0, 16, 0] and y.argmax(1).numpy().reshape(3, 16)[1].tolist() == [1] * 16


def test_mcnemar_exact():
    ok1 = np.array([True] * 10 + [False] * 10)
    ok2 = np.array([True] * 10 + [True] * 10)
    b, c, p = S1.mcnemar(ok1, ok2)
    assert (b, c) == (0, 10) and abs(p - 2 * 0.5 ** 10) < 1e-12
    assert S1.mcnemar(ok1, ok1)[2] == 1.0


def test_cut3_starts_at_start_and_ends_before_the_vote():
    run = synthetic_run()
    lag = 0.987
    for k in range(5):
        start = run.rec_epoch + 10.0 + 7.3 * k
        trial = {"start": start, "at": start + 4.25}
        i0 = int(np.searchsorted(run.ts, start - run.rec_epoch - lag, side="left"))
        seg = run.cut3(trial, lag)
        assert len(seg) == C.SEG and np.array_equal(seg, run.eeg[i0:i0 + C.SEG])
        first_wall = run.rec_epoch + lag + run.ts[i0]
        last_wall = run.rec_epoch + lag + run.ts[i0 + C.SEG - 1]
        assert start - 1e-6 <= first_wall < start + 1.0 / C.FS + 1e-6
        assert last_wall < trial["at"]


def test_cut1_equals_pt_x_data_on_real_trials():
    import glob
    import torch
    if not os.path.isdir(RUN2_DIR) or not glob.glob(os.path.join(RUN2_DIR, "data_*.pt")) or not glob.glob(os.path.join(RUN2_DIR, "eeg_record_*.csv")):
        skip("sub01/ses1 run2 data not found")
    run = C.Run(2)
    checked, worst = 0, 0.0
    for bi, path in enumerate(run.data_files[:3]):
        d = torch.load(path, map_location="cpu", weights_only=False)
        block = [t for t in run.trials if t["block"] == bi]
        for k, tid in enumerate(d["trial_ids"]):
            t = next(t for t in block if t["idx"] == tid)
            cut = np.transpose(run.cut1(t))[C.config.channel_positions(32)]
            assert cut.shape == d["x_data"][k].shape == (32, C.SEG)
            worst = max(worst, float(np.abs(cut - d["x_data"][k]).max()))
            assert t["label"] == d["y_data"][k] and t["failed"] == bool(d["failures"][k])
            checked += 1
    assert checked >= 8
    assert worst == 0.0, worst
    print(f"      cut 1 vs x_data: {checked} trials, max abs difference {worst}")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
        except Skip as e:
            print(f"SKIP {name}: {e}")
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
