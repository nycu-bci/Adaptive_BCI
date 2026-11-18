"""
Tests for F2 (python/analysis/finetune_eval/f2_counterfactual.py, f2_analyze.py): gate rule, decision list (run resets, parent index, test window),
per-decision effect, pooling, sign test, buffer walk vs the F1b stream. Real-data tests are skipped when the recordings are not on this machine.

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_f2.py   (VRBCIenv has no pytest; pytest also works)
"""
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "finetune_eval"))
import f2_counterfactual as CF
import f2_analyze as A
import f2_inuse as I
R, F = CF.R, CF.F


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def fake_session(runs):
    """runs: list of lists of (n_trials, correct_A, ft_A); one chain checkpoint per fired block."""
    trials, rs, chain = [], [], [{"path": "cA_000.pth"}]
    seq = 0
    for ri, blocks in enumerate(runs):
        rtr, rows = [], []
        for bi, (n, c, ft) in enumerate(blocks):
            for _ in range(n):
                t = {"seq": seq}
                rtr.append(t)
                trials.append(t)
                seq += 1
            rows.append({"block": bi + 1, "n_trials": n, "correct_A": c, "ft_A": str(ft)})
            if ft:
                chain.append({"path": f"cA_{len(chain):03d}.pth"})
        rs.append(types.SimpleNamespace(run=ri + 1, test_only=False, blocks=rows, trials=rtr))
    return types.SimpleNamespace(sub="subX", runs=rs, trials=trials, chain={"A": chain})


def test_gate_rule():
    assert CF.gate(1, 2, 4)
    assert not CF.gate(0, 2, 4)
    assert not CF.gate(1, 3, 4)
    assert CF.gate(1, 2, 3, threshold=0.75)
    assert not CF.gate(1, 0, 0)


def test_windows_at_the_session_end():
    assert list(CF.test_seqs(10, 100)) == list(range(11, 59))
    assert list(CF.test_seqs(95, 100)) == [96, 97, 98, 99]
    assert len(CF.test_seqs(99, 100)) == 0
    assert list(CF.prev_seqs(5)) == list(range(0, 6))
    assert list(CF.prev_seqs(40)) == list(range(17, 41))


def test_decisions_reset_per_run_and_track_the_parent():
    sess = fake_session([[(4, 2, 1), (4, 4, 0), (4, 2, 1)], [(4, 4, 0), (4, 1, 1)]])
    d = CF.decisions(sess, "A")
    assert [x["d"] for x in d] == [1, 2, 3, 4, 5]
    assert [x["run_scored"] for x in d] == [4, 8, 12, 4, 8]
    assert [x["run_correct"] for x in d] == [2, 6, 8, 4, 5]
    assert not CF.gate(1, 9, 12)  # 0.75 exactly: not below the threshold
    assert [x["gate"] for x in d] == [1, 0, 1, 0, 1]
    assert [x["gate"] for x in d] == [x["fired"] for x in d]
    assert [x["parent_index"] for x in d] == [0, 1, 1, 2, 2]
    assert [x["parent"] for x in d] == ["cA_000", "cA_001", "cA_001", "cA_002", "cA_002"]
    assert [x["last_seq"] for x in d] == [3, 7, 11, 15, 19]
    assert [x["n_test"] for x in d] == [16, 12, 8, 4, 0]


def test_decision_effect():
    lab = np.array([0, 1, 0, 1, 0, 1])
    par = np.array([0, 0, 0, np.nan, 1, 1])
    ch = np.array([[0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 0, np.nan]])
    e = A.decision_effect(lab, par, ch)
    assert e["n"] == 4
    assert np.isclose(e["parent_acc"], 2 / 4)
    assert np.isclose(e["child_acc"], (3 + 3 + 4) / 12)
    assert np.isclose(e["effect"], 10 / 12 - 0.5)
    assert np.allclose(e["seed_effects"], [0.25, 0.25, 0.5])
    assert (e["b"], e["c"]) == (0, 2)
    assert A.decision_effect(lab, np.full(6, np.nan), ch)["n"] == 0


def test_pooling_and_sign_test():
    assert np.isclose(A.pooled([0.1, -0.1], [30, 10]), 0.05)
    assert np.isnan(A.pooled([0.1], [0]))
    assert np.isclose(A.sign_p(0, 6), 0.03125)
    assert A.sign_p(0, 0) == 1.0
    rng = np.random.default_rng(0)
    lo, hi = A.boot_diff_ci([0.1] * 5, [48] * 5, [0.0] * 5, [48] * 5, rng)
    assert np.isclose(lo, 0.1) and np.isclose(hi, 0.1)


def test_inuse_window_runs_to_the_next_fired_block():
    last, fired = np.array([3, 7, 11, 15]), np.array([1, 0, 1, 0])  # fire at block 1, skip 2, fire 3, skip 4
    assert list(I.inuse_seqs(last, fired, 0, 20)) == list(range(4, 12))  # blocks 2 and 3
    assert list(I.inuse_seqs(last, fired, 1, 20)) == list(range(8, 12))  # block 3
    assert list(I.inuse_seqs(last, fired, 2, 20)) == list(range(12, 20))  # no later fire: to the session end
    assert list(I.inuse_seqs(last, fired, 3, 20)) == list(range(16, 20))
    assert list(I.inuse_seqs(np.array([3, 103]), np.array([0, 1]), 0, 200)) == list(range(4, 52))  # capped at 48
    assert len(I.inuse_seqs(last, fired, 3, 16)) == 0


def test_window_without_the_selecting_block():
    last, fired = np.array([3, 7, 11, 15, 19]), np.array([1, 0, 0, 1, 0])
    w = I.inuse_seqs(last, fired, 0, 24)
    assert list(I.without_selecting_block(w, last, fired, 0)) == list(range(4, 12))  # blocks 2-3; block 4 (next fire) dropped
    w = I.inuse_seqs(last, fired, 2, 24)
    assert len(I.without_selecting_block(w, last, fired, 2)) == 0  # window was only the selecting block
    w = I.inuse_seqs(last, fired, 3, 24)
    assert list(I.without_selecting_block(w, last, fired, 3)) == list(range(16, 24))  # no later fire: kept whole
    last2, fired2 = np.array([3, 103]), np.array([0, 1])
    w = I.inuse_seqs(last2, fired2, 0, 200)
    assert list(I.without_selecting_block(w, last2, fired2, 0)) == list(range(4, 52))  # capped before the fire: kept whole


def test_direction_needs_size_and_seed_agreement():
    assert I.direction(1.0, [0.25, 0.25, 0.25, 0.25, -0.25]) == 1
    assert I.direction(0.8, [0.25] * 5) == 0  # under 1 trial
    assert I.direction(1.0, [0.25, 0.25, 0.25, -0.25, -0.25]) == 0  # only 3 seeds agree
    assert I.direction(-2.0, [-0.5, -0.5, -0.5, -0.5, 0.0]) == -1
    assert I.LABELS[1][-1] == "wrong fire" and I.LABELS[0][1] == "missed fire"


def real_session(sub):
    if not os.path.isdir(F.session_dir(sub)):
        skip(f"{sub} recordings not on this machine")
    return F.Session(sub, load_eeg=False)


def test_real_decisions_reproduce_the_logged_gate():
    want = {("sub01", "A"): (85, 31), ("sub01", "B"): (85, 20), ("sub05", "A"): (12, 0), ("sub05", "B"): (12, 10)}
    for sub in ("sub01", "sub05"):
        sess = real_session(sub)
        for m in "AB":
            d = CF.decisions(sess, m)
            assert (len(d), sum(x["fired"] for x in d)) == want[(sub, m)], (sub, m)
            assert all(x["gate"] == x["fired"] for x in d), (sub, m)
            assert d[-1]["parent_index"] + d[-1]["fired"] == len(sess.chain[m]) - 1


def test_real_buffer_walk_matches_the_f1b_stream_at_fired_blocks():
    sess = real_session("sub05")
    if not os.path.isdir(R.CACHE):
        skip("corrected-trial cache missing (run f1b_variants.py --prep)")
    ours = {dec["d"]: (len(tr), len(va), gen.get_state()) for dec, tr, va, gen, _, _ in CF.events(sess, "B", lambda d: True)}
    decs = {x["block"]: x["d"] for x in CF.decisions(sess, "B")}
    n = 0
    for ev in R.Stream(sess, "B", "corr", ("v0",)).events():
        d = decs[ev["blk"].number]
        assert ours[d][:2] == (len(ev["v0"]["train"]), len(ev["v0"]["val"]))
        assert bool((ours[d][2] == ev["gen0"].get_state()).all())
        n += 1
    assert n == 10


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
