"""
Tests for python/analysis/finetune_eval (chain / finetune list, F1 windows, in-sample flag, live tag, test-only run loading, live path).
Real-data tests are skipped when the recordings are not on this machine.

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_finetune_eval.py   (VRBCIenv has no pytest; pytest also works)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "finetune_eval"))
import ft_common as F
import f1_analyze as A
import trial_table as T
C = F.C


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def row(block, ft_a, ck_a, ft_b, ck_b, swap):
    return {"block": block, "n_trials": 4, "ft_A": str(ft_a), "ft_B": str(ft_b), "ckpt_A": ck_a, "ckpt_B": ck_b, "swap_time": swap}


def synthetic_blocks():
    """Run 1: A finetunes in blocks 1, 3; block 2 repeats cA_001 (no finetune, no swap). B finetunes in 1, 2 and repeats in 3. Run 2 continues."""
    r1 = [row(1, 1, r"C:\x\cA_001.pth", 1, r"C:\x\cB_001.pth", 100.0),
          row(2, 0, r"C:\x\cA_001.pth", 1, r"C:\x\cB_002.pth", 200.0),
          row(3, 1, r"C:\x\cA_002.pth", 0, r"C:\x\cB_002.pth", 300.0)]
    r2 = [row(4, 0, r"C:\x\cA_002.pth", 0, r"C:\x\cB_002.pth", None),
          row(5, 1, r"C:\x\cA_003.pth", 0, r"C:\x\cB_002.pth", 500.0)]
    return [r1, r2]


def test_finetune_list_parent_child_with_repeated_checkpoints():
    blocks = synthetic_blocks()
    a = F.build_chain(blocks, "A", "base_A.pth")
    assert [F.ckpt_name(c["path"]) for c in a] == ["base_A", "cA_001", "cA_002", "cA_003"]
    assert [c["time"] for c in a] == [None, 100.0, 300.0, 500.0]
    fa = F.finetune_list(a, "A")
    assert [(f["parent"], f["child"], f["swap_time"]) for f in fa] == [("base_A", "cA_001", 100.0), ("cA_001", "cA_002", 300.0), ("cA_002", "cA_003", 500.0)]
    assert [(f["parent_index"], f["child_index"]) for f in fa] == [(0, 1), (1, 2), (2, 3)]
    b = F.build_chain(blocks, "B", "base_B.pth")
    fb = F.finetune_list(b, "B")
    assert [(f["parent"], f["child"], f["swap_time"]) for f in fb] == [("base_B", "cB_001", 100.0), ("cB_001", "cB_002", 200.0)]
    assert F.finetune_list(F.build_chain([[row(1, 0, "x/cA_000.pth", 0, "x/cB_000.pth", None)]], "A", "b.pth"), "A") == []


def test_chain_rejects_inconsistent_rows():
    for bad in (row(1, 1, r"C:\x\cA_001.pth", 0, "b", None), ):
        try:
            F.build_chain([[bad]], "A", "base.pth")
        except ValueError:
            continue
        raise AssertionError("ft=1 without swap_time accepted")
    try:
        F.build_chain([[row(1, 1, "x/cA_001.pth", 0, "b", 5.0), row(2, 1, "x/cA_001.pth", 0, "b", 6.0)]], "A", "base.pth")
    except ValueError:
        return
    raise AssertionError("repeated child accepted")


def test_live_index_and_in_sample():
    chain = F.build_chain(synthetic_blocks(), "A", "base_A.pth")
    assert [F.live_index(chain, t) for t in (50, 99.9, 100.0, 299.9, 300.0, 1000)] == [0, 0, 1, 1, 2, 3]
    assert F.in_sample(50, 100.0) and F.in_sample(99.99, 100.0) and not F.in_sample(100.0, 100.0) and not F.in_sample(150, 100.0)
    assert not F.in_sample(-1e9, None)  # base checkpoint: never in-sample


def test_w1_w2_windows():
    ats = np.arange(1, 201) * 10.0  # trials at 10, 20, ..., 2000
    w1, w2 = F.windows(ats, swap_time=105.0, next_swap_time=205.0)
    assert ats[w1].tolist() == [110.0, 120, 130, 140, 150, 160, 170, 180, 190, 200] and len(w2) == 48 and ats[w2][0] == 110.0 and ats[w2][-1] == 580.0
    w1, w2 = F.windows(ats, swap_time=110.0, next_swap_time=None)  # a trial exactly at the swap is not a test trial (AT > swap)
    assert ats[w1][0] == 120.0 and len(w1) == 189 and len(w2) == 48
    w1, w2 = F.windows(ats, swap_time=1950.0, next_swap_time=None)  # end of session: fewer than 48
    assert ats[w1].tolist() == [1960.0, 1970, 1980, 1990, 2000] and len(w2) == 5
    w1, _ = F.windows(ats, swap_time=105.0, next_swap_time=200.0)  # next swap at a trial time: that trial (AT <= next swap) still belongs to W1
    assert ats[w1][-1] == 200.0
    assert len(F.windows(ats, 5000.0, None)[0]) == 0


def test_pre_trials():
    ats = np.arange(1, 31) * 10.0
    assert ats[F.pre_trials(ats, 205.0, None)].tolist() == [90.0 + 10 * i for i in range(12)]
    assert ats[F.pre_trials(ats, 205.0, 150.0)].tolist() == [160.0, 170, 180, 190, 200]


def test_live_tag():
    assert F.live_tag(["A", "B", "A", "B"], "A").tolist() == [True, False, True, False]
    assert F.live_tag(["A", "B", "A", "B"], "B").tolist() == [False, True, False, True]


def test_window_stats_discordant_counts_and_sign_test():
    ok_p = np.array([1, 1, 0, 0, 1, np.nan])
    ok_c = np.array([1, 0, 1, 1, 1, 1])
    st = A.window_stats(ok_p, ok_c, np.array([.9, .8, .4, .3, .7, .5]), np.array([.9, .4, .6, .7, .8, .9]))
    assert (st["n"], st["b"], st["c"]) == (5, 1, 2) and abs(st["acc_p"] - 0.6) < 1e-12 and abs(st["acc_c"] - 0.8) < 1e-12
    assert abs(st["mdiff"] - np.mean([0, -.4, .2, .4, .1])) < 1e-12
    assert A.sign_test(0, 10) == 2 * 0.5 ** 10 and A.sign_test(0, 0) == 1.0 and A.sign_test(5, 5) == 1.0


def test_pooled_and_bootstrap_are_trial_weighted():
    recs = [{"session": "s1", "w1_all_n": 10, "w1_all_sum_p": 5.0, "w1_all_sum_c": 7.0, "w1_all_b": 1, "w1_all_c": 3},
            {"session": "s2", "w1_all_n": 30, "w1_all_sum_p": 15.0, "w1_all_sum_c": 15.0, "w1_all_b": 2, "w1_all_c": 2}]
    p = A.pooled(recs, "w1", "all")
    assert p["n"] == 40 and abs(p["delta"] - 2 / 40) < 1e-12 and (p["b"], p["c"]) == (3, 5)
    lo, hi = A.boot_ci(recs, "w1", "all", 300, np.random.default_rng(0))
    assert lo <= 0.2 and hi >= 0.0 and lo < hi


def test_test_only_run_loading_and_session_chain_sub05():
    if not os.path.isdir(os.path.join(F.session_dir("sub05"), "run2")):
        skip("sub05 not found")
    s = F.Session("sub05", load_eeg=False)
    r2 = s.runs[1]
    assert r2.test_only and not s.runs[0].test_only
    assert len(r2.trials) == 48 and sum(b["n_trials"] for b in r2.blocks) == 48 and len(r2.blocks) == 12
    assert all(b["swap_time"] is None and b["ft_A"] == "0" and b["ft_B"] == "0" for b in r2.blocks)
    assert all(t["run"] == 2 and t["block"] == (i // 4) for i, t in enumerate(r2.trials))
    assert len(s.chain["A"]) == 1 and len(s.chain["B"]) == 11 and len(s.finetunes) == 10
    assert [t["seq"] for t in s.trials] == list(range(96))
    assert F.live_index(s.chain["B"], s.ats[-1]) == 10  # run 2 keeps the checkpoints of the end of run 1


def test_sub01_finetune_counts_and_timeline_matches_cut_compare():
    if not os.path.isdir(os.path.join(F.session_dir("sub01"), "run7")):
        skip("sub01 not found")
    s = F.Session("sub01", load_eeg=False)
    assert (len(s.chain["A"]) - 1, len(s.chain["B"]) - 1) == (31, 20) and len(s.trials) == 384
    for m in "AB":
        tl = C.checkpoint_timeline(m, s.runs)
        for t in s.trials:
            assert F.ckpt_name(C.ckpt_at(tl, t["at"])) == F.ckpt_name(s.chain[m][F.live_index(s.chain[m], t["at"])]["path"])


def test_live_path_sub01_run7_reproduces_logged_votes():
    run_dir = os.path.join(F.session_dir("sub01"), "run7")
    if not os.path.isdir(run_dir) or not os.path.exists(F.SUB01_LAGS):
        skip("sub01 run7 not found")
    s = F.Session("sub01", load_eeg=False)
    lags = s.load_lags()
    run = s.runs[6]
    run.ts, run.eeg = F.cached_eeg(os.path.join(run_dir, [f for f in os.listdir(run_dir) if f.startswith("eeg_record_")][0]), run.name, os.path.join(F.CACHE_ROOT, "sub01"))
    n = agree = 0
    for m in "AB":
        chain = s.chain[m]
        models = {}
        for ti, t in enumerate(run.trials):
            k = F.live_index(chain, t["at"])
            if k not in models:
                models[k] = C.load_model(F.local_ckpt("sub01", chain[k]["path"]))
            model, pos = models[k]
            P = run.preds[m]
            rows = C.last_before(P[:, 0], t["at"])
            ends = run.window_ends(P[rows, 0], lags[run.name])
            pred, _ = C.predict_ends(model, pos, run.eeg, ends)
            vote = C.unity_vote(pred)
            n += 1
            agree += int(vote == t["logged"][m])
    assert n == 96 and agree / n >= 0.97, (agree, n)
    print(f"      live path run7: {agree}/{n} votes equal the END-line votes")


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
