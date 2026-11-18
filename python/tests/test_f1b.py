"""
Tests for step 0b / F1b (python/analysis/finetune_eval/retrain_common.py, f1b_variants.py): buffer rebuild, per-model failure flags, V1 hold-out,
V2 decision, generator replay, trainer equivalence, corrected cut with lag 0, data_*.pt <-> block mapping. Real-data tests are skipped when the
recordings are not on this machine.

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_f1b.py   (VRBCIenv has no pytest; pytest also works)
"""
import os
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "finetune_eval"))
import retrain_common as R
import f1b_variants as V
F, C = R.F, R.C
from main.EEG.EEG_Train import arrange_by_label, EEGFineTunePipeline  # noqa: E402
from main.EEG.MI_train import ReplayBuffer, OnlineCalibrationTrainer  # noqa: E402

TMP = R.set_tmp_ckpt_dir("test_f1b")


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def synth_ds(first_id, labels, fails, per=16):
    """Dataset of len(labels) trials x per windows; X[0,0,0] of every window is a unique id."""
    n = len(labels) * per
    X = torch.zeros(n, 1, 2, 3)
    X[:, 0, 0, 0] = torch.arange(first_id, first_id + n).float()
    y = torch.nn.functional.one_hot(torch.tensor(np.repeat(labels, per)).long(), 2).float()
    f = torch.tensor(np.repeat(fails, per)).float()
    return TensorDataset(X, y, f)


def ids_of(ds):
    return [int(v) for v in ds.tensors[0][:, 0, 0, 0].tolist()]


# ---------------- buffer ----------------
def test_buffer_equals_reference_rules_on_synthetic_blocks():
    """Spec: per class FIFO train list up to 384; while it is not full every new window is also copied to val; once full the evicted oldest window
    moves to val; val capped at int(0.4*384) = 153 per class (oldest dropped). Reference is written independently of ReplayBuffer."""
    assert C.config.REPLAY_BUFFER_LIMIT == 384
    limit, vcap = 384, int(0.4 * 384)
    assert vcap == 153
    rng = np.random.default_rng(0)
    buf = ReplayBuffer()
    ref_train, ref_val = {0: [], 1: []}, {0: [], 1: []}
    nid = 0
    sizes = [48] + [4] * 40  # calibration block, then normal blocks
    for bi, n in enumerate(sizes):
        labels = np.tile([0, 1], n // 2)
        fails = rng.random(n) < 0.3
        ds = synth_ds(nid, labels, fails)
        buf.add(ds)
        for w in range(len(ds)):
            c = int(ds.tensors[1][w].argmax())
            wid = nid + w
            ref_train[c].append(wid)
            if len(ref_train[c]) > limit:
                ref_val[c].append(ref_train[c].pop(0))
            else:
                ref_val[c].append(wid)
            if len(ref_val[c]) > vcap:
                ref_val[c].pop(0)
        nid += len(ds)
        for c in (0, 1):
            assert [int(it[0][0, 0, 0]) for it in buf.train[c]] == ref_train[c], (bi, c)
            assert [int(it[0][0, 0, 0]) for it in buf.val[c]] == ref_val[c], (bi, c)
    assert len(ref_train[0]) == 384 and len(ref_val[0]) == 153
    # weights: failed window = 2.0
    ts = buf.train_set()
    assert set(ts.tensors[2].tolist()) <= {1.0, 2.0}


def test_stream_buffer_is_replay_buffer_of_all_blocks_gate_or_not():
    """Real data: the V0 snapshot at a fired block equals a ReplayBuffer fed with every block up to it (blocks without a finetune included)."""
    if not os.path.isdir(os.path.join(F.session_dir("sub01"), "run1")):
        skip("sub01 not found")
    sess = F.Session("sub01", load_eeg=False)
    pos = C.config.channel_positions(22)
    blocks = R.load_blocks(sess)
    for ev in R.Stream(sess, "A", "asran", ("v0",), assigned=lambda k: k in (3, 4)).events():
        ref = ReplayBuffer()
        for blk in blocks:
            f, _ = R.model_failures(blk, "A")
            ref.add(R.window_ds(blk.x, blk.y, f, pos))
            if blk.number == ev["blk"].number:
                break
        a, b = ev["v0"]["train"], ref.train_set()
        assert torch.equal(a.tensors[0], b.tensors[0]) and torch.equal(a.tensors[2], b.tensors[2])
        assert torch.equal(ev["v0"]["val"].tensors[0], ref.val_set().tensors[0])
        n_prev = sum(1 for bl in blocks if bl.number < ev["blk"].number)
        assert ev["n_prev_blocks"] == n_prev
        assert ev["blk"].ft("A") and ev["ft"]["child"] == F.ckpt_name(ev["blk"].row["ckpt_A"])


# ---------------- failure flags ----------------
def fake_block(order_ids, labels, live, logged, stored):
    trials = [{"idx": int(i), "label": int(l), "live": lv, "logged": lg} for i, l, lv, lg in zip(order_ids, labels, live, logged)]
    return types.SimpleNamespace(trials=trials, stored=np.asarray(stored, bool))


def test_failure_flags_live_stored_nonlive_from_logged_votes_with_arrange_by_label_order():
    ids = np.array([10, 11, 12, 13, 14, 15])  # trial (time) order
    y = np.array([1, 0, 0, 1, 1, 0])
    live = ["A", "A", "B", "B", "A", "B"]
    logged = [{"A": 1, "B": 0}, {"A": 0, "B": 1}, {"A": 1, "B": 0}, {"A": None, "B": 0}, {"A": 0, "B": 1}, {"A": 0, "B": None}]
    stored_time = np.array([False, True, False, True, True, False])  # Unity's result for the live model of each trial
    # data rows are reordered 0/1/0/1 by arrange_by_label (the same permutation for ids and stored failures)
    x = np.zeros((6, 1, 1))
    _, y_r, f_r = arrange_by_label(x, y, stored_time)
    _, _, ids_r = arrange_by_label(x, y, ids)
    assert list(ids_r) != list(ids)
    by_id = {int(i): k for k, i in enumerate(ids)}
    blk = fake_block(ids_r, y_r, [live[by_id[int(i)]] for i in ids_r], [logged[by_id[int(i)]] for i in ids_r], f_r)
    fa, _ = R.model_failures(blk, "A")
    fb, _ = R.model_failures(blk, "B")
    for row, i in enumerate(ids_r):
        k = by_id[int(i)]
        exp_a = stored_time[k] if live[k] == "A" else (logged[k]["A"] is not None and logged[k]["A"] != y[k])
        exp_b = stored_time[k] if live[k] == "B" else (logged[k]["B"] is not None and logged[k]["B"] != y[k])
        assert fa[row] == exp_a and fb[row] == exp_b, (row, i)
    # trial 13: live B, A's vote NA -> not failed for A; trial 15: B's vote NA but B not live? live is B -> stored (False)
    assert not fa[list(ids_r).index(13)]
    assert fb[list(ids_r).index(15)] == stored_time[5]
    # a row without logged votes keeps the stored flag
    blk2 = fake_block([1, 2], [0, 1], ["A", "A"], [None, None], [True, False])
    f2, n_nolog = R.model_failures(blk2, "B")
    assert list(f2) == [True, False] and n_nolog == 2


def test_flags_match_pipeline_function_for_the_nonlive_model():
    """EEGFineTunePipeline._failures_per_model with per_trial built from the same logged votes gives the same flags."""
    ids = np.array([3, 1, 2, 0])
    y = np.array([0, 1, 1, 0])
    live = {0: "A", 1: "B", 2: "A", 3: "B"}
    logged = {0: {"A": 0, "B": 1}, 1: {"A": 1, "B": 0}, 2: {"A": 0, "B": 0}, 3: {"A": 1, "B": None}}
    label = {int(i): int(l) for i, l in zip(ids, y)}
    per_trial = {i: {"live": live[i], "A": None if logged[i]["A"] is None else logged[i]["A"] == label[i],
                     "B": None if logged[i]["B"] is None else logged[i]["B"] == label[i]} for i in live}
    stored = np.array([True, False, True, False])
    exp = EEGFineTunePipeline._failures_per_model({"trial_ids": ids, "failures": stored}, per_trial)
    blk = fake_block(ids, y, [live[int(i)] for i in ids], [logged[int(i)] for i in ids], stored)
    for m in "AB":
        got, _ = R.model_failures(blk, m)
        assert list(got) == list(np.asarray(exp[m], bool)), m


# ---------------- V1 ----------------
def test_holdout_normal_block_alternates_class_and_holds_whole_trials():
    ids = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 0, 1])
    picks = []
    for counter in range(4):
        held, fb = R.holdout_trials(y, ids, counter)
        assert len(held) == 1 and not fb
        picks.append(int(y[held[0]]))
    assert picks == [0, 1, 0, 1]
    held, fb = R.holdout_trials(np.array([1, 1, 1]), np.array([5, 6, 7]), 0)  # no class-0 trial: the other class, flagged
    assert fb and held == [2]


def test_holdout_buffer_val_rolling_12_train_fifo_and_evicted_dropped():
    hb = R.HoldoutBuffer()
    nid = 0
    ds = synth_ds(nid, np.tile([0, 1], 24), np.zeros(48, bool))
    hb.add_block(ds, np.tile([0, 1], 24), np.arange(48))
    nid += len(ds)
    assert len(hb.val[0]) == 6 and len(hb.val[1]) == 6  # calibration block: 12 trials, 6 per class
    assert n_windows(hb.val_set()) == 12 * 16 and n_windows(hb.train_set()) == 36 * 16
    all_val_ids = set(ids_of(hb.val_set()))
    assert not (all_val_ids & set(ids_of(hb.train_set())))
    held_log, ever_train = [], set()
    for b in range(60):
        ds = synth_ds(nid, [0, 1, 0, 1], np.zeros(4, bool))
        hb.add_block(ds, np.array([0, 1, 0, 1]), np.arange(4) + 100 + 4 * b)
        held_log.append(hb.last_held)
        nid += len(ds)
        assert len(hb.val[0]) <= 6 and len(hb.val[1]) <= 6
        v, t = set(ids_of(hb.val_set())), set(ids_of(hb.train_set()))
        assert not (v & t)
        ever_train |= t
        assert not ((ever_train - t) & v)  # a window evicted from train never shows up in val
        # val holds whole trials only: windows come in complete groups of 16 consecutive ids
        vids = sorted(v)
        assert all(vids[i + 15] - vids[i] == 15 for i in range(0, len(vids), 16))
    ts = hb.train_set()
    assert n_windows(ts) == 768  # per class FIFO full at 384
    # evicted train windows are dropped: no window of the first calibration block's training trials is anywhere any more
    assert held_log[0] != held_log[1]  # class alternates block to block
    assert len(hb.val[0]) == 6 and len(hb.val[1]) == 6
    # the held-out trials never enter train, also after they leave val
    first_train_ids = set(range(0, 48 * 16))
    assert not (set(ids_of(hb.train_set())) & first_train_ids)


def n_windows(ds):
    return len(ds)


def tiny_model(seed=0, ch=4):
    torch.manual_seed(seed)
    return C.config.USE_MODEL(**dict(C.config.LOAD_MODEL_PARAM, channels=ch))


def tiny_ds(n, ch=4, seed=0, flip=False, weights=None):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    X = torch.randn(n, 1, ch, 500, generator=g) * 0.5 + (y.float() * 2 - 1).view(-1, 1, 1, 1) * torch.linspace(0.3, 1, 500).view(1, 1, 1, 500)
    lab = 1 - y if flip else y
    w = torch.ones(n) if weights is None else weights
    return TensorDataset(X, torch.nn.functional.one_hot(lab, 2).float(), w)


def test_train_select_without_parent_equals_online_trainer_on_cpu():
    cpu = torch.device("cpu")
    m0 = tiny_model()
    path = os.path.join(TMP, "tiny_parent.pth")
    torch.save({"model_state_dict": m0.state_dict(), "n_chans": 4}, path)
    train, val = tiny_ds(40, seed=1, weights=torch.tensor([1.0, 2.0] * 20)), tiny_ds(24, seed=2)
    tr = OnlineCalibrationTrainer(model_class=C.config.USE_MODEL, model_kwargs=dict(C.config.LOAD_MODEL_PARAM, channels=4), batch_size=8, num_epochs=4,
                                  lr=C.config.adaption_learning_rate, device=cpu, ft=True, tag="eqtest", seed=0)
    tr.gen = R.make_gen(7)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tr.load_checkpoint(path)
        torch.manual_seed(5)
        tr.train(train, val)
    ref = {k: v.clone() for k, v in tr.model.state_dict().items()}
    m = tiny_model(seed=99)
    m.load_state_dict(m0.state_dict())
    torch.manual_seed(5)
    info = R.train_select(m, train, val, R.make_gen(7), include_parent=False)
    for k in ref:
        assert torch.equal(ref[k], m.state_dict()[k]), k
    assert info["chosen"] in (1, 2, 3, 4) and len(info["val"]) == 4


def test_train_select_can_keep_the_parent():
    val = tiny_ds(40, seed=1, flip=True)
    m = tiny_model()
    R.train_select(m, val, val, R.make_gen(3), include_parent=False, epochs=8)  # parent fits the validation labels
    train = tiny_ds(40, seed=1)  # training labels contradict them: every epoch makes validation worse
    parent = {k: v.clone() for k, v in m.state_dict().items()}
    info = R.train_select(m, train, val, R.make_gen(3), include_parent=True)
    assert info["chosen"] == 0 and len(info["val"]) == 5 and info["val"][0] == min(info["val"])
    assert all(torch.equal(parent[k], v) for k, v in m.state_dict().items())
    # with a validation set that agrees with training, an epoch wins
    m2 = tiny_model()
    info2 = R.train_select(m2, train, tiny_ds(40, seed=1), R.make_gen(3), include_parent=True)
    assert info2["chosen"] >= 1 and info2["val"][info2["chosen"]] == min(info2["val"])


def test_generator_replay_equals_consumption_by_training():
    cpu = torch.device("cpu")
    m0 = tiny_model()
    path = os.path.join(TMP, "tiny_parent2.pth")
    torch.save({"model_state_dict": m0.state_dict(), "n_chans": 4}, path)
    train = tiny_ds(37, seed=4)
    g1, g2 = R.make_gen(42), R.make_gen(42)
    tr = OnlineCalibrationTrainer(model_class=C.config.USE_MODEL, model_kwargs=dict(C.config.LOAD_MODEL_PARAM, channels=4), batch_size=8, num_epochs=4,
                                  lr=C.config.adaption_learning_rate, device=cpu, ft=True, tag="gentest", seed=0)
    tr.gen = g1
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tr.load_checkpoint(path)
        tr.train(train, tiny_ds(8, seed=9))
    R.advance_gen(g2, 37)
    assert torch.equal(g1.get_state(), g2.get_state())


def test_seed_values():
    assert R.seed_values("A", 0) == (42, 42) and R.seed_values("B", 0) == (42, 43)
    assert R.seed_values("A", 3) == (3042, 3042) and R.N_SEEDS == 5


# ---------------- V2 ----------------
def test_v2_decision_accepts_only_a_strictly_better_candidate():
    block = tiny_ds(32, seed=6)
    good, bad = tiny_model(1), tiny_model(2)
    # make `good` fit the block: train it on the block itself, leave `bad` random
    R.train_select(good, block, block, R.make_gen(1), include_parent=False, epochs=6)
    acc, lp, lc = V.v2_decide(bad, good, block)
    assert acc and lc < lp
    acc2, lp2, lc2 = V.v2_decide(good, bad, block)
    assert not acc2 and lc2 >= lp2
    acc3, _, _ = V.v2_decide(good, good, block)  # tie keeps the parent
    assert not acc3


def test_v2_candidate_buffer_has_only_blocks_before_k_and_decision_set_is_block_k():
    if not os.path.isdir(os.path.join(F.session_dir("sub01"), "run1")):
        skip("sub01 not found")
    sess = F.Session("sub01", load_eeg=False)
    pos = C.config.channel_positions(22)
    blocks = R.load_blocks(sess)
    seen = 0
    for ev in R.Stream(sess, "A", "asran", ("v0", "v2"), assigned=lambda k: k in (1, 2, 3, 4)).events():
        k_block = ev["blk"].number
        before = [b for b in blocks if b.number < k_block]
        if not before:
            assert ev["v2"] is None and ev["n_prev_blocks"] == 0
            continue
        ref = ReplayBuffer()
        for b in before:
            f, _ = R.model_failures(b, "A")
            ref.add(R.window_ds(b.x, b.y, f, pos))
        assert torch.equal(ev["v2"]["train"].tensors[0], ref.train_set().tensors[0])
        assert torch.equal(ev["v2"]["val"].tensors[0], ref.val_set().tensors[0])
        f, _ = R.model_failures(ev["blk"], "A")
        blk_ds = R.weighted(R.window_ds(ev["blk"].x, ev["blk"].y, f, pos))
        assert torch.equal(ev["v2"]["block"].tensors[0], blk_ds.tensors[0]) and torch.equal(ev["v2"]["block"].tensors[2], blk_ds.tensors[2])
        # the decision block is not in the candidate's buffer, but is in the v0 buffer
        assert len(ev["v0"]["train"]) >= len(ev["v2"]["train"])
        seen += 1
    assert seen >= 2


# ---------------- real data ----------------
def test_block_mapping_counts_sub01_sub05():
    if not os.path.isdir(os.path.join(F.session_dir("sub01"), "run1")):
        skip("sub01 not found")
    for sub, exp_blocks, exp_missing in (("sub01", 85, {("run1", 3): [4]}), ("sub05", 12, {})):
        sess = F.Session(sub, load_eeg=False)
        blocks = R.load_blocks(sess)
        assert len(blocks) == exp_blocks, (sub, len(blocks))
        assert [b.number for b in blocks] == list(range(1, exp_blocks + 1))  # ab_blocks numbering is continuous: no Python restart inside the session
        miss = {(b.run.name, b.number - int(b.run.blocks[0]["block"]) + 1): b.missing_trials for b in blocks if b.missing_trials}
        assert miss == exp_missing, miss
        for run in sess.runs:
            if not run.test_only:
                assert len(run.data_files) == len(run.blocks)
        for b in blocks:
            assert b.n == b.n_expected - len(b.missing_trials) and b.x.shape == (b.n, 32, 2000)


def test_corrected_cut_with_lag_zero_equals_x_data():
    if not os.path.isdir(os.path.join(F.session_dir("sub05"), "run1")):
        skip("sub05 not found")
    sess = F.Session("sub05", load_eeg=True)
    blocks = R.load_blocks(sess)
    worst = max(R.cut_lag0_equals_data(b) for b in blocks)
    assert worst == 0.0, worst
    lags = sess.load_lags()
    b = blocks[0]
    x = R.corrected_x(sess, b, lags, cache=False)
    assert x.shape == b.x.shape and not np.array_equal(x, b.x)  # the corrected cut differs (about 1 s earlier)
    assert 0.9 < lags["run1"] < 1.1


def test_late_set_belongs_to_finetunes_trained_in_run_1():
    if not os.path.isdir(os.path.join(F.session_dir("sub01"), "run7")):
        skip("sub01 not found")
    sess = F.Session("sub01", load_eeg=False)
    fts = {(f["model"], f["k"]): f for f in sess.finetunes}
    for k, early in ((9, True), (10, True), (11, False)):  # cA_010 is trained on the last block of run 1 but goes live in run 2
        w1, w2, late = R.ft_windows(sess, fts[("A", k)])
        assert R.trained_in_first_run(sess, fts[("A", k)]) == early
        assert (len(late) == 48 and all(sess.trials[i]["run"] == 7 for i in late)) == early and (len(late) == 0) == (not early)
    w1, w2, late = R.ft_windows(sess, fts[("A", 10)])
    assert sess.trials[int(w2[0])]["run"] == 2
    assert len(R.eval_seqs(sess, fts[("A", 10)])) >= 48 + 48


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
