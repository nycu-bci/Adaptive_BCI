"""
A/B 雙模型平行微調、Unity 式計分、每個模型自己的微調 gate、vote flag、checkpoint 編號的測試
(不需要 Unity / EEG 裝置，用暫存資料夾)。

用法：C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_parallel_finetune.py   (或 pytest 這個檔案)
"""
import io
import os
import re
import sys
import time
import hashlib
import tempfile
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
from torch.utils.data import TensorDataset

import main.Utils.config as config
import main.Utils.global_value as gv
import main.Utils.trial_scoring as ts
from main.EEG.models import SCCNet
from main.EEG.EEGPrediction import EEGPredictor
from main.EEG.EEG_Train import EEGFineTunePipeline, arrange_by_label, prepare_calibration_dataset
from main.EEG.MI_train import ReplayBuffer
from main.EEG.data_process_np import EEGDataLoader
from main.Utils.UnityMarkerReader import UnityTCPReader
from main.Utils.file_pointer_reader import readLogFromPointer
from main.Utils.some_functions import get_next_version_path, resolve_start_checkpoint

LIVE_PATTERN = re.compile(r'Trial\s+(\d+)\s+(START|CUT|WRONG_CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?', re.IGNORECASE)
STATE_KEYS = ("A_CHECKPOINT", "B_CHECKPOINT", "NOW_TRAINED_CHECKPOINT", "LIVE_MODE", "live_model", "train_np_data",
              "last_block_trials", "eeg_buffer", "broadcast_model", "trial_live", "trial_start_ts", "run_mode",
              "runCount", "vote_on_at", "measured_vote_offset", "unity_marker_string_calibration",
              "unity_marker_string_stage", "subject", "session", "is_calibration_run", "run_start_time", "last_swap_time")


def make_ckpt(path, n_chans, seed=0):
    torch.manual_seed(seed)
    m = SCCNet(samples=500, channels=n_chans, n_classes=2, sfreq=500)
    torch.save({"model_state_dict": m.state_dict(), "n_chans": n_chans, "n_outputs": 2, "n_times": 500}, path)


class Env:
    """把 checkpoint / 暫存 / run 資料夾導向暫存目錄，並重設 global_value 的 A/B 狀態"""

    def __init__(self):
        self.dir = tempfile.mkdtemp()
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")  # 微調後的 cA_NNN / cB_NNN (config.getSessionCheckpointDir)
        os.makedirs(self.ckpt_dir)
        self.saved = {k: getattr(gv, k) for k in STATE_KEYS}
        self.saved_cfg = (config.EEG_CHECKPOINT_TMP_BASE_FILE, config.getRunDataDir, config.getSessionCheckpointDir)
        self.saved_gate = (config.FINETUNE_GATE, config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA)
        config.FINETUNE_GATE = "block"  # 舊測試照舊走 block gate；SPRT 測試自己切換
        config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA = 0.5, 0.75, 0.05, 0.2
        gv.A_CHECKPOINT = os.path.join(self.dir, "cA_000.pth")
        gv.B_CHECKPOINT = os.path.join(self.dir, "cB_000.pth")
        gv.NOW_TRAINED_CHECKPOINT = gv.A_CHECKPOINT
        gv.LIVE_MODE, gv.live_model, gv.broadcast_model = "A", "A", "A"
        gv.train_np_data, gv.last_block_trials = [], None
        gv.trial_live, gv.trial_start_ts, gv.run_mode, gv.runCount = {}, {}, "", 1
        gv.vote_on_at = gv.measured_vote_offset = None
        gv.unity_marker_string_calibration = ""
        gv.is_calibration_run = False
        for log in gv.prediction_logs.values():
            log.clear()
        config.EEG_CHECKPOINT_TMP_BASE_FILE = os.path.join(self.dir, "tmp") + os.sep
        os.makedirs(config.EEG_CHECKPOINT_TMP_BASE_FILE)
        config.getRunDataDir = lambda: self.dir
        config.getSessionCheckpointDir = lambda: self.ckpt_dir
        make_ckpt(gv.A_CHECKPOINT, 22)  # 預訓練 A：22 通道

    def close(self):
        for k, v in self.saved.items():
            setattr(gv, k, v)
        config.EEG_CHECKPOINT_TMP_BASE_FILE, config.getRunDataDir, config.getSessionCheckpointDir = self.saved_cfg
        config.FINETUNE_GATE, config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA = self.saved_gate

    def new_block(self, t0, seed, a_ok=(False,) * 4, b_ok=(False,) * 4, reverse=False):
        """
        產生一個 4-trial block：.pt、trial 時間/標籤/投票、兩個模型的假預測。
        a_ok：live 的 A 每個 trial 是否被 Unity 判對 (CUT / WRONG_CUT)；b_ok：B 的 Unity 式投票是否對。
        reverse=True 時 .pt 裡的 trial 順序倒過來 (測 trial_ids 對齊)。
        """
        rng = np.random.default_rng(seed)
        y = np.array([0, 1, 0, 1])
        ids = np.arange(4)
        x = rng.standard_normal((4, config.N_CHANNELS, 1000))
        fails = np.array([not ok for ok in a_ok])
        if reverse:
            x, y_pt, fails_pt, ids = x[::-1], y[::-1], fails[::-1], ids[::-1]
        else:
            y_pt, fails_pt = y, fails
        path = os.path.join(self.dir, f"data_{seed}.pt")
        torch.save({"x_data": np.ascontiguousarray(x), "y_data": np.ascontiguousarray(y_pt),
                    "failures": np.ascontiguousarray(fails_pt), "trial_ids": np.ascontiguousarray(ids)}, path)
        gv.train_np_data.append(path)
        trials = {}
        for i in range(4):
            start = t0 + 10 * i
            t = {"start": start, "end": start + 4.95, "label": int(y[i])}
            if a_ok[i]:
                t["cut"] = [start + 4.25]
            else:
                t["wrong_cut"] = start + 4.25
            trials[i] = t
        gv.last_block_trials = trials
        with gv.prediction_log_lock:
            for i, t in trials.items():
                for k in range(120):  # 投票前 >= 100 個預測，Unity 式投票只看這個 trial 自己的預測
                    tt = t["start"] + 0.035 * k
                    pa = t["label"] if a_ok[i] else 1 - t["label"]
                    pb = t["label"] if b_ok[i] else 1 - t["label"]
                    gv.prediction_logs["A"].append((tt, pa, 0.9 if pa else 0.1))
                    gv.prediction_logs["B"].append((tt, pb, 0.9 if pb else 0.1))
        return trials


def ckpt_files(env):
    return sorted(f for f in os.listdir(env.ckpt_dir))


def abrows(env):
    return [r.split(",") for r in open(os.path.join(env.dir, "ab_blocks.csv")).read().splitlines()]


def buffer_total(slot):
    tr, va = slot.buffer.sizes()
    return sum(tr.values()), sum(va.values())


def test_unity_vote_matches_autosaber():
    assert ts.unity_vote([1]) is None and ts.unity_vote([]) is None
    # n=2：權重 1,5 -> total 6，count1 只有最新=1 時 5 >= 3
    assert ts.unity_vote([0, 1]) == 1 and ts.unity_vote([1, 0]) == 0
    # n=5：權重 1,2,3,4,5 total 15；[1,1,1,0,0] -> count1=6 < 7.5 -> 0；[0,0,1,1,1] -> 12 -> 1
    assert ts.unity_vote([1, 1, 1, 0, 0]) == 0 and ts.unity_vote([0, 0, 1, 1, 1]) == 1
    # 只看最後 100 個
    assert ts.unity_vote([1] * 500 + [0] * 100) == 0
    # 半數捨入 (round half to even)：n=3 權重 1,3,5；total 9，[0,0,1] -> 5>=4.5 -> 1
    assert ts.unity_vote([0, 0, 1]) == 1


def test_window_vote_tie_uses_mean_probability():
    e = lambda p, q: (0.0, p, q)
    assert ts.window_vote([e(1, .9), e(1, .8), e(0, .1)]) == 1
    assert ts.window_vote([e(1, .6), e(0, .1)]) == 0     # 平手，mean prob 0.35 -> 0
    assert ts.window_vote([e(1, .9), e(0, .8)]) == 1     # 平手，mean 0.85 -> 1
    assert ts.window_vote([]) is None


def test_model_prediction_suffix_and_parse():
    logs = {"A": [(100.0 + 0.01 * i, 1, 0.9) for i in range(200)], "B": []}
    gv.live_model, gv.broadcast_model = "A", ""
    suffix = ts.model_prediction_suffix(102.70, logs)             # 沒有投票時間：AT = END - 0.70
    assert suffix == "MODEL_PREDICTION: A=1 B=NA LIVE=A AT: 102.000", suffix
    suffix = ts.model_prediction_suffix(102.70, logs, vote_ts=101.5, live="B")   # 有投票時間：AT = 投票時間
    assert suffix == "MODEL_PREDICTION: A=1 B=NA LIVE=B AT: 101.500", suffix
    line = f"Trial 2 END: 102.700 LABEL: 0 {suffix}"
    m = LIVE_PATTERN.match(line)
    assert m and float(m.group(3)) == 102.7 and m.group(4) == "0"


def test_score_block_from_cut_wrong_cut_and_fallback():
    logs = {"A": [(100.0 + 0.1 * i, 1, 0.9) for i in range(400)],     # A 一直預測 1
            "B": [(100.0 + 0.1 * i, 0, 0.1) for i in range(400)]}     # B 一直預測 0
    trials = {
        0: {"end": 105.0, "label": 1, "cut": [104.3]},               # live A，CUT -> A 對；B 預測 0 -> 錯
        1: {"end": 115.0, "label": 0, "wrong_cut": 114.3},           # live A，WRONG_CUT -> A 錯；B 預測 0 -> 對
        2: {"end": 125.0, "label": 1},                               # live B，沒有任何投票行 -> B 錯 (沒有 CUT)；A 重算 -> 對
        3: {"label": 1},                                             # 沒有 END：不計分
    }
    scores, per = ts.score_block(trials, logs, {0: "A", 1: "A", 2: "B"})
    assert scores == {"A": (2, 3), "B": (1, 3)}, scores
    assert per[0]["A"] is True and per[0]["B"] is False and per[0]["vote_ts"] == 104.3
    assert per[1]["A"] is False and per[1]["B"] is True and per[1]["vote_ts"] == 114.3
    assert per[2]["live"] == "B" and per[2]["B"] is False and per[2]["A"] is True
    assert abs(per[2]["vote_ts"] - (125.0 - config.PREDICTION_LOG_OFFSET_S)) < 1e-9 and 3 not in per
    # 另一個模型在投票時間之前沒有足夠預測 -> 無法計分
    scores, per = ts.score_block({0: {"end": 90.0, "label": 1, "cut": [89.0]}}, logs, {0: "A"})
    assert per[0]["B"] is None and scores["B"] == (0, 0) and scores["A"] == (1, 1)
    assert config.PREDICTION_LOG_OFFSET_S == 0.70


def test_end_line_written_only_on_end_and_at_is_vote_time():
    gv.live_model = gv.broadcast_model = "A"
    gv.trial_live, gv.trial_start_ts, gv.run_mode = {}, {}, ""
    reader = UnityTCPReader()
    reader.csv_file = io.StringIO()
    now = time.time()
    with gv.prediction_log_lock:
        for log in gv.prediction_logs.values():
            log.clear()
        for k in range(200):
            gv.prediction_logs["A"].append((now - 3 + 0.01 * k, 1, 0.9))
            gv.prediction_logs["B"].append((now - 3 + 0.01 * k, 0, 0.1))
    msgs = (f"Trial 0 START: {now - 3:.3f} LABEL: 1 MODE: Adaptive", f"Trial 0 CUT: {now - 1:.3f}",
            f"Trial 0 END: {now:.3f} LABEL: 1", "Close eyes: 123.456",
            f"Trial 1 START: {now + 1:.3f} LABEL: 0 MODE: Adaptive", f"Trial 1 WRONG_CUT: {now + 2:.3f}",
            f"Trial 1 END: {now + 3:.3f} LABEL: 0",
            f"Trial 2 START: {now + 4:.3f} LABEL: 0 MODE: Adaptive", f"Trial 2 END: {now + 8:.3f} LABEL: 0")
    for msg in msgs:
        reader.process_message(msg, None)
    lines = reader.csv_file.getvalue().splitlines()
    assert lines[0].endswith("LABEL: 1 MODE: Adaptive") and "MODEL_PREDICTION" not in lines[0]
    assert gv.run_mode == "Adaptive"
    assert lines[1] == f"Trial 0 CUT: {now - 1:.3f}"
    assert lines[2].startswith(f"Trial 0 END: {now:.3f} LABEL: 1 MODEL_PREDICTION: A=1 B=0 LIVE=A AT: {now - 1:.3f}"), lines[2]
    assert lines[3] == "Close eyes: 123.456"
    assert lines[5] == f"Trial 1 WRONG_CUT: {now + 2:.3f}"                       # WRONG_CUT 會寫進 log
    assert lines[6].endswith(f"LIVE=A AT: {now + 2:.3f}"), lines[6]
    assert lines[8].endswith(f"AT: {now + 8 - config.PREDICTION_LOG_OFFSET_S:.3f}"), lines[8]  # 沒有投票行 -> END - 0.70

    path = os.path.join(tempfile.mkdtemp(), "log.txt")
    open(path, "w").write("\n".join(lines) + "\n")
    trials, _, n = readLogFromPointer(path, 0)
    assert n == 9 and trials[0]["label"] == 1 and "start" in trials[0] and "end" in trials[0] and len(trials[0]["cut"]) == 1
    assert abs(trials[1]["wrong_cut"] - (now + 2)) < 1e-3 and "cut" not in trials[1]


def test_run_dir_uses_subject_and_session():
    base = tempfile.mkdtemp()
    saved = (getattr(config, "__REALTIME_BASE_FILE"), gv.subject, gv.session, gv.runCount)
    try:
        setattr(config, "__REALTIME_BASE_FILE", base)
        gv.runCount = 2
        gv.subject = gv.session = ""
        assert config.getRunDataDir() == os.path.join(base, "run2")
        gv.subject, gv.session = "01", "3"
        assert config.getRunDataDir() == os.path.join(base, "sub01", "ses3", "run2")
        assert os.path.isdir(os.path.join(base, "sub01", "ses3", "run2"))
        assert config.getSessionCheckpointDir() == os.path.join(base, "sub01", "ses3", "checkpoints")
    finally:
        setattr(config, "__REALTIME_BASE_FILE", saved[0])
        gv.subject, gv.session, gv.runCount = saved[1:]


def test_switch_rule_strict_two_in_a_row():
    env = Env()
    try:
        p = EEGFineTunePipeline()
        gv.LIVE_MODE = "auto"
        S = lambda a, b: {"A": (a, 4), "B": (b, 4)}
        p._decide_live(S(2, 3)); assert gv.live_model == "A" and p.streak == 1
        p._decide_live(S(3, 3)); assert gv.live_model == "A" and p.streak == 0   # 平手重設
        p._decide_live(S(2, 3)); p._decide_live(S(4, 3)); assert gv.live_model == "A" and p.streak == 0  # 輸了重設
        p._decide_live(S(1, 2)); p._decide_live(S(1, 4)); assert gv.live_model == "B" and p.streak == 0  # 連兩次 -> 切換
        p._decide_live(S(4, 3)); p._decide_live(S(3, 4)); assert gv.live_model == "B" and p.streak == 0  # A 贏一次又輸 -> 不切
        p._decide_live(S(4, 3)); p._decide_live(S(4, 0)); assert gv.live_model == "A"                    # 可切回
        p._decide_live({"A": (0, 0), "B": (3, 4)}); assert gv.live_model == "A" and p.streak == 0        # 無可計分 trial
        gv.LIVE_MODE = "B"; p._decide_live(S(4, 0)); assert gv.live_model == "B"                          # 固定模式
    finally:
        env.close()


def test_streak_switching_with_unity_based_scores():
    env = Env()
    try:
        gv.LIVE_MODE = "auto"
        pipe = EEGFineTunePipeline()
        pipe.init_pipeline()
        bad, good = (False,) * 4, (True,) * 4
        env.new_block(1_000_000.0, 1, a_ok=bad, b_ok=good)       # A (live) 被 Unity 判 0/4，B 重算 4/4
        pipe.run_calibration()
        assert gv.live_model == "A" and pipe.streak == 1
        env.new_block(1_000_100.0, 2, a_ok=bad, b_ok=good)       # 還沒 reload，broadcast 仍是 A -> 第二次 -> 切換
        pipe.run_calibration()
        assert gv.live_model == "B" and pipe.streak == 0 and gv.broadcast_model == "A"
        rows = abrows(env)
        assert rows[1][2:4] == ["0", "4"] and rows[2][2:4] == ["0", "4"]
    finally:
        env.close()


def test_parallel_finetune_two_chains_and_stale_guard():
    env = Env()
    try:
        pred_b = EEGPredictor(tcp_server=None, name="B", live=False)  # 檔案不存在 -> 產生 32 通道隨機 cB_000
        assert os.path.exists(gv.B_CHECKPOINT)
        assert torch.load(gv.B_CHECKPOINT, map_location="cpu", weights_only=False)["n_chans"] == config.N_CHANNELS
        a_before = hashlib.md5(open(gv.A_CHECKPOINT, "rb").read()).hexdigest()

        pipe = EEGFineTunePipeline(tcp_server=None)
        pipe.init_pipeline()
        assert pipe.slots["A"].n_chans == 22 and pipe.slots["B"].n_chans == config.N_CHANNELS
        assert pipe.slots["A"].buffer is not pipe.slots["B"].buffer

        t0 = 1_000_000.0
        a_ok = (True, False, False, False)   # A 1/4 -> 累計 < 0.75 且有錯 -> 微調
        env.new_block(t0, seed=1, a_ok=a_ok)
        pipe.run_calibration()
        env.new_block(t0 + 100, seed=2, a_ok=a_ok)
        pipe.run_calibration()

        assert ckpt_files(env) == ["cA_001.pth", "cA_002.pth", "cB_001.pth", "cB_002.pth"], ckpt_files(env)
        assert "ab_blocks.csv" in os.listdir(env.dir)
        ck = lambda f: torch.load(os.path.join(env.ckpt_dir, f), map_location="cpu", weights_only=False)
        assert ck("cA_002.pth")["n_chans"] == 22 and ck("cB_002.pth")["n_chans"] == config.N_CHANNELS
        assert gv.A_CHECKPOINT == os.path.join(env.ckpt_dir, "cA_002.pth") and gv.B_CHECKPOINT.endswith("cB_002.pth")
        assert gv.NOW_TRAINED_CHECKPOINT == gv.A_CHECKPOINT                    # live = A
        assert hashlib.md5(open(os.path.join(env.dir, "cA_000.pth"), "rb").read()).hexdigest() == a_before
        # 各自的 replay buffer、各自的暫存 checkpoint
        assert buffer_total(pipe.slots["A"]) == buffer_total(pipe.slots["B"]) and buffer_total(pipe.slots["A"])[0] == 48
        tmp = os.listdir(config.EEG_CHECKPOINT_TMP_BASE_FILE)
        assert "ftA-best.pth" in tmp and "ftB-best.pth" in tmp
        # A：Unity 判 1/4；B：重算全錯 0/4
        rows = abrows(env)
        assert len(rows) == 3 and rows[1][2:4] == ["1", "0"]
        assert rows[0] == EEGFineTunePipeline.HEADER and rows[0][:10][-1] == "mode"
        assert rows[1][10:12] == ["1", "1"] and float(rows[1][12]) > 0 and rows[1][13] == ""

        # 舊資料保護：沒有新 block 時不訓練、不新增檔案，仍會廣播 calibration done
        sent = []
        pipe.tcp_server = type("T", (), {"broadcast": lambda self, m: sent.append(m)})()
        before = ckpt_files(env)
        pipe.run_calibration()
        assert ckpt_files(env) == before and sent == [config.CALIBRATION_FINISH_STR]
        assert pipe.block == 2
        # B 變成 live：NOW_TRAINED_CHECKPOINT 指向 B 的鏈
        gv.LIVE_MODE = "B"
        env.new_block(t0 + 200, seed=3, a_ok=a_ok)
        pipe.run_calibration()
        assert gv.live_model == "B" and gv.NOW_TRAINED_CHECKPOINT == gv.B_CHECKPOINT and gv.B_CHECKPOINT.endswith("cB_003.pth")
        del pred_b
    finally:
        env.close()


def test_gate_matrix_and_buffers_always_grow():
    cases = [  # (A 被 Unity 判對的 trial, B 重算對的 trial, A 微調, B 微調)
        ((False,) * 4, (True,) * 4, 1, 0),
        ((True,) * 4, (False,) * 4, 0, 1),
        ((True,) * 4, (True,) * 4, 0, 0),
        ((False,) * 4, (False,) * 4, 1, 1),
    ]
    for a_ok, b_ok, want_a, want_b in cases:
        env = Env()
        try:
            make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
            pipe = EEGFineTunePipeline()
            pipe.init_pipeline()
            env.new_block(1_000_000.0, 7, a_ok=a_ok, b_ok=b_ok)
            pipe.run_calibration()
            want = sorted([f"cA_001.pth"] * want_a + [f"cB_001.pth"] * want_b)
            assert ckpt_files(env) == want, (a_ok, b_ok, ckpt_files(env))
            row = abrows(env)[1]
            assert row[10:12] == [str(want_a), str(want_b)], row
            assert buffer_total(pipe.slots["A"])[0] == 24 and buffer_total(pipe.slots["B"])[0] == 24  # 兩個 buffer 都收到這個 block
            if not want_a:
                assert gv.A_CHECKPOINT == os.path.join(env.dir, "cA_000.pth")
        finally:
            env.close()


def test_gate_needs_run_accuracy_below_threshold_and_resets_per_run():
    env = Env()
    try:
        make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
        pipe = EEGFineTunePipeline()
        pipe.init_pipeline()
        good = (True,) * 4
        ft = lambda: abrows(env)[-1][10:12]
        env.new_block(1_000_000.0, 1, a_ok=good, b_ok=good)
        pipe.run_calibration(); assert ft() == ["0", "0"]
        env.new_block(1_000_100.0, 2, a_ok=(True, True, True, False), b_ok=good)  # 累計 7/8 = 0.875 >= 0.75，有錯也不訓練
        pipe.run_calibration(); assert ft() == ["0", "0"]
        env.new_block(1_000_200.0, 3, a_ok=(True, False, False, False), b_ok=good)  # 累計 8/12 = 0.67 < 0.75 -> A 訓練
        pipe.run_calibration(); assert ft() == ["1", "0"]
        assert pipe.run_counts["A"] == [8, 12] and pipe.run_counts["B"] == [12, 12]
        gv.runCount += 1                                                         # 新 run：累計重設
        env.new_block(1_000_300.0, 4, a_ok=(True, True, True, False), b_ok=good)  # 3/4 = 0.75，不 < 0.75
        pipe.run_calibration(); assert ft() == ["0", "0"] and pipe.run_counts["A"] == [3, 4]
        env.new_block(1_000_400.0, 5, a_ok=(False, False, True, True), b_ok=good)  # 累計 5/8
        pipe.run_calibration(); assert ft() == ["1", "0"]
    finally:
        env.close()


def test_failure_flag_per_model_and_trial_ids_alignment():
    env = Env()
    try:
        make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
        pipe = EEGFineTunePipeline()
        pipe.init_pipeline()
        # trial 1 (label 1) A 被判錯；trial 0 (label 0) B 重算錯；.pt 裡的順序倒過來；兩個模型都 >= 0.75，不訓練，只看 buffer
        env.new_block(1_000_000.0, 3, a_ok=(True, False, True, True), b_ok=(False, True, True, True), reverse=True)
        pipe.run_calibration()
        assert ckpt_files(env) == []
        weight_sum = lambda slot, c: float(sum(it[2] for it in slot.buffer.train[c]))
        assert (weight_sum(pipe.slots["A"], 0), weight_sum(pipe.slots["A"], 1)) == (12.0, 18.0)   # A：trial 1 (label 1) 的 6 個 window 權重 2
        assert (weight_sum(pipe.slots["B"], 0), weight_sum(pipe.slots["B"], 1)) == (18.0, 12.0)   # B：trial 0 (label 0)
    finally:
        env.close()


def test_trial_ids_follow_arrange_by_label():
    fs, labels = 500, [0, 0, 1, 0, 1, 1, 1, 0]
    t = np.arange(0, 8 * 6 * fs) / fs
    eeg = np.repeat(np.floor(t // 6)[:, None], 4, axis=1)               # 每個 trial 的訊號值 = trial 編號
    trials = {i: {"start": 6.0 * i, "end": 6.0 * i + 4.95, "label": labels[i], "cut": [6.0 * i + 4.25]} for i in range(8)}
    loader = EEGDataLoader([], [], channel_index=[0, 1, 2, 3])
    loader.rec_epoch = 0.0
    loader._extract_segments(eeg, t, trials)
    x, y, f = loader.get_eeg_trial_channel_sample_np()
    ids = loader.get_trial_ids_np()
    assert list(ids) == list(range(8)) and x.shape[0] == 8
    _, _, ids2 = arrange_by_label(x, y, ids)
    x2, y2, f2 = arrange_by_label(x, y, f)
    assert list(ids2) != list(ids)                                       # 真的被重排
    for r in range(8):
        assert round(float(x2[r].mean())) == ids2[r] and y2[r] == labels[ids2[r]]


def test_failure_rule_cut_wrong_cut_and_old_logs():
    fs = 500
    t = np.arange(0, 5 * 6 * fs) / fs
    eeg = np.zeros((len(t), 2))
    def fails(trials):
        loader = EEGDataLoader([], [], channel_index=[0, 1])
        loader.rec_epoch = 0.0
        loader._extract_segments(eeg, t, trials)
        return [bool(v) for v in loader.failures]
    mk = lambda i, **kw: dict({"start": 6.0 * i, "end": 6.0 * i + 4.95, "label": 0}, **kw)
    trials = {0: mk(0, cut=[4.25]), 1: mk(1, wrong_cut=10.25), 2: mk(2), 3: mk(3, cut=[1, 2, 3, 4, 5]), 4: mk(4, cut=[1])}
    assert fails(trials) == [False, True, True, False, False]          # success_cut_count = 1：有 CUT 就成功
    old = config.SUCCESS_CUT_COUNT
    try:
        config.SUCCESS_CUT_COUNT = 5                                    # 舊資料：要 5 個 CUT
        assert fails(trials) == [True, True, True, False, True]
    finally:
        config.SUCCESS_CUT_COUNT = old


def test_replay_buffer_fifo_validation_cap_and_weights():
    ds = lambda n, label, fail: TensorDataset(
        torch.randn(n, 1, 2, 5), torch.nn.functional.one_hot(torch.full((n,), label), 2).float(), torch.full((n,), float(fail)))
    buf = ReplayBuffer(limit=10, use_val=True)
    assert buf.train_set() is None and buf.val_set() is None
    buf.add(ds(8, 0, False))
    assert buf.sizes() == ({0: 8, 1: 0}, {0: 4, 1: 0})                  # val 上限 int(10 * 0.4) = 4
    buf.add(ds(5, 0, True))
    buf.add(ds(3, 1, False))
    assert buf.sizes() == ({0: 10, 1: 3}, {0: 4, 1: 3})
    tr = buf.train_set()
    assert len(tr) == 13
    w = tr.tensors[2]
    assert int((w == 2.0).sum()) == 5 and int((w == 1.0).sum()) == 8      # 最舊的 3 個 label-0 (權重 1) 被擠出
    off = ReplayBuffer(limit=10, use_val=False)
    off.add(ds(4, 1, False))
    assert off.val_set() is None and off.sizes() == ({0: 0, 1: 4}, {0: 0, 1: 0})
    full = ReplayBuffer()                                                # 預設值取自 config
    assert full.limit == config.REPLAY_BUFFER_LIMIT


def test_vote_flag_window_and_measured_offset():
    env = Env()
    try:
        assert config.VOTE_OFFSET_S == 4.25 and config.SWAP_GUARD_S == 1.5
        ts.on_trial_start(0, 100.0)                                      # 第一個 trial：用 config 的 4.25 - 1.5
        assert gv.vote_on_at == 102.75
        assert not ts.vote_flag_on(102.0) and ts.vote_flag_on(102.8)
        ts.on_trial_vote(0, 104.30)                                      # 投票到達：關閉並量測 offset
        assert not ts.vote_flag_on(103.0) and gv.vote_on_at is None
        assert abs(gv.measured_vote_offset - 4.30) < 1e-9
        ts.on_trial_start(1, 200.0)                                      # 之後用實測值
        assert abs(gv.vote_on_at - (200.0 + 4.30 - 1.5)) < 1e-9
        ts.on_trial_end(1)                                               # 沒有投票行：END 也要關
        assert gv.vote_on_at is None and not ts.vote_flag_on(1e12)
        gv.vote_on_at = 100.0                                            # 另一種歌曲：實測值蓋過 config
        ts.on_trial_start(2, 300.0); ts.on_trial_vote(2, 303.1)
        ts.on_trial_start(3, 400.0)
        assert abs(gv.vote_on_at - (400.0 + 3.1 - 1.5)) < 1e-9
        gv.reset_run_state()
        assert gv.measured_vote_offset is None and gv.vote_on_at is None and gv.run_mode == "" and gv.trial_live == {}
    finally:
        env.close()


def test_swap_waits_for_vote_flag_off():
    env = Env()
    try:
        from main.game_state import Game
        calls = []
        game = types.SimpleNamespace(reload_predictors=lambda: calls.append(("reload", time.time())),
                                     ft_pipeline=types.SimpleNamespace(models_changed=True, record_swap=lambda t: calls.append(("swap", t))))
        gv.vote_on_at = time.time() - 1                                  # 旗標開著
        assert ts.wait_vote_flag_off(timeout=0.2) is False               # 逾時回傳 False，不會卡死
        t_off = time.time() + 0.4
        threading.Timer(0.4, lambda: setattr(gv, "vote_on_at", None)).start()
        Game.swap_in_new_models(game)
        assert calls[0][0] == "reload" and calls[0][1] >= t_off - 0.05   # 旗標關閉之後才載入
        assert calls[1][0] == "swap" and gv.last_swap_time is not None
        calls.clear()
        gv.vote_on_at = None
        Game.swap_in_new_models(game)                                    # 旗標關著：直接載入
        assert [c[0] for c in calls] == ["reload", "swap"]
    finally:
        env.close()


def test_record_swap_fills_last_block_row():
    env = Env()
    try:
        make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
        pipe = EEGFineTunePipeline()
        pipe.init_pipeline()
        env.new_block(1_000_000.0, 1)
        pipe.run_calibration()
        pipe.record_swap(1234.5678)
        assert abrows(env)[1][13] == "1234.568"
        pipe.record_swap(999.0)                                          # 沒有新 block：不改
        assert abrows(env)[1][13] == "1234.568"
    finally:
        env.close()


def test_predictions_csv():
    env = Env()
    try:
        with gv.prediction_log_lock:
            gv.prediction_logs["A"].extend([(10.0, 0, 0.2), (20.0, 1, 0.8)])
            gv.prediction_logs["B"].append((21.0, 1, 0.6))
        path = ts.dump_predictions(since=15.0)
        assert path == os.path.join(env.dir, "predictions.csv")
        assert open(path).read().splitlines() == ["model,time,pred,prob1", "A,20.000,1,0.8000", "B,21.000,1,0.6000"]
    finally:
        env.close()


def test_trigger_python_side_and_legacy_path():
    env = Env()
    try:
        reader = UnityTCPReader()
        reader.csv_file = io.StringIO()
        flag = lambda: gv.unity_marker_string_calibration
        now = time.time()

        def trial(i, mode_text):
            reader.process_message(f"Trial {i} START: {now + 10 * i:.3f} LABEL: 0{mode_text}", None)
            reader.process_message(f"Trial {i} END: {now + 10 * i + 5:.3f} LABEL: 0", None)

        # 舊版 Unity (START 沒有 MODE)：Python 不自己觸發，照舊跟著 Unity 的 calibration_start
        for i in range(4):
            trial(i, "")
        assert flag() == "" and gv.run_mode == ""
        reader.process_message(config.RECEIVE_UNITY_CALIBRATION_START_STR, None)
        assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
        gv.unity_marker_string_calibration = ""

        # 新版 + Adaptive：第 4、8 個 trial 的 END (編號 3、7) 觸發，其他不觸發；Unity 的 calibration_start 被忽略
        for i in range(3):
            trial(i, " MODE: Adaptive")
            assert flag() == ""
        trial(3, " MODE: Adaptive")
        assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
        assert reader.csv_file.getvalue().splitlines()[-1].startswith("Trial 3 END")   # 觸發時 END 行已經寫進 log
        gv.unity_marker_string_calibration = ""
        reader.process_message(config.RECEIVE_UNITY_CALIBRATION_START_STR, None)
        assert flag() == ""
        trial(4, " MODE: Adaptive"); assert flag() == ""
        trial(7, " MODE: Adaptive"); assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
        gv.unity_marker_string_calibration = ""

        # Static：不觸發 (除非現在是 Calibration 歌曲)
        trial(3, " MODE: Static"); assert flag() == ""
        gv.is_calibration_run = True
        trial(3, " MODE: Static"); assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
    finally:
        env.close()


def test_calibration_song_triggers_with_static_mode_after_stage_switches_to_mi():
    env = Env()
    try:
        gv.reset_run_state()
        reader = UnityTCPReader()
        reader.csv_file = io.StringIO()
        flag = lambda: gv.unity_marker_string_calibration
        now = time.time()
        # 真實順序 (log_20261013_145213.txt)：Calibration -> MI -> Trial N START (MODE: Static) ...
        reader.process_message(config.GameSTATE.Calibration.value, None)
        reader.process_message(config.GameSTATE.MI.value, None)
        assert gv.unity_marker_string_stage == config.GameSTATE.MI.value and gv.is_calibration_run
        for i in range(4):
            reader.process_message(f"Trial {i} START: {now + 10 * i:.3f} LABEL: 0 MODE: Static", None)
            assert flag() == ""
            reader.process_message(f"Trial {i} END: {now + 10 * i + 5:.3f} LABEL: 0", None)
            assert (flag() != "") == (i == 3)
        assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
        # 沒有 Calibration 訊息 (一般 MI 歌曲) + Static：不觸發
        gv.reset_run_state()
        assert not gv.is_calibration_run
        reader.process_message(config.GameSTATE.MI.value, None)
        for i in range(4):
            reader.process_message(f"Trial {i} START: {now + 10 * i:.3f} LABEL: 0 MODE: Static", None)
            reader.process_message(f"Trial {i} END: {now + 10 * i + 5:.3f} LABEL: 0", None)
        assert flag() == ""
    finally:
        env.close()


def test_swap_skipped_when_nothing_changed():
    env = Env()
    try:
        from main.game_state import Game
        calls = []
        pipe = types.SimpleNamespace(models_changed=False, record_swap=lambda t: calls.append("swap"))
        game = types.SimpleNamespace(reload_predictors=lambda: calls.append("reload"), ft_pipeline=pipe)
        gv.vote_on_at = time.time() - 1                                  # 旗標開著：沒變動就連等都不等
        t = time.time()
        Game.swap_in_new_models(game)
        assert calls == [] and time.time() - t < 1.0 and gv.last_swap_time is None
        gv.vote_on_at = None
        pipe.models_changed = True
        Game.swap_in_new_models(game)
        assert calls == ["reload", "swap"]

        # pipeline 端：A 對 B 對 (不微調、live 不變) -> False；A 錯 (A 微調) -> True；auto 切換 live -> True
        make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
        p = EEGFineTunePipeline()
        p.init_pipeline()
        env.new_block(1_000_000.0, 3, a_ok=(True,) * 4, b_ok=(True,) * 4)
        p.run_calibration()
        assert p.models_changed is False and abrows(env)[1][13] == ""    # swap_time 留空
        env.new_block(2_000_000.0, 4, a_ok=(False,) * 4, b_ok=(True,) * 4)
        p.run_calibration()
        assert p.models_changed is True
        env.new_block(3_000_000.0, 5, a_ok=(True,) * 4, b_ok=(True,) * 4)
        p.run_calibration()
        assert p.models_changed is False
        p.run_calibration()                                              # 沒有新資料 -> 沒變動
        assert p.models_changed is False
        gv.LIVE_MODE = "B"                                               # live 模型改變也算 (沒有任何微調)
        env.new_block(4_000_000.0, 6, a_ok=(True,) * 4, b_ok=(True,) * 4)
        p.run_calibration()
        assert gv.live_model == "B" and p.models_changed is True
    finally:
        env.close()


def test_predictions_csv_written_each_block_and_on_training():
    env = Env()
    try:
        from main.game_state import Game, GameState, MIState, CalibrationState
        gv.run_start_time = 0.0
        with gv.prediction_log_lock:
            gv.prediction_logs["A"].append((10.0, 1, 0.8))
        path = os.path.join(env.dir, "predictions.csv")
        for state in (MIState(), CalibrationState()):
            if os.path.exists(path):
                os.remove(path)
            order = []
            game = types.SimpleNamespace(
                saveDataAsPtFromPointer=lambda: order.append("save"),
                ft_pipeline=types.SimpleNamespace(run_calibration=lambda: order.append("calib")),
                swap_in_new_models=lambda: order.append("swap"),
                dump_predictions=lambda: (order.append("dump"), Game.dump_predictions()))
            gv.unity_marker_string_stage = ""
            gv.unity_marker_string_calibration = config.RECEIVE_UNITY_CALIBRATION_START_STR
            state.update(game)
            assert order == ["save", "calib", "dump", "swap"], order
            assert open(path).read().splitlines()[-1] == "A,10.000,1,0.8000"
        os.remove(path)                                                  # go_to_training 也要存
        order = []
        game = types.SimpleNamespace(saveDataAsPtFromPointer=lambda: order.append("save"), end_predictors=lambda: order.append("end"),
                                     dump_predictions=lambda: (order.append("dump"), Game.dump_predictions()),
                                     change_state=lambda s: order.append("state"))
        GameState.go_to_training(MIState(), game)
        assert order == ["save", "dump", "end", "state"] and os.path.exists(path)
    finally:
        env.close()


def test_start_from_choices_and_highest_plus_one_numbering():
    base = tempfile.mkdtemp()
    main_dir = tempfile.mkdtemp()
    saved = (getattr(config, "__REALTIME_BASE_FILE"), config.EEG_CHECKPOINT_MAIN_BASE_FILE, gv.subject, gv.session)
    touch = lambda d, names: [(os.makedirs(d, exist_ok=True), open(os.path.join(d, n), "w").close()) for n in names]
    try:
        setattr(config, "__REALTIME_BASE_FILE", base)
        config.EEG_CHECKPOINT_MAIN_BASE_FILE = main_dir + os.sep
        gv.subject = "01"
        sub = os.path.join(base, "sub01")
        touch(os.path.join(sub, "ses1", "checkpoints"), ["cA_001.pth", "cA_009.pth", "cB_012.pth", "cA_003.pth"])
        touch(os.path.join(sub, "ses2", "checkpoints"), ["cA_001.pth"])
        touch(os.path.join(sub, "ses4", "checkpoints"), [])
        gv.session = "3"                                               # 這個 session 還沒有 checkpoint
        base_a, base_b = os.path.join(main_dir, "cA_000.pth"), os.path.join(main_dir, "cB_000.pth")
        assert resolve_start_checkpoint("cA_", "base") == base_a and resolve_start_checkpoint("cB_", "base") == base_b
        assert resolve_start_checkpoint("cA_", "session") == base_a and resolve_start_checkpoint("cB_", "session") == base_b
        assert resolve_start_checkpoint("cA_", "subject") == os.path.join(sub, "ses2", "checkpoints", "cA_001.pth")  # 有 A 的最高 session
        assert resolve_start_checkpoint("cB_", "subject") == os.path.join(sub, "ses1", "checkpoints", "cB_012.pth")  # B 只有 ses1
        gv.session = "1"                                               # 同一個 session：A、B 編號不同
        assert resolve_start_checkpoint("cA_", "session") == os.path.join(sub, "ses1", "checkpoints", "cA_009.pth")
        assert resolve_start_checkpoint("cB_", "session") == os.path.join(sub, "ses1", "checkpoints", "cB_012.pth")
        assert resolve_start_checkpoint("cA_", "base") == base_a

        d = os.path.join(base, "numbering")
        assert get_next_version_path(os.path.join(d, "x.pth"), "cA_") == os.path.join(d, "cA_001.pth")   # 空資料夾從 001 開始
        touch(d, ["cA_001.pth", "cA_002.pth", "cA_003.pth", "cB_007.pth"])
        os.remove(os.path.join(d, "cA_002.pth"))
        assert get_next_version_path(os.path.join(d, "x.pth"), "cA_") == os.path.join(d, "cA_004.pth")   # 最大編號 + 1，不補洞
        assert get_next_version_path(os.path.join(d, "x.pth"), "cB_") == os.path.join(d, "cB_008.pth")
    finally:
        setattr(config, "__REALTIME_BASE_FILE", saved[0])
        config.EEG_CHECKPOINT_MAIN_BASE_FILE, gv.subject, gv.session = saved[1:]


def test_predictor_subset_channels_and_only_live_broadcasts():
    env = Env()
    try:
        make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
        gv.eeg_buffer = np.random.default_rng(0).standard_normal((config.N_CHANNELS, 500))
        sent = []
        tcp = type("T", (), {"broadcast": lambda self, m: sent.append(m)})()
        a = EEGPredictor(tcp, name="A", live=True)
        b = EEGPredictor(tcp, name="B", live=False)
        assert len(a.channel_pos) == 22 and len(b.channel_pos) == config.N_CHANNELS
        a.start_predict_eeg_thread(); b.start_predict_eeg_thread()
        time.sleep(1.5)
        a.end_predict_eeg_thread(); b.end_predict_eeg_thread()
        assert len(gv.prediction_logs["A"]) > 5 and len(gv.prediction_logs["B"]) > 5
        assert len(sent) == len(gv.prediction_logs["A"])                    # 只有 live 的 A 送給 Unity
        assert all(m in ("0", "1") for m in sent)
        # 切換 live
        a.live, b.live = False, True
        sent.clear()
        n_a = len(gv.prediction_logs["A"]); n_b = len(gv.prediction_logs["B"])
        a.start_predict_eeg_thread(); b.start_predict_eeg_thread()
        time.sleep(1.0)
        a.end_predict_eeg_thread(); b.end_predict_eeg_thread()
        assert len(sent) == len(gv.prediction_logs["B"]) - n_b > 0
    finally:
        env.close()


def test_channel_mismatch_raises_and_does_not_overwrite():
    env = Env()
    try:
        # 32 通道權重卻標成 22 通道 -> 載入失敗；檔案不能被改寫
        make_ckpt(gv.A_CHECKPOINT, 32)
        ck = torch.load(gv.A_CHECKPOINT, map_location="cpu", weights_only=False); ck["n_chans"] = 22; torch.save(ck, gv.A_CHECKPOINT)
        before = hashlib.md5(open(gv.A_CHECKPOINT, "rb").read()).hexdigest()
        try:
            EEGPredictor(None, name="A")
            raise AssertionError("應該要報錯")
        except RuntimeError:
            pass
        assert hashlib.md5(open(gv.A_CHECKPOINT, "rb").read()).hexdigest() == before
    finally:
        env.close()


def test_switch_streak_is_read_from_config():
    env = Env()
    old = config.SWITCH_STREAK
    try:
        gv.LIVE_MODE = "auto"
        S = lambda a, b: {"A": (a, 4), "B": (b, 4)}
        config.SWITCH_STREAK = 3
        p = EEGFineTunePipeline()
        for _ in range(2):
            p._decide_live(S(1, 3))
        assert gv.live_model == "A" and p.streak == 2        # 2 次還不夠
        p._decide_live(S(1, 3))
        assert gv.live_model == "B"                          # 第 3 次才切換
        config.SWITCH_STREAK = 1
        p._decide_live(S(4, 0)); assert gv.live_model == "A"  # 1 次就切回 A
    finally:
        config.SWITCH_STREAK = old
        env.close()


def _trial_lines(idx, t0):
    return [f"Trial {idx} START: {t0:.3f} LABEL: 1 MODE: Adaptive", f"Trial {idx} CUT: {t0 + 1:.3f}",
            f"Trial {idx} END: {t0 + 2:.3f} LABEL: 1"]


def test_log_pointer_holds_back_running_trial():
    path = os.path.join(tempfile.mkdtemp(), "log.txt")
    lines = []
    for i in range(4):
        lines += _trial_lines(i, 100 + 5 * i)
    lines.append("Trial 4 START: 120.000 LABEL: 1 MODE: Adaptive")
    open(path, "w").write("\n".join(lines) + "\n")
    trials, ptr, n = readLogFromPointer(path, 0)
    assert n == 13 and ptr == 12 and sorted(t for t, v in trials.items() if "end" in v) == [0, 1, 2, 3]
    with open(path, "a") as f:
        f.write("\n".join(["Trial 4 CUT: 121.000", "Trial 4 END: 122.000 LABEL: 1"] + sum((_trial_lines(i, 125 + 5 * (i - 5)) for i in range(5, 8)), [])) + "\n")
    trials, ptr2, n2 = readLogFromPointer(path, ptr)
    assert sorted(trials) == [4, 5, 6, 7] and all("end" in v for v in trials.values()), trials
    assert ptr2 == ptr + n2
    trials, ptr3, n3 = readLogFromPointer(path, ptr2)
    assert trials == {} and n3 == 0 and ptr3 == ptr2


def test_log_pointer_abandoned_trial_not_held():
    path = os.path.join(tempfile.mkdtemp(), "log.txt")
    lines = _trial_lines(3, 100) + ["Trial 4 START: 105.000 LABEL: 1 MODE: Adaptive"]
    for i in range(5, 8):
        lines += _trial_lines(i, 110 + 5 * (i - 5))
    open(path, "w").write("\n".join(lines) + "\n")
    trials, ptr, n = readLogFromPointer(path, 0)
    assert ptr == n == len(lines) and "end" not in trials[4]
    trials, _, n2 = readLogFromPointer(path, ptr)
    assert trials == {} and n2 == 0


def test_lobby_clears_block_flag_and_states_skip_lobby_log():
    from main import game_state as gs
    old_flag, old_stage = gv.unity_marker_string_calibration, gv.unity_marker_string_stage
    path = os.path.join(tempfile.mkdtemp(), "log.txt")
    lines = []
    for i in range(3):
        lines += _trial_lines(i, 100 + 5 * i)
    open(path, "w").write("\n".join(lines) + "\n")
    game = types.SimpleNamespace(
        logLinePointer=0, unity_reader=types.SimpleNamespace(filename=path, flushLog=lambda: None),
        ft_pipeline=types.SimpleNamespace(init_pipeline=lambda: None))
    game.skipLogToEnd = lambda: gs.Game.skipLogToEnd(game)
    try:
        gv.unity_marker_string_stage = ""
        gv.unity_marker_string_calibration = config.RECEIVE_UNITY_CALIBRATION_START_STR
        buf = io.StringIO()
        old_out, sys.stdout = sys.stdout, buf
        try:
            gs.LobbyState().update(game)
            assert gv.unity_marker_string_calibration == ""
            assert "Trial data received in Lobby" in buf.getvalue()
            gs.MIState().start(game)
            assert "9" in buf.getvalue()
        finally:
            sys.stdout = old_out
        assert game.logLinePointer == len(lines)
        with open(path, "a") as f:
            f.write("\n".join(_trial_lines(3, 130)) + "\n")
        trials, _, _ = readLogFromPointer(path, game.logLinePointer)
        assert sorted(trials) == [3]
        game.logLinePointer = 0
        gs.CalibrationState().start(game)
        assert game.logLinePointer == len(lines) + 3
    finally:
        gv.unity_marker_string_calibration, gv.unity_marker_string_stage = old_flag, old_stage


# ---- SPRT gate (online_adaptation.finetune_gate = "sprt") ----
T0 = 1_000_000.0


class SprtGame:
    """main.game_state.Game 的替身：真的 log 檔 + 真的 run_calibration；存檔只把 log pointer 之後的完整 trial 存成 .pt (沒有 EEG csv)"""

    def __init__(self, env, pipe, swap_at=None):
        from main.EEG.sprt_finetune import SprtTicker
        self.env, self.ft_pipeline = env, pipe
        self.logLinePointer = 0
        self.unity_reader = types.SimpleNamespace(filename=os.path.join(env.dir, "log.txt"), flushLog=lambda: None)
        open(self.unity_reader.filename, "w").close()
        self.sprt_ticker = SprtTicker()
        self.saves, self.swap_at = [], swap_at or (lambda: time.time())

    def trial(self, idx, a_ok, b_ok, tick=True):
        """寫一個 trial 的 START / CUT 或 WRONG_CUT / END 和兩個模型的假預測；tick=True 時像 main loop 一樣處理"""
        start, label = T0 + 10 * idx, idx % 2
        vote = start + 4.25
        lines = [f"Trial {idx} START: {start:.3f} LABEL: {label} MODE: Adaptive",
                 f"Trial {idx} {'CUT' if a_ok else 'WRONG_CUT'}: {vote:.3f}", f"Trial {idx} END: {start + 4.95:.3f} LABEL: {label}"]
        with open(self.unity_reader.filename, "a") as f:
            f.write("\n".join(lines) + "\n")
        with gv.prediction_log_lock:
            for k in range(120):
                pa, pb = (label if a_ok else 1 - label), (label if b_ok else 1 - label)
                gv.prediction_logs["A"].append((start + 0.035 * k, pa, 0.9 if pa else 0.1))
                gv.prediction_logs["B"].append((start + 0.035 * k, pb, 0.9 if pb else 0.1))
        if tick:
            self.sprt_ticker.tick(self)

    def saveDataAsPtFromPointer(self):
        trials, ptr, _ = readLogFromPointer(self.unity_reader.filename, self.logLinePointer)
        ended = sorted(i for i, t in trials.items() if "start" in t and "end" in t)
        if not ended:
            return
        n = len(ended)
        x = np.random.default_rng(ended[0]).standard_normal((n, config.N_CHANNELS, 1000))
        y = np.array([trials[i]["label"] for i in ended])
        fails = np.array(["wrong_cut" in trials[i] for i in ended])
        path = os.path.join(self.env.dir, f"data_sprt_{ended[0]}.pt")
        torch.save({"x_data": x, "y_data": y, "failures": fails, "trial_ids": np.array(ended)}, path)
        gv.train_np_data.append(path)
        gv.last_block_trials = {i: trials[i] for i in ended}
        self.logLinePointer = ptr
        self.saves.append(ended)

    def dump_predictions(self):
        pass

    def swap_in_new_models(self):
        if self.ft_pipeline.models_changed:
            gv.last_swap_time = self.swap_at()
            self.ft_pipeline.record_swap(gv.last_swap_time)


def sprt_rows(env):
    return [r.split(",") for r in open(os.path.join(env.dir, "sprt_trials.csv")).read().splitlines()]


def sprt_env():
    env = Env()
    config.FINETUNE_GATE = "sprt"
    make_ckpt(gv.B_CHECKPOINT, config.N_CHANNELS, seed=1)
    gv.last_swap_time = None
    pipe = EEGFineTunePipeline()
    pipe.init_pipeline()
    return env, pipe


def test_sprt_mid_block_finetunes_only_the_model_that_crossed():
    env, pipe = sprt_env()
    try:
        game = SprtGame(env, pipe, swap_at=lambda: T0 + 25)      # 新 A 在 trial 2 投票 (T0+24.25) 之後、trial 3 之前載入
        game.trial(0, False, True); game.trial(1, False, True)
        assert game.saves == [] and ckpt_files(env) == []         # A -1.39，還沒跨線：不存檔、不微調
        game.trial(2, False, True)                                # A -2.08 <= -1.56 -> 立刻微調 A
        assert game.saves == [[0, 1, 2]] and ckpt_files(env) == ["cA_001.pth"]
        row = abrows(env)[1]
        h = EEGFineTunePipeline.HEADER
        assert row[h.index("trigger")] == "sprt" and row[h.index("n_trials")] == "3" and row[10:12] == ["1", "0"]
        assert row[h.index("sprt_A")] == "0.000" and row[h.index("swap_time")] == f"{T0 + 25:.3f}"
        assert game.sprt_ticker.gates["A"].score == 0.0 and abs(game.sprt_ticker.gates["B"].score - 3 * 0.405465) < 1e-5
        game.trial(3, True, True)                                 # block 結束：存檔但不微調，兩個 buffer 都長大
        assert game.saves == [[0, 1, 2], [3]] and ckpt_files(env) == ["cA_001.pth"]
        row = abrows(env)[2]
        assert row[h.index("trigger")] == "block" and row[10:12] == ["0", "0"] and row[h.index("n_trials")] == "1"
        assert buffer_total(pipe.slots["A"])[0] == buffer_total(pipe.slots["B"])[0] == 24
        assert sum(int(r[1]) for r in abrows(env)[1:]) == 4       # n_trials 加起來 = trial 數
        rows = sprt_rows(env)
        assert rows[0] == ["trial", "model", "live", "correct", "step", "score", "decision", "p0", "p1", "alpha", "beta"]
        assert len(rows) == 1 + 8 and rows[5][:7] == ["2", "A", "A", "0", "-0.693", "-2.079", "finetune"]
        assert rows[7][:4] == ["3", "A", "A", "1"] and rows[7][6] == ""   # trial 3 是新 A 的 trial
    finally:
        env.close()


def test_sprt_good_resets_and_trials_before_swap_are_stale():
    env, pipe = sprt_env()
    try:
        game = SprtGame(env, pipe, swap_at=lambda: T0 + 45)      # 載入很慢：trial 3、4 還是舊 A 投的票
        for i in range(3):
            game.trial(i, False, True, tick=False)
        game.sprt_ticker.tick(game)                               # main loop 卡住後一次處理 3 個 trial
        assert game.saves == [[0, 1, 2]] and ckpt_files(env) == ["cA_001.pth"]
        game.trial(3, False, True); game.trial(4, False, True)
        rows = [r for r in sprt_rows(env)[1:] if r[1] == "A"]
        assert [r[6] for r in rows] == ["", "", "finetune", "stale", "stale"]
        assert game.sprt_ticker.gates["A"].score == 0.0 and ckpt_files(env) == ["cA_001.pth"]
        game.trial(5, True, True); game.trial(6, True, True)      # B 第 7 個正確 -> good，分數歸零，不微調
        b = [r for r in sprt_rows(env)[1:] if r[1] == "B"]
        assert b[6][6] == "good" and game.sprt_ticker.gates["B"].score == 0.0
        assert abs(game.sprt_ticker.gates["A"].score - 2 * 0.405465) < 1e-5
        assert "cB_001.pth" not in ckpt_files(env)
    finally:
        env.close()


def test_sprt_scores_reset_on_new_run():
    env, pipe = sprt_env()
    try:
        game = SprtGame(env, pipe)
        game.trial(0, False, True); game.trial(1, False, True)
        assert abs(game.sprt_ticker.gates["A"].score + 1.386294) < 1e-5
        gv.runCount += 1                                          # 新 run：新 log 檔，pointer 從 0 開始
        game.unity_reader.filename = os.path.join(env.dir, "log_run2.txt")
        open(game.unity_reader.filename, "w").close()
        game.logLinePointer = 0
        game.trial(0, False, True)
        assert abs(game.sprt_ticker.gates["A"].score + 0.693147) < 1e-5 and ckpt_files(env) == []
        assert game.sprt_ticker.score_pointer == 3
    finally:
        env.close()


def test_sprt_switching_still_decided_on_four_trial_blocks():
    env, pipe = sprt_env()
    try:
        gv.LIVE_MODE = "auto"
        game = SprtGame(env, pipe)
        a_pattern = (True, False, True, False)                    # A 2/4，不會跨 SPRT 線；B 全對 (第 7 個 -> good)
        for i in range(8):
            game.trial(i, a_pattern[i % 4], True)
            if i == 3:
                assert pipe.streak == 1 and gv.live_model == "A"
        assert gv.live_model == "B" and pipe.streak == 0          # 連兩個 block B 較好 -> 切換
        rows = abrows(env)
        assert len(rows) == 3 and [r[1] for r in rows[1:]] == ["4", "4"] and ckpt_files(env) == []
    finally:
        env.close()


def test_sprt_trigger_every_trial_and_calibration_song():
    env = Env()
    try:
        config.FINETUNE_GATE = "sprt"
        reader = UnityTCPReader()
        reader.csv_file = io.StringIO()
        flag = lambda: gv.unity_marker_string_calibration
        now = time.time()
        for i in range(3):                                        # Adaptive：每個 trial 都觸發
            reader.process_message(f"Trial {i} START: {now + 10 * i:.3f} LABEL: 0 MODE: Adaptive", None)
            reader.process_message(f"Trial {i} END: {now + 10 * i + 5:.3f} LABEL: 0", None)
            assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
            gv.unity_marker_string_calibration = ""
        reader.process_message(f"Trial 5 START: {now + 50:.3f} LABEL: 0 MODE: Static", None)
        reader.process_message(f"Trial 5 END: {now + 55:.3f} LABEL: 0", None)
        assert flag() == ""                                       # 一般 Static 歌曲：不觸發
        gv.is_calibration_run = True                              # Calibration 歌曲：一樣每個 trial
        reader.process_message(f"Trial 6 END: {now + 65:.3f} LABEL: 0", None)
        assert flag() == config.RECEIVE_UNITY_CALIBRATION_START_STR
    finally:
        env.close()


def test_sprt_states_call_ticker_and_block_states_do_not():
    env = Env()
    try:
        from main.game_state import MIState, CalibrationState
        old_path = ["save", "calib", "dump", "swap"]
        # 舊版 Unity (沒有 MODE，例如 VR 專案的 Calibration2 會等 calibration done)：SPRT 設定下也走舊路徑
        for gate, mode, want in (("sprt", "Adaptive", ["tick"]), ("sprt", "", old_path), ("block", "Adaptive", old_path)):
            config.FINETUNE_GATE = gate
            for state in (MIState(), CalibrationState()):
                order = []
                game = types.SimpleNamespace(
                    sprt_ticker=types.SimpleNamespace(tick=lambda g: order.append("tick")),
                    saveDataAsPtFromPointer=lambda: order.append("save"),
                    ft_pipeline=types.SimpleNamespace(run_calibration=lambda: order.append("calib")),
                    swap_in_new_models=lambda: order.append("swap"), dump_predictions=lambda: order.append("dump"))
                gv.unity_marker_string_stage, gv.run_mode = "", mode
                gv.unity_marker_string_calibration = config.RECEIVE_UNITY_CALIBRATION_START_STR
                state.update(game)
                assert order == want and gv.unity_marker_string_calibration == "", (gate, mode, order)
    finally:
        env.close()


def test_sprt_real_save_keeps_a_single_trial():
    """Game.saveDataAsPtFromPointer with a real EEG csv: SPRT mode saves one trial (3 log lines); block mode still needs >= 7 lines."""
    from main.game_state import Game
    from main.Utils.file_pointer_reader import extractRecordEpoch
    env = Env()
    try:
        csv_path, log_path = os.path.join(env.dir, "eeg.csv"), os.path.join(env.dir, "log.txt")
        header = ["# header"] * 11
        header[3] = "Record datetime: 2026-11-16 10:00:00.000000"
        fs, n_cols = 500, max(config.channel_index) + 1
        with open(csv_path, "w") as f:
            f.write("\n".join(header) + "\n")
            rng = np.random.default_rng(0)
            for i in range(30 * fs):
                f.write(",".join([f"{i / fs:.3f}"] + [f"{v:.2f}" for v in rng.standard_normal(n_cols - 1)]) + "\n")
        epoch = extractRecordEpoch(csv_path)
        trial_lines = lambda idx, rel: [f"Trial {idx} START: {epoch + rel:.3f} LABEL: {idx % 2} MODE: Adaptive",
                                        f"Trial {idx} CUT: {epoch + rel + 4.25:.3f}", f"Trial {idx} END: {epoch + rel + 4.95:.3f} LABEL: {idx % 2}"]
        open(log_path, "w").write("\n".join(trial_lines(0, 2.0)) + "\n")

        game = Game.__new__(Game)
        game.eeg_reader = types.SimpleNamespace(filename=csv_path, flushCsv=lambda: None)
        game.unity_reader = types.SimpleNamespace(filename=log_path, flushLog=lambda: None)
        game.csvLinePointer = game.logLinePointer = 0
        gv.run_mode = "Adaptive"

        config.FINETUNE_GATE = "block"
        game.saveDataAsPtFromPointer()
        assert gv.train_np_data == [] and game.logLinePointer == 0           # 3 行 < 7：block 模式照舊略過
        config.FINETUNE_GATE = "sprt"
        game.saveDataAsPtFromPointer()
        assert len(gv.train_np_data) == 1 and game.logLinePointer == 3
        d = torch.load(gv.train_np_data[0], map_location="cpu", weights_only=False)
        assert d["x_data"].shape[:2] == (1, config.N_CHANNELS) and list(d["trial_ids"]) == [0] and sorted(gv.last_block_trials) == [0]
        with open(log_path, "a") as f:
            f.write("\n".join(trial_lines(1, 10.0)) + "\n")
        game.saveDataAsPtFromPointer()
        d = torch.load(gv.train_np_data[1], map_location="cpu", weights_only=False)
        assert len(gv.train_np_data) == 2 and list(d["trial_ids"]) == [1] and game.logLinePointer == 6
    finally:
        env.close()


def test_sprt_config_keys_read():
    adapt = config.JSON_CONFIG.get("online_adaptation", {})
    assert adapt.get("finetune_gate") == "sprt" and config.FINETUNE_GATE == "sprt"
    assert (config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA) == (0.65, 0.85, 0.05, 0.2)


def test_tcp_watchdog_waiting_then_silent_then_lost():
    from main.Utils.TCPServer import TCPServer
    server, out = TCPServer(), []
    server.start_watchdog(check_s=0.02, grace_s=0.1, repeat_s=0.3, log=out.append)
    time.sleep(0.2)
    assert out and all("Waiting for Unity connection" in m for m in out), out
    server.clients.append(object())
    server.ever_connected = True
    time.sleep(0.1)
    n = len(out)
    time.sleep(0.5)
    assert len(out) == n                                   # 有 client 就不再警告
    server.clients.clear()
    time.sleep(0.25)
    assert "Connection to unity lost" in out[-1], out
    server.stop_event.set()


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t:.1f}s)")
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
