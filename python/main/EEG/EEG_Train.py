"""
VR-BCI 線上微調與模型訓練模組 (EEG Training & Fine-Tuning Pipeline)
用於 Calibration 場景蒐集資料後進行即時在線微調 (Fine-tune) 與重新訓練。
"""

import os
import sys
import csv
import time
import shutil
import random
import threading
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# 自動路徑解析，允許直接以 python python/main/EEG/EEG_Train.py 執行
_current_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.dirname(_current_dir)
_python_dir = os.path.dirname(_main_dir)
for p in [_python_dir, _main_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# UTF-8 輸出防護
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import main.Utils.config as config
import main.Utils.preprocess as preprocess
import main.Utils.global_value as global_value
from main.EEG.data_process_np import EEGDataLoader
from main.EEG.MI_train import BraindecodeTrainer, OnlineCalibrationTrainer, ReplayBuffer
from main.Utils.some_functions import rename_file_with_time, get_next_version_path
from main.Utils.trial_scoring import score_block


class EEGSelfDataLoader:
    """受試者自身 EEG 與標籤數據載入與存檔器"""
    def __init__(self, file_paths, log_paths, channel_index):
        self.file_paths = file_paths
        self.log_paths = log_paths
        self.channel_index = channel_index
        self.x_data = None
        self.y_data = None
        self.failures_data = None

    def load_data(self):
        loader = EEGDataLoader(
            file_paths=self.file_paths,
            log_paths=self.log_paths,
            channel_index=self.channel_index
        )
        loader.load_and_preprocess_data()
        self.x_data, self.y_data, self.failures_data = loader.get_eeg_trial_channel_sample_np()

    def get_data(self):
        return self.x_data, self.y_data, self.failures_data

    def save_as_pt_with_name(self, name):
        """將資料依標籤排序平衡後存成 .pt 檔案"""
        self.load_data()
        self.x_data, self.y_data, self.failures_data = arrange_by_label(
            self.x_data, self.y_data, self.failures_data
        )
        torch.save({
            'x_data': self.x_data,
            'y_data': self.y_data,
            'failures': self.failures_data
        }, name)
        print(f"✅ 模型資料已儲存為: {name}")

    def save_as_pt(self):
        """自動生成時間戳檔名存檔，並將路徑登錄至全域訓練資料清單"""
        data_name = rename_file_with_time(config.getRunPtFilename())
        global_value.train_np_data.append(data_name)
        self.save_as_pt_with_name(data_name)


def arrange_by_label(x, y, f):
    """
    將資料依 y 標籤按照 0 1 0 1 交錯排列以平衡訓練集；若不平衡則將配對放後面，剩下的放前面。
    :param x: (trial, channel, sample)
    :param y: (trial,)
    :param f: (trial,) failures 標籤
    :return: 排序平衡後的 (x_sorted, y_sorted, f_sorted)
    """
    label_0_idx = np.where(y == 0)[0]
    label_1_idx = np.where(y == 1)[0]

    label_0_rev = label_0_idx[::-1]
    label_1_rev = label_1_idx[::-1]

    num_pairs = min(len(label_0_rev), len(label_1_rev))
    paired_0 = label_0_rev[:num_pairs][::-1]
    paired_1 = label_1_rev[:num_pairs][::-1]

    # 交錯排列：0, 1, 0, 1...
    interleaved_idx = np.empty(num_pairs * 2, dtype=int)
    interleaved_idx[0::2] = paired_0
    interleaved_idx[1::2] = paired_1

    # 無法配對的剩餘索引放在最前面
    remaining_0 = label_0_rev[num_pairs:][::-1]
    remaining_1 = label_1_rev[num_pairs:][::-1]
    remaining_idx = np.concatenate((remaining_0, remaining_1))

    final_idx = np.concatenate((remaining_idx, interleaved_idx))
    return x[final_idx], y[final_idx], f[final_idx]


def prepare_calibration_dataset(data_x_list, data_y_list, data_failures, segment_len=500, stride=100):
    """將校正資料以滑動視窗切片並前處理，轉換為含失敗權重之 TensorDataset"""
    augmented_segments = []
    augmented_labels = []
    augmented_failures = []

    for idx in range(len(data_x_list)):
        x_np = np.transpose(data_x_list[idx], (0, 2, 1))  # (T, C, S) -> (T, S, C)
        y_np = data_y_list[idx]
        f_np = data_failures[idx]

        for i, s in enumerate(x_np):
            if s.shape[0] < segment_len:
                continue
            label = y_np[i]
            failure = f_np[i]

            for start in range(0, s.shape[0] - segment_len + 1, stride):
                data = s[start:start + segment_len]
                window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))

                augmented_segments.append(window)
                augmented_labels.append(label)
                augmented_failures.append(failure)

    if not augmented_segments:
        return None

    data_x = np.transpose(np.stack(augmented_segments), (0, 2, 1))  # (trial, channel, sample)
    X = torch.tensor(data_x).unsqueeze(1).float()  # (T, 1, C, N)
    y_tensor = torch.tensor(augmented_labels).long()
    n_class = config.N_Class if hasattr(config, 'N_Class') else 2
    y = F.one_hot(y_tensor, num_classes=n_class).float()
    f = torch.tensor(augmented_failures).float()

    return TensorDataset(X, y, f)


def prepare_datasets(data_x_list, data_y_list, valid_ratio=0.3, segment_len=500, stride=100):
    """
    將資料集列表按比例劃分為訓練集與驗證集，並進行帶通濾波與去均值處理
    """
    augmented_segments_valid, augmented_labels_valid = [], []
    augmented_segments_train, augmented_labels_train = [], []

    for idx in range(len(data_x_list)):
        x_np = np.transpose(data_x_list[idx], (0, 2, 1))  # (T, C, S) -> (T, S, C)
        y_np = data_y_list[idx]
        valid_num = int(len(x_np) * valid_ratio)

        for i, s in enumerate(x_np):
            if s.shape[0] < segment_len:
                continue
            label = y_np[i]

            for start in range(0, s.shape[0] - segment_len + 1, stride):
                data = s[start:start + segment_len]
                window = preprocess.bandpass(data - np.mean(data, axis=1, keepdims=True))

                if i < valid_num:
                    augmented_segments_valid.append(window)
                    augmented_labels_valid.append(label)
                else:
                    augmented_segments_train.append(window)
                    augmented_labels_train.append(label)

    def _to_dataset(seg_list, lbl_list):
        data_x = np.transpose(np.stack(seg_list), (0, 2, 1))
        X = torch.tensor(data_x).unsqueeze(1).float()
        y = F.one_hot(torch.tensor(lbl_list).long(), num_classes=config.N_Class).float()
        return TensorDataset(X, y)

    train_ds = _to_dataset(augmented_segments_train, augmented_labels_train)
    if valid_ratio > 0 and augmented_segments_valid:
        val_ds = _to_dataset(augmented_segments_valid, augmented_labels_valid)
        return train_ds, val_ds
    return train_ds, None


# 向後相容別名
prepare_datasets_v2 = prepare_datasets


def run_braindecode_training(model_class, dataset, dataset_valid, epochs=100, batch_size=16, lr=1e-4,
                             freeze_layers=False, seed=42, params=None, ft=False, tcp_server=None):
    """執行 Braindecode 模型訓練與 Checkpoint 儲存"""
    if params is None:
        raise RuntimeError(f"請設定模型 {model_class} 的超參數 params")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = BraindecodeTrainer(
        dataset=dataset,
        val_dataset=dataset_valid,
        model_class=model_class,
        model_kwargs=params,
        batch_size=batch_size,
        num_epochs=epochs,
        lr=lr,
        ft=ft
    )

    name = "ft" if ft else "train"
    if ft:
        trainer.load_checkpoint(global_value.NOW_TRAINED_CHECKPOINT)

    hist = trainer.train(freeze_layer=freeze_layers, tcp_server=tcp_server, patience=30)
    best_loss_epoch = int(np.argmin(hist["val_loss"]))

    if config.adaption_use_val:
        best_loss_path = os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{name}-best.pth")
    else:
        best_loss_path = os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{name}-epoch{best_loss_epoch}.pth")

    save_path = get_next_version_path(global_value.NOW_TRAINED_CHECKPOINT)
    shutil.copyfile(best_loss_path, save_path)
    global_value.NOW_TRAINED_CHECKPOINT = save_path

    print(f"\n{config.TAGS.INFO.value} 訓練完成，最佳模型已複製至: {save_path}")


class _ModelSlot:
    """雙模型中的一個 (A 或 B)：各自的 checkpoint 鏈、trainer、replay buffer、暫存 checkpoint tag"""

    def __init__(self, name, prefix, seed):
        self.name, self.prefix, self.seed = name, prefix, seed
        self.buffer = ReplayBuffer()
        self.trainer = None
        self.save_path = None
        self.ft_seconds = 0.0
        self.n_chans = config.N_CHANNELS
        self.positions = list(range(config.N_CHANNELS))

    @property
    def base_path(self):  # 這個模型目前最新的 checkpoint，也是下一次微調的起點
        return getattr(global_value, f"{self.name}_CHECKPOINT")

    def init(self, batch_size, epochs, lr):
        if os.path.exists(self.base_path):
            self.n_chans = int(torch.load(self.base_path, map_location="cpu", weights_only=False).get("n_chans", config.N_CHANNELS))
        self.positions = config.channel_positions(self.n_chans)
        self.trainer = OnlineCalibrationTrainer(
            model_class=config.USE_MODEL,
            model_kwargs=dict(config.LOAD_MODEL_PARAM, channels=self.n_chans),
            batch_size=batch_size, num_epochs=epochs, lr=lr, ft=True,
            tag=f"ft{self.name}", seed=self.seed
        )

    def finetune(self, train_data, train, failures=None):
        """這個 block 的 window 一定加入自己的 replay buffer；train 為 True 才訓練並存成新的 checkpoint"""
        x = train_data['x_data'][:, self.positions, :]  # (trial, channel, sample) 取這個模型的通道子集
        failures = train_data['failures'] if failures is None else failures
        self.buffer.add(prepare_calibration_dataset([x], [train_data['y_data']], [failures]))
        self.ft_seconds = 0.0
        if not train:
            return False

        start = time.perf_counter()
        self.save_path = get_next_version_path(os.path.join(config.getSessionCheckpointDir(), "x.pth"), self.prefix)  # 最大編號 + 1
        self.trainer.load_checkpoint(self.base_path)
        loss = self.trainer.train(self.buffer.train_set(), self.buffer.val_set())
        print(f"{config.TAGS.INFO.value} [{self.name}] 線上微調完成，最新 Loss: {loss:.4f}")

        tag = self.trainer.tag
        best_name = f"{tag}-best.pth" if config.adaption_use_val else f"{tag}-epoch{self.trainer.num_epochs - 1}.pth"
        checkpoint = torch.load(os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, best_name), map_location="cpu", weights_only=False)
        checkpoint['n_chans'] = self.n_chans  # predictor 靠這個決定用哪些通道
        torch.save(checkpoint, self.save_path)
        setattr(global_value, f"{self.name}_CHECKPOINT", self.save_path)
        self.ft_seconds = time.perf_counter() - start
        print(f"{config.TAGS.INFO.value} [{self.name}] 校正微調完成，更新模型至: {self.save_path}")
        return True


class EEGFineTunePipeline:
    """線上 Calibration 快速微調 Pipeline（Replay Buffer 增量學習）：A、B 兩個模型平行、各自判斷要不要微調，只有 live 的送給 Unity"""

    HEADER = ["block", "n_trials", "correct_A", "correct_B", "streak", "live", "ckpt_A", "ckpt_B", "failed", "mode",
              "ft_A", "ft_B", "ft_seconds", "swap_time", "trigger", "sprt_A", "sprt_B"]

    def __init__(self, tcp_server=None):
        self.batch_size = config.adaption_batch_size
        self.channel_index = config.channel_index
        self.lr = config.adaption_learning_rate
        self.epochs = config.adaption_epochs
        self.strides = 100
        self.segment_len = 500
        self.seed = 42
        self.ft = True
        self.tcp_server = tcp_server
        self.slots = {
            "A": _ModelSlot("A", "cA_", 42),
            "B": _ModelSlot("B", "cB_", 43),
        }
        self.streak = 0
        self.block = 0
        self.run_counts = {"A": [0, 0], "B": [0, 0]}  # 本 run 的 [Unity 式正確 trial 數, 計分 trial 數]，global_value.runCount 改變時重設
        self.counter_run = None
        self.models_changed = False  # 最近一次 run_calibration 有沒有任何模型被微調或 live 模型改變 (沒有的話不用重新載入權重)
        self.last_row = None  # 最近一個 block 寫進 ab_blocks.csv 的位置，載入新權重後補上 swap_time
        self.last_trained = {}  # 最近一次 run_calibration 各模型有沒有微調成功 (SPRT 用來判斷哪些 trial 是舊模型的)

    def init_pipeline(self):
        """初始化兩個模型的在線微調器"""
        for slot in self.slots.values():
            slot.init(self.batch_size, self.epochs, self.lr)

    def run_calibration(self, seed=42, sprt=None):
        """sprt=None：舊的 block gate。SPRT 模式由 sprt_finetune 傳入
        {"gates": {name: bool}, "trigger": "block"|"sprt", "switch_scores": scores 或 None, "scores": {name: float}}"""
        self.models_changed = False
        self.last_trained = {}
        trials, global_value.last_block_trials = global_value.last_block_trials, None
        if not trials or not global_value.train_np_data:  # 這次沒有存到新資料（舊資料不重複訓練、不重複計分）
            print(f"{config.TAGS.WARNING.value} 無新的校正資料，略過微調")
        else:
            self._calibrate(trials, seed, sprt)

        if self.tcp_server:
            time.sleep(0.5)
            self.tcp_server.broadcast(config.CALIBRATION_FINISH_STR)

    def _calibrate(self, trials, seed, sprt=None):
        if any(slot.trainer is None for slot in self.slots.values()):
            self.init_pipeline()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        latest_data_path = global_value.train_np_data[-1]
        print(f"{config.TAGS.INFO.value} 載入最新校正資料: {latest_data_path}")
        train_data = torch.load(latest_data_path, map_location="cpu", weights_only=False)

        scores, per_trial = score_block(trials)  # 微調前計分：live 模型用 Unity 的結果，另一個模型用同一時間的 Unity 式投票
        live_before = global_value.live_model
        if sprt is None:
            self._decide_live(scores)
            gates = self._gates(scores)
        else:  # SPRT：gate 由每個 trial 的 SPRT 決定；切換 live 只在 4-trial block 結束時，用這 4 個 trial 的分數
            if sprt.get("switch_scores") is not None:
                self._decide_live(sprt["switch_scores"])
            gates = sprt["gates"]
        failures = self._failures_per_model(train_data, per_trial)

        failed = []
        trained = {}
        start = time.perf_counter()

        def work(slot):
            try:
                trained[slot.name] = slot.finetune(train_data, gates[slot.name], failures[slot.name])
            except Exception:
                failed.append(slot.name)
                traceback.print_exc()

        threads = [threading.Thread(target=work, args=(slot,)) for slot in self.slots.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ft_seconds = time.perf_counter() - start
        self.models_changed = any(trained.values()) or global_value.live_model != live_before
        self.last_trained = {name: bool(trained.get(name)) for name in self.slots}

        global_value.NOW_TRAINED_CHECKPOINT = getattr(global_value, f"{global_value.live_model}_CHECKPOINT")
        self._log_block(scores, failed, trained, ft_seconds, sprt)

    def _gates(self, scores):
        """每個模型自己決定要不要微調：本 block 至少錯一個 trial，且本 run 累計的 Unity 式正確率 < finetune_acc_threshold"""
        if self.counter_run != global_value.runCount:
            self.counter_run = global_value.runCount
            self.run_counts = {"A": [0, 0], "B": [0, 0]}
        gates = {}
        for name, (correct, scored) in scores.items():
            self.run_counts[name][0] += correct
            self.run_counts[name][1] += scored
            run_correct, run_scored = self.run_counts[name]
            gates[name] = (scored - correct) >= 1 and run_scored > 0 and run_correct / run_scored < config.FINETUNE_ACC_THRESHOLD
        return gates

    @staticmethod
    def _failures_per_model(train_data, per_trial):
        """live 的模型沿用資料裡的 failure (Unity 的結果)；另一個模型的 failure = 它的 Unity 式投票是錯的"""
        ids = train_data.get('trial_ids')
        stored = np.asarray(train_data['failures'])
        out = {"A": stored, "B": stored}
        if ids is None or not per_trial:
            return out
        for name in ("A", "B"):
            arr = stored.copy()
            for row, idx in enumerate(np.asarray(ids)):
                info = per_trial.get(int(idx))
                if info is not None and info['live'] != name:
                    arr[row] = (info[name] is False)
            out[name] = arr
        return out

    def _decide_live(self, scores):
        if global_value.LIVE_MODE in ("A", "B"):
            global_value.live_model = global_value.LIVE_MODE
            return
        acc = {n: (c / k if k else None) for n, (c, k) in scores.items()}
        cur = global_value.live_model
        other = "B" if cur == "A" else "A"
        if acc[cur] is not None and acc[other] is not None and acc[other] > acc[cur]:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= config.SWITCH_STREAK:
            global_value.live_model = other
            self.streak = 0
            print(f"{config.TAGS.INFO.value} auto：切換 live 模型 {cur} -> {other}")

    def _log_block(self, scores, failed, trained, ft_seconds, sprt=None):
        self.block += 1
        path = os.path.join(config.getRunDataDir(), "ab_blocks.csv")
        new_file = not os.path.exists(path)
        sprt_scores = (sprt or {}).get("scores", {})
        row = [self.block, max(scores["A"][1], scores["B"][1]), scores["A"][0], scores["B"][0],
               self.streak, global_value.live_model,
               os.path.basename(global_value.A_CHECKPOINT), os.path.basename(global_value.B_CHECKPOINT),
               "".join(failed), global_value.run_mode,
               int(bool(trained.get("A"))), int(bool(trained.get("B"))), f"{ft_seconds:.2f}", "",
               sprt["trigger"] if sprt else "block",
               *(f"{sprt_scores[n]:.3f}" if n in sprt_scores else "" for n in ("A", "B"))]
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self.HEADER)
            w.writerow(row)
        self.last_row = (path, self.block)
        print(f"{config.TAGS.INFO.value} block {self.block}: trial 正確數 A={scores['A']} B={scores['B']} live={global_value.live_model} "
              f"ft A={row[10]} B={row[11]} ({ft_seconds:.1f}s)")

    def record_swap(self, swap_time):
        """新權重載入後，把載入時間補進最近一個 block 的資料列 (swap_time 欄)"""
        if not self.last_row:
            return
        path, block = self.last_row
        self.last_row = None
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            for row in rows[1:]:
                if row and row[0] == str(block):
                    row[self.HEADER.index("swap_time")] = f"{swap_time:.3f}"
            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)
        except Exception as e:
            print(f"{config.TAGS.WARNING.value} swap_time 寫入失敗: {e}")


class EEGTrainingPipeline:
    """離線 / Run 結束後常規訓練 Pipeline"""

    def __init__(self, epochs=150, tcp_server=None):
        self.batch_size = 8
        self.channel_index = config.channel_index
        self.lr = 1e-4
        self.epochs = epochs
        self.strides = 100
        self.segment_len = 500
        self.seed = 42
        self.tcp_server = tcp_server

    def set_the_ft_set(self):
        """讀取最新資料並劃分 70% 訓練集 / 30% 驗證集"""
        if not global_value.train_np_data:
            raise RuntimeError("無可用訓練資料 (train_np_data 為空)")

        latest_data_path = global_value.train_np_data[-1]
        print(f"{config.TAGS.INFO.value} 載入訓練資料: {latest_data_path}")
        train_data = torch.load(latest_data_path, map_location="cpu", weights_only=False)

        data_x_list = [train_data['x_data']]
        data_y_list = [train_data['y_data']]

        train_ds, val_ds = prepare_datasets(
            data_x_list, data_y_list,
            valid_ratio=0.3,
            segment_len=self.segment_len,
            stride=self.strides
        )
        return train_ds, val_ds

    def run_training(self):
        dataset, dataset_val = self.set_the_ft_set()
        params = config.LOAD_MODEL_PARAM

        run_braindecode_training(
            model_class=config.USE_MODEL,
            dataset=dataset,
            dataset_valid=dataset_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            freeze_layers=False,
            params=params,
            seed=self.seed,
            ft=True,
            tcp_server=self.tcp_server
        )


def main():
    pipeline = EEGFineTunePipeline()
    pipeline.run_calibration()


if __name__ == "__main__":
    main()
