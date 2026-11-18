"""
VR-BCI 跨受試者預訓練腳本 (Cross-Subject Pre-training)

把多位受試者的 MI .pt 資料合併起來，訓練出一個「受試者無關」的基礎模型 (pretrained model)，
供線上系統 (EEGPrediction / EEGFineTunePipeline) 當作 fine-tune 的起點，
取代目前 checkpoint_main/c_000.pth 的隨機初始權重。

驗證方式為 held-out subject group：以「受試者」為單位切分，驗證集受試者的資料完全不參與訓練，
因此驗證準確率代表的是模型對「沒見過的人」的泛化能力 (subject-independent accuracy)。

前處理與 else/utils/common_utils.prepare_datasets 以及線上推論 EEGPrediction.predict_loop
完全一致：滑動視窗 -> CAR (跨通道去均值) -> 1-40 Hz 四階 Butterworth 零相位濾波 (filtfilt)。
濾波是「逐視窗」做的，跟線上推論對 500 點 buffer 濾波的行為相同。
刻意不做任何 per-subject / per-run 正規化，以確保訓練與線上推論的輸入分佈相同。

輸入與輸出位置 (皆可用參數覆蓋)：
  資料來源  --data_dir   python/Aden/data_np/DATA_Cygnus/<subject>/<session>/run<N>/mi_*.pt
  模型權重  --out        python/Aden/pretrained_result/model_weights/pretrained_<ch>c_<model>_<時間>.pth
  曲線與log --curve_dir  python/Aden/pretrained_result/learning_curve/pretrain_<ch>c_<model>_<時間>.{png,txt}

注意：權重「不會」自動進 main/EEG/checkpoint_main，所以 Unity 的模型清單不會直接看到它。
要線上使用請依訓練結束時印出的步驟複製過去。

範例：
  # 先列出可用受試者與資料量 (不訓練)
  python python/else/pretraining/pretrain_cross_subject.py --list

  # 22 通道 SCCNet，隨機保留 20% 受試者當驗證集
  python python/else/pretraining/pretrain_cross_subject.py --channels 22 --model SCCNet --epochs 100

  # 指定要訓練的受試者與驗證受試者
  python python/else/pretraining/pretrain_cross_subject.py \
      --ids 35,37,38,40,41,42,43,44 --val_subjects 43,44 \
      --model ShallowFBCSPNet --epochs 60

  # 只用 s1、只用 run1~4，並排除失敗 trial
  python python/else/pretraining/pretrain_cross_subject.py --sessions s1 --runs 1-4 --exclude_failed
"""

import os
import sys
import glob
import time
import random
import shutil
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# --- 路徑設定 ---
# 注意：python/else 因為 "else" 是 Python 保留字，無法當成 package import，
# 所以跟 train_cv.py 一樣直接把 else/utils 塞進 sys.path。
_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
_else_dir = os.path.join(_python_dir, "else")
for _p in [_python_dir, os.path.join(_else_dir, "utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# UTF-8 輸出防護
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # 無頭伺服器，必須在 pyplot 被 import 之前設定 (common_utils 會 import pyplot)
import matplotlib.pyplot as plt

from braindecode.models import ShallowFBCSPNet, EEGNetv4, EEGConformer, ATCNet

from common_utils import bandpass, tee_log, CHANNEL_CONFIGS, DEFAULT_IDS, DEFAULT_SESSIONS
import main.Utils.config as config
from main.EEG.models import SCCNet
from main.EEG.MI_train import BraindecodeTrainer

MODEL_DICT = {
    "SCCNet": SCCNet,
    "ShallowFBCSPNet": ShallowFBCSPNet,
    "EEGNetv4": EEGNetv4,
    "EEGConformer": EEGConformer,
    "ATCNet": ATCNet,
}

DEFAULT_DATA_DIR = os.path.join(_python_dir, "Aden", "data_np", "DATA_Cygnus")
PRETRAIN_RESULT_DIR = os.path.join(_python_dir, "Aden", "pretrained_result")
DEFAULT_WEIGHT_DIR = os.path.join(PRETRAIN_RESULT_DIR, "model_weights")
DEFAULT_CURVE_DIR = os.path.join(PRETRAIN_RESULT_DIR, "learning_curve")

# 學習曲線配色：dataviz 預設類別色 slot 1 (blue) 與 slot 2 (orange)。
# 已用 validate_palette 檢驗過：CVD dE 24.7 (門檻 8)、normal dE 33.6 (底線 15)、對比度皆 >= 3:1。
C_TRAIN = "#2a78d6"
C_VAL = "#eb6834"
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e5e4e0"
C_REF = "#8a8984"

# 逐受試者曲線配色：避開 train(blue)/held-out(orange) 已佔用的語意，改用 violet/aqua/red/yellow。
# validate_palette 全配對檢驗：normal dE 20.8 (底線 15)、CVD dE 6.9 (落在 6-8 floor 區間)，
# 因此每條線額外用不同虛線樣式 + 線末直接標註受試者編號作為顏色以外的第二種區分方式。
C_SUBJECTS = ["#4a3aa7", "#1baf7a", "#e34948", "#eda100", "#e87ba4", "#008300"]
DASH_SUBJECTS = [(0, ()), (0, (5, 2)), (0, (1, 1.6)), (0, (7, 2, 1.6, 2))]


class PerSubjectEvalTrainer(BraindecodeTrainer):
    """讓每個 epoch 的驗證同時記錄「逐受試者」準確率，且不改動共用的 BraindecodeTrainer。

    val_loader 建立時是 shuffle=False，所以 batch 的輸出順序與 codes 陣列一一對應，
    可以用一個 running offset 把預測分桶到各受試者 —— 不需要多跑一次 forward pass。
    pooled 的 loss/accuracy 計算與父類別完全相同，所以 early stopping 的行為不受影響。
    """

    _codes = None
    _subjects = None

    def __init__(self, *a, beta1=None, weight_decay=0.0, dropout=None, **kw):
        # 這三個要在 super().__init__() 之前設好，因為它會呼叫 _init_model()
        self._beta1 = beta1
        self._weight_decay = weight_decay
        self._dropout = dropout
        super().__init__(*a, **kw)

    def _init_model(self):
        """沿用父類別建模，再套用 dropout 率與 optimizer 覆寫 (不動共用的 MI_train.py)。"""
        super()._init_model()
        if self._dropout is not None:
            for mod in self.model.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = self._dropout
        beta1 = 0.5 if self._beta1 is None else self._beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                    betas=(beta1, 0.999),
                                    weight_decay=self._weight_decay or 0.0)

    def track_subjects(self, codes, subjects):
        self._codes = codes
        self._subjects = list(subjects)
        self.per_subject_hist = {s: [] for s in self._subjects}

    def _evaluate(self, loader):
        if loader is None:
            return 0.0, 0.0
        track = (self._codes is not None) and (loader is getattr(self, "val_loader", None))

        self.model.eval()
        total_correct, total_samples, losses = 0, 0, []
        if track:
            hits = np.zeros(len(self._subjects), dtype=np.int64)
            seen = np.zeros(len(self._subjects), dtype=np.int64)
            offset = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float()
                output = self.model(x_batch.squeeze(1))
                loss = self.criterion(output, y_batch)
                losses.append(loss.item())

                preds = output.argmax(dim=1)
                truth = y_batch.argmax(dim=1)
                total_correct += (preds == truth).sum().item()
                total_samples += y_batch.size(0)

                if track:
                    n = y_batch.size(0)
                    correct_mask = (preds == truth).cpu().numpy()
                    codes_batch = self._codes[offset:offset + n]
                    for code in np.unique(codes_batch):
                        sel = codes_batch == code
                        hits[code] += int(correct_mask[sel].sum())
                        seen[code] += int(sel.sum())
                    offset += n

        mean_loss = float(np.mean(losses)) if losses else 0.0
        accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

        if track:
            for i, s in enumerate(self._subjects):
                self.per_subject_hist[s].append(
                    float(hits[i] / seen[i]) if seen[i] else float("nan"))

        return mean_loss, accuracy


class ConsoleReporter:
    """假裝自己是 TCPServer 的進度回報器。

    BraindecodeTrainer.train() 每個 epoch 只會透過 tcp_server.broadcast() 回報進度
    (原本是要傳給 Unity)，所以這裡塞一個只會 print 的替身，長時間訓練才看得到進度。
    """

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start = time.perf_counter()

    def broadcast(self, message):
        elapsed = time.perf_counter() - self.start
        print(f"[{elapsed / 60:6.1f} min] {message}", flush=True)


# ============================================================
# 資料探索與載入
# ============================================================

def parse_id_list(raw, default):
    """解析逗號分隔的清單；'all' 或空值回傳 default"""
    if raw is None or str(raw).strip().lower() in ("", "all"):
        return list(default)
    return [x.strip() for x in str(raw).split(',') if x.strip()]


def parse_run_list(raw):
    """解析 run 清單，支援 '1-7'、'1,3,5' 兩種寫法"""
    raw = str(raw).strip()
    if '-' in raw and ',' not in raw:
        lo, hi = raw.split('-', 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def resolve_mi_name(data_dir, channels, override=None):
    """決定 MI .pt 的檔名。

    CHANNEL_CONFIGS["13"]["mi_filename"] 是 "mi.pt"，但實際資料集是用 "mi_13.pt" 命名，
    直接照 CHANNEL_CONFIGS 找會一個檔案都找不到 (train_cv.py --channels 13 就是這個問題)。
    所以這裡依照磁碟上真正存在的檔名自動判斷。
    """
    if override:
        return override

    candidates = []
    for name in (CHANNEL_CONFIGS[channels]["mi_filename"], f"mi_{channels}.pt", "mi.pt"):
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        if glob.glob(os.path.join(data_dir, "*", "*", "run*", name)):
            return name
    return candidates[0]


def discover_runs(data_dir, ids, sessions, runs, mi_name):
    """列舉 <data_dir>/<subject>/<session>/run<N>/<mi_name>，回傳存在與缺少的清單"""
    found, missing = [], []
    for sub in ids:
        for sess in sessions:
            for r in runs:
                path = os.path.join(data_dir, sub, sess, f"run{r}", mi_name)
                if os.path.exists(path):
                    found.append(dict(subject=sub, session=sess, run=r, path=path))
                else:
                    missing.append(path)
    return found, missing


def window_one_run(x_trials, y_trials, segment_len, stride):
    """把單一 run 的 trial 切成視窗並前處理。

    輸入 x_trials: (trial, channel, sample)
    輸出 (windows, 1, channel, segment_len) float32 與對應標籤

    前處理刻意逐視窗進行，跟 common_utils.prepare_datasets 與線上推論一致。
    """
    # (trial, channel, sample) -> (trial, sample, channel)，對齊 prepare_datasets 的內部排列
    x_trials = np.transpose(x_trials, (0, 2, 1))

    segments, labels, trial_idx = [], [], []
    for i, trial in enumerate(x_trials):
        if trial.shape[0] < segment_len:
            continue
        for start in range(0, trial.shape[0] - segment_len + 1, stride):
            data = trial[start:start + segment_len]                      # (segment_len, channel)
            window = bandpass(data - np.mean(data, axis=1, keepdims=True))  # CAR 後沿時間軸濾波
            segments.append(window)
            labels.append(y_trials[i])
            trial_idx.append(i)

    if not segments:
        return None, None, None

    # (N, segment_len, channel) -> (N, channel, segment_len) -> 補上 SCCNet 需要的 dummy 維度
    x = np.transpose(np.stack(segments), (0, 2, 1)).astype(np.float32, copy=False)
    return x[:, None, :, :], np.asarray(labels, dtype=np.int64), np.asarray(trial_idx, dtype=np.int64)


def build_window_arrays(records, channel_index, segment_len, stride, exclude_failed, label_name,
                        return_trials=False):
    """載入並前處理一組 run，回傳 (X, y, subject_codes, subjects, stats)；
    return_trials=True 時多回傳每個視窗所屬的全域 trial 編號 (trial 層級多數決用)

    X: (N, 1, channel, segment_len) float32
    subject_codes: 每個視窗對應的受試者索引，供 per-subject 評估使用
    """
    blocks, label_blocks, code_blocks, trial_blocks = [], [], [], []
    trial_offset = 0
    subjects = sorted({r["subject"] for r in records}, key=lambda s: int(s))
    subject_to_code = {s: i for i, s in enumerate(subjects)}
    n_failed_dropped, n_trials_used, n_runs_used = 0, 0, 0

    print(f"\n{config.TAGS.INFO.value} 載入 {label_name} 資料：{len(records)} 個 run，"
          f"{len(subjects)} 位受試者 {subjects}")

    t0 = time.perf_counter()
    for idx, rec in enumerate(records, 1):
        data = torch.load(rec["path"], map_location="cpu", weights_only=False)
        x = np.asarray(data["x_data"], dtype=np.float32)   # (trial, channel, sample)
        y = np.asarray(data["y_data"], dtype=np.int64)

        if exclude_failed and "failures" in data:
            keep = ~np.asarray(data["failures"], dtype=bool)
            n_failed_dropped += int((~keep).sum())
            x, y = x[keep], y[keep]
            if len(x) == 0:
                continue

        x = x[:, channel_index, :]
        xb, yb, tb = window_one_run(x, y, segment_len, stride)
        if xb is None:
            print(f"  ⚠️  {rec['path']} 的 trial 長度不足 {segment_len}，略過")
            continue

        blocks.append(xb)
        label_blocks.append(yb)
        code_blocks.append(np.full(len(yb), subject_to_code[rec["subject"]], dtype=np.int16))
        trial_blocks.append(tb + trial_offset)
        trial_offset += len(y)
        n_trials_used += len(y)
        n_runs_used += 1

        if idx % 25 == 0 or idx == len(records):
            done_gb = sum(b.nbytes for b in blocks) / 1e9
            print(f"  ├── {idx}/{len(records)} runs  ({done_gb:.1f} GB, "
                  f"{time.perf_counter() - t0:.0f}s)", flush=True)

    if not blocks:
        raise RuntimeError(f"{label_name} 沒有任何可用資料")

    X = np.concatenate(blocks, axis=0)
    y = np.concatenate(label_blocks, axis=0)
    codes = np.concatenate(code_blocks, axis=0)
    trials = np.concatenate(trial_blocks, axis=0)
    del blocks, label_blocks, code_blocks, trial_blocks

    counts = np.bincount(y, minlength=config.N_Class)
    stats = dict(n_runs=n_runs_used, n_trials=n_trials_used, n_windows=len(X),
                 n_failed_dropped=n_failed_dropped, label_counts=counts.tolist(),
                 subjects=subjects)

    print(f"  └── {label_name} 完成：{len(X)} 視窗，shape={X.shape}，"
          f"標籤分佈={counts.tolist()}，{X.nbytes / 1e9:.1f} GB，"
          f"耗時 {time.perf_counter() - t0:.0f}s")
    if exclude_failed:
        print(f"      (已排除 {n_failed_dropped} 個失敗 trial)")

    if return_trials:
        return X, y, codes, subjects, stats, trials
    return X, y, codes, subjects, stats


def euclidean_align(X, codes, n_subjects, label=""):
    """Euclidean Alignment (He & Wu, 2020)：用每位受試者自己的平均共變異數做白化。

    對每位受試者計算 R = mean_i(X_i X_i^T)，再把每個 window 變換成 R^{-1/2} X，
    使每個人的平均共變異數都變成單位矩陣，不同人的輸入分佈因此被對齊到同一個空間。

    只用到輸入訊號的分佈，完全不需要標籤 —— 所以對「沒看過的新受試者」也能套用，
    但仍需要該受試者的一小段 EEG 才能估出 R (是 zero-label，不是 zero-data)。
    """
    X = np.ascontiguousarray(X)
    for c in range(n_subjects):
        idx = np.where(codes == c)[0]
        if len(idx) == 0:
            continue
        Z = X[idx, 0]                                   # (n, ch, t)
        # 平均共變異數 (對 window 與時間點取平均)
        R = np.einsum('nct,ndt->cd', Z, Z) / (len(idx) * Z.shape[2])
        w, V = np.linalg.eigh(R.astype(np.float64))
        # CAR (跨通道去均值) 會讓共變異數降秩，至少有一個特徵值≈0。
        # 直接取 w^{-1/2} 會把那個 null 方向放大到爆掉，所以改用 pseudo-inverse：
        # 低於最大特徵值 1e-6 倍的方向直接歸零 (那些方向本來就沒有訊號)。
        tol = w.max() * 1e-6
        w_inv_sqrt = np.where(w > tol, np.maximum(w, tol) ** -0.5, 0.0)
        R_inv_sqrt = (V * w_inv_sqrt) @ V.T
        X[idx, 0] = np.einsum('cd,ndt->nct', R_inv_sqrt.astype(np.float32), Z)
    print(f"{config.TAGS.INFO.value} 已對 {label} 套用 Euclidean Alignment "
          f"({n_subjects} 位受試者各自白化)")
    return X


def to_dataset(X, y):
    """包成 BraindecodeTrainer 需要的 TensorDataset (標籤為 one-hot float)。

    torch.from_numpy 與 numpy 陣列共用記憶體，不會再複製一份。
    """
    x_tensor = torch.from_numpy(X)
    y_tensor = F.one_hot(torch.from_numpy(y), num_classes=config.N_Class).float()
    return TensorDataset(x_tensor, y_tensor)


# ============================================================
# 評估
# ============================================================

def plot_learning_curve(hist, best_epoch, out_png, title, subtitle,
                        train_label="Train (in-sample subjects)",
                        val_label="Held-out subjects",
                        per_subject_ylabel="Held-out accuracy\nper subject"):
    """畫出 loss 與 accuracy 的學習曲線並存成 PNG。

    loss 與 accuracy 尺度不同，所以分成上下兩張圖，不用雙 y 軸。
    實線 = 訓練集，虛線 = held-out 驗證受試者；虛線同時作為顏色之外的第二種區分方式，
    黑白列印或色盲情況下仍可分辨。
    """
    epochs = np.arange(1, len(hist["loss"]) + 1)
    per_subject = hist.get("val_acc_per_subject") or {}
    n_panels = 3 if per_subject else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.8 * n_panels), sharex=True)
    fig.patch.set_facecolor(C_SURFACE)

    panels = [
        (axes[0], "loss", "val_loss", "Cross-entropy loss", None),
        (axes[1], "acc", "val_acc", "Accuracy", 0.5),
    ]

    for ax, tr_key, va_key, ylabel, chance in panels:
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color=C_GRID, linewidth=0.8, linestyle="-")  # 網格線用實線細線，不用虛線
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
            ax.spines[side].set_linewidth(0.8)

        chance_handle = None
        if chance is not None:
            # 基準線用圖例標示，不用線上文字，避免和曲線或最佳 epoch 標註擠在一起
            chance_handle = ax.axhline(chance, color=C_REF, linewidth=1.0,
                                       linestyle=(0, (4, 3)), zorder=1,
                                       label=f"chance ({chance:.0%})")

        line_tr, = ax.plot(epochs, hist[tr_key], color=C_TRAIN, linewidth=2.0,
                           label=train_label, zorder=3)
        series_handles = [line_tr]
        if val_label is not None:
            line_va, = ax.plot(epochs, hist[va_key], color=C_VAL, linewidth=2.0,
                               linestyle=(0, (5, 2)), label=val_label, zorder=3)
            series_handles.append(line_va)
        else:
            va_key = tr_key   # 沒有驗證集 (--no_val)：標註改標在訓練曲線上

        # 只在最佳 epoch 直接標數值，不是每個點都標
        ax.axvline(best_epoch + 1, color=C_REF, linewidth=1.2, alpha=0.45, zorder=2)
        best_val = hist[va_key][best_epoch]
        ax.plot([best_epoch + 1], [best_val], marker="o", markersize=9,
                color=C_VAL if val_label is not None else C_TRAIN,
                markeredgecolor=C_SURFACE, markeredgewidth=2, zorder=4)  # 2px surface ring
        # early stopping 時最佳 epoch 常常就是最後一個，標註要往左翻才不會被裁掉
        near_right = (best_epoch + 1) > len(epochs) * 0.75
        xoff, ha = (-10, "right") if near_right else (10, "left")
        ax.annotate(f"{best_val:.3f}", xy=(best_epoch + 1, best_val), xytext=(xoff, 8),
                    textcoords="offset points", ha=ha, fontsize=10,
                    fontweight="bold", color=C_INK)

        ax.set_ylabel(ylabel, fontsize=11, color=C_INK2)
        ax.tick_params(colors=C_INK2, labelsize=10, length=0)

        # 上圖放兩條曲線的圖例，下圖只放 chance 基準線的圖例；
        # loc="best" 讓 matplotlib 自動挑不擋到資料的位置
        if chance_handle is None:
            ax.legend(handles=series_handles, loc="best", frameon=False,
                      fontsize=10, labelcolor=C_INK2)
        else:
            ax.legend(handles=[chance_handle], loc="best", frameon=False,
                      fontsize=9, labelcolor=C_INK2)

        if chance is not None:
            lo = min(min(hist[tr_key]), min(hist[va_key]), chance) - 0.04
            hi = max(max(hist[tr_key]), max(hist[va_key]), chance) + 0.06
            ax.set_ylim(max(0.0, lo), min(1.0, hi))

    if per_subject:
        ax = axes[2]
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color=C_GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
            ax.spines[side].set_linewidth(0.8)

        ref = ax.axhline(0.5, color=C_REF, linewidth=1.0, linestyle=(0, (4, 3)),
                         zorder=1, label="chance (50%)")
        ax.axvline(best_epoch + 1, color=C_REF, linewidth=1.2, alpha=0.45, zorder=2)

        subj_handles, endpoints, lo, hi = [], [], 1.0, 0.0
        for i, (subject, accs) in enumerate(per_subject.items()):
            colour = C_SUBJECTS[i % len(C_SUBJECTS)]
            line, = ax.plot(epochs[:len(accs)], accs, color=colour, linewidth=1.8,
                            linestyle=DASH_SUBJECTS[(i // len(C_SUBJECTS)) % len(DASH_SUBJECTS)],
                            label=f"subject {subject}", zorder=3)
            subj_handles.append(line)
            endpoints.append([float(accs[-1]), f"S{subject}"])
            lo, hi = min(lo, min(accs)), max(hi, max(accs))

        y_lo = max(0.0, min(lo, 0.5) - 0.04)
        y_hi = min(1.0, max(hi, 0.5) + 0.05)

        # 受試者結束在相近準確率時線末標註會疊在一起，依序往上推開
        min_gap = (y_hi - y_lo) * 0.05
        endpoints.sort(key=lambda e: e[0])
        for i in range(1, len(endpoints)):
            if endpoints[i][0] - endpoints[i - 1][0] < min_gap:
                endpoints[i][0] = endpoints[i - 1][0] + min_gap
        if endpoints:
            y_hi = max(y_hi, endpoints[-1][0] + min_gap * 0.5)

        # 線末直接標註：同時滿足對比度 relief 與「識別不能只靠顏色」
        for y_label, text in endpoints:
            ax.annotate(text, xy=(epochs[-1], y_label), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=9, color=C_INK2, annotation_clip=False)

        # 右側只留標註所需的空間；x 軸是共用的，留太多上面兩張圖會出現空白
        ax.set_xlim(epochs[0] - 0.5, epochs[-1] + max(0.35, len(epochs) * 0.035))
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel(per_subject_ylabel, fontsize=11, color=C_INK2)
        ax.tick_params(colors=C_INK2, labelsize=10, length=0)
        ax.legend(handles=subj_handles + [ref], loc="best", frameon=False,
                  fontsize=9, labelcolor=C_INK2, ncol=2)

    xlabel = (f"Epoch  (best = {best_epoch + 1}, selected by val_loss)" if val_label is not None
              else f"Epoch  (fixed length, last epoch {best_epoch + 1} saved)")
    axes[-1].set_xlabel(xlabel, fontsize=11, color=C_INK2)

    # 標題與副標題畫在 figure 層級並預留空間，避免兩行文字互相重疊
    top = 0.92 if n_panels == 2 else 0.945
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.text(0.012, 0.992, title, fontsize=13, fontweight="bold",
             color=C_INK, ha="left", va="top")
    fig.text(0.012, 0.992 - (0.040 if n_panels == 2 else 0.027),
             subtitle, fontsize=9.5, color=C_INK2, ha="left", va="top")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    return out_png


def evaluate_per_subject(model, X, y, codes, subjects, device, batch_size=512):
    """逐受試者計算驗證準確率，回傳 {subject: acc} 與整體 acc"""
    model.eval()
    correct = np.zeros(len(subjects), dtype=np.int64)
    total = np.zeros(len(subjects), dtype=np.int64)

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device).float()
            preds = model(xb.squeeze(1)).argmax(dim=1).cpu().numpy()
            yb = y[i:i + batch_size]
            cb = codes[i:i + batch_size]
            for code in np.unique(cb):
                mask = cb == code
                correct[code] += int((preds[mask] == yb[mask]).sum())
                total[code] += int(mask.sum())

    per_subject = {s: (float(correct[i] / total[i]) if total[i] else float('nan'))
                   for i, s in enumerate(subjects)}
    overall = float(correct.sum() / total.sum()) if total.sum() else float('nan')
    return per_subject, overall


def evaluate_per_subject_trials(model, X, y, codes, trials, subjects, device, batch_size=512):
    """trial 層級準確率：每個 trial 的視窗多數決；平手時取平均機率較高的類別。回傳 ({subject: acc}, 整體 acc)"""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device).float()
            probs.append(torch.softmax(model(xb.squeeze(1)), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    votes = probs.argmax(1)
    correct = np.zeros(len(subjects), dtype=np.int64)
    total = np.zeros(len(subjects), dtype=np.int64)
    for t in np.unique(trials):
        sel = trials == t
        n1 = int(votes[sel].sum())
        n0 = int(sel.sum()) - n1
        pred = 1 if n1 > n0 else 0 if n0 > n1 else int(probs[sel, 1].mean() > probs[sel, 0].mean())
        code = int(codes[sel][0])
        correct[code] += int(pred == int(y[sel][0]))
        total[code] += 1
    per_subject = {s: (float(correct[i] / total[i]) if total[i] else float('nan'))
                   for i, s in enumerate(subjects)}
    overall = float(correct.sum() / total.sum()) if total.sum() else float('nan')
    return per_subject, overall


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VR-BCI 跨受試者預訓練 (held-out subject 驗證)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- 資料選擇 ---
    g = parser.add_argument_group("資料選擇")
    g.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="資料根目錄，結構為 <data_dir>/<subject>/<session>/run<N>/mi_*.pt")
    g.add_argument("--channels", type=str, default="22", choices=["13", "22"],
                   help="通道模式")
    g.add_argument("--mi_name", type=str, default=None,
                   help="MI 檔名 (預設自動偵測 mi_13.pt / mi_22.pt / mi.pt)")
    g.add_argument("--ids", type=str, default="all",
                   help="參與訓練的受試者，逗號分隔，或 'all' 表示 common_utils.DEFAULT_IDS 的 24 人")
    g.add_argument("--exclude_ids", type=str, default=None,
                   help="完全排除的受試者 (逗號分隔)；訓練集與 held-out 驗證集都不會出現，"
                        "等同於直接從 --ids 名單移除")
    g.add_argument("--val_subjects", type=str, default=None,
                   help="指定當驗證集的受試者 (必須是 --ids 的子集)；未指定則依 --val_frac 隨機挑選")
    g.add_argument("--test_subjects", type=str, default=None,
                   help="測試受試者 (逗號分隔)：不參與訓練、不參與 early stopping，只在選定 epoch 後評估一次")
    g.add_argument("--val_frac", type=float, default=0.2,
                   help="未指定 --val_subjects 時，隨機保留多少比例的受試者當驗證集")
    g.add_argument("--no_val", action="store_true",
                   help="不保留驗證集：所有受試者都拿來訓練，固定跑滿 --epochs 並存最後一個 epoch "
                        "(用於產生部署用模型；epoch 數請依 held-out 實驗決定)")
    g.add_argument("--sessions", type=str, default=",".join(DEFAULT_SESSIONS),
                   help="要使用的 session (split_mode=subject 時對訓練與驗證都適用)")
    g.add_argument("--split_mode", type=str, default="subject", choices=["subject", "session"],
                   help="subject: 以受試者切分 (跨受試者泛化)；"
                        "session: 同一批受試者，用 --train_sessions 訓練、--val_sessions 驗證 "
                        "(受試者內跨 session，可用來驗證 pipeline 本身是否有效)")
    g.add_argument("--train_sessions", type=str, default="s1",
                   help="split_mode=session 時的訓練 session")
    g.add_argument("--val_sessions", type=str, default="s2",
                   help="split_mode=session 時的驗證 session")
    g.add_argument("--runs", type=str, default="1-7",
                   help="要使用的 run，支援 '1-7' 或 '1,3,5'")
    g.add_argument("--exclude_failed", action="store_true",
                   help="排除 failures=True 的 trial (CUT 次數不足 group_note_num 的失敗 trial)")

    # --- 前處理 ---
    g = parser.add_argument_group("前處理")
    g.add_argument("--segment_len", type=int, default=config.BUFFER_SIZE,
                   help="視窗長度 (取樣點)。必須與線上推論的 BUFFER_SIZE 一致才能直接部署")
    g.add_argument("--align", action="store_true",
                   help="套用 Euclidean Alignment：以每位受試者自己的平均共變異數白化，"
                        "對齊跨受試者的輸入分佈 (不需標籤)")
    g.add_argument("--stride", type=int, default=100,
                   help="滑動視窗步長。越小資料越多、記憶體需求越大")

    # --- 模型與訓練 ---
    g = parser.add_argument_group("模型與訓練")
    g.add_argument("--model", type=str, default="SCCNet", choices=list(MODEL_DICT.keys()),
                   help="模型架構")
    g.add_argument("--epochs", type=int, default=100, help="最大 epoch 數")
    g.add_argument("--batch_size", type=int, default=256,
                   help="批次大小 (跨受試者資料量遠大於單 run，建議 128~512)")
    g.add_argument("--lr", type=float, default=1e-3, help="學習率")
    g.add_argument("--beta1", type=float, default=0.5,
                   help="Adam beta1。專案沿用的 0.5 是 DCGAN 的設定，PyTorch 預設為 0.9")
    g.add_argument("--weight_decay", type=float, default=0.0,
                   help="Adam weight decay (L2)，預設 0 表示不使用")
    g.add_argument("--dropout", type=float, default=None,
                   help="覆寫 SCCNet 的 dropout 率 (模型內建 0.5)")
    g.add_argument("--patience", type=int, default=20,
                   help="Early stopping 容忍 epoch 數 (依 val_loss 判斷)")
    g.add_argument("--select_by", type=str, default="val_loss", choices=["val_loss", "val_acc"],
                   help="儲存哪個 epoch 的權重：val_loss 最低或 val_acc 最高 (兩者都會印出並記在 meta)")
    g.add_argument("--seed", type=int, default=42, help="隨機種子")
    g.add_argument("--deterministic", action="store_true", default=True,
                   help="固定 cuDNN 演算法選擇，讓相同指令+相同 seed 得到完全一樣的結果 "
                        "(與 else/training/train_cv.py 一致)")
    g.add_argument("--no_deterministic", dest="deterministic", action="store_false",
                   help="關閉決定性，允許 cuDNN autotune (較快但每次結果會有些微差異)")
    g.add_argument("--device", type=str, default=None, help="cuda / cpu (預設自動)")

    # --- 輸出 ---
    g = parser.add_argument_group("輸出")
    g.add_argument("--out", type=str, default=None,
                   help=f"輸出權重路徑 (預設 {DEFAULT_WEIGHT_DIR}/"
                        f"pretrained_<ch>c_<model>_<時間>.pth)")
    g.add_argument("--curve_dir", type=str, default=DEFAULT_CURVE_DIR,
                   help="學習曲線 PNG 與 log 檔的存放資料夾")
    g.add_argument("--no_plot", action="store_true",
                   help="不產生學習曲線 PNG (history 仍會存進 checkpoint，隨時可事後重畫)")
    g.add_argument("--work_dir", type=str, default=None,
                   help="每個 epoch 暫存權重的資料夾 (預設 <out 同層>/.pretrain_work)")
    g.add_argument("--keep_epoch_checkpoints", action="store_true",
                   help="保留每個 epoch 的暫存權重 (預設訓練結束後刪除，只留最佳權重)")
    g.add_argument("--log_file", type=str, default=None,
                   help="log 檔路徑 (預設 <curve_dir>/pretrain_<ch>c_<model>_<時間>.txt)")

    # --- 其他 ---
    g = parser.add_argument_group("其他")
    g.add_argument("--list", action="store_true",
                   help="只列出找到的受試者與資料量後結束，不訓練")
    g.add_argument("--max_ram_gb", type=float, default=48.0,
                   help="視窗陣列的記憶體上限估計 (超過就直接中止，避免 OOM)")

    return parser.parse_args()


def run(args):
    ch_cfg = CHANNEL_CONFIGS[args.channels]
    channel_index = ch_cfg["channel_index"]
    ch_names = ch_cfg["ch_names"]

    session_mode = (args.split_mode == "session")
    train_sessions = parse_id_list(args.train_sessions, ["s1"]) if session_mode else None
    val_sessions = parse_id_list(args.val_sessions, ["s2"]) if session_mode else None
    if session_mode and set(train_sessions) & set(val_sessions):
        raise SystemExit(f"❌ --train_sessions {train_sessions} 與 --val_sessions {val_sessions} 不能重疊")

    ids = parse_id_list(args.ids, DEFAULT_IDS)
    # 被排除的受試者在切分之前就移除，所以訓練集與驗證集都不會看到他們
    excluded = parse_id_list(args.exclude_ids, []) if args.exclude_ids else []
    if excluded:
        ids = [i for i in ids if i not in excluded]
        if not ids:
            raise SystemExit("❌ --exclude_ids 把所有受試者都排除了，沒有資料可訓練")
    sessions = (train_sessions + val_sessions) if session_mode \
        else parse_id_list(args.sessions, DEFAULT_SESSIONS)
    runs = parse_run_list(args.runs)
    mi_name = resolve_mi_name(args.data_dir, args.channels, args.mi_name)

    print("=" * 78)
    print("🎯 VR-BCI 跨受試者預訓練 (Cross-Subject Pre-training)")
    print(f"  ├── data_dir : {args.data_dir}")
    print(f"  ├── 通道模式 : {args.channels} ch -> {ch_names}")
    print(f"  ├── MI 檔名  : {mi_name}")
    print(f"  ├── 受試者   : {len(ids)} 位 {ids}")
    if excluded:
        print(f"  ├── 已排除   : {excluded} (訓練集與驗證集皆不使用)")
    if session_mode:
        print(f"  ├── 切分模式 : session (受試者內跨 session)")
        print(f"  ├── session  : 訓練={train_sessions}, 驗證={val_sessions}")
    else:
        print(f"  ├── 切分模式 : subject (跨受試者)")
        print(f"  ├── session  : {sessions}")
    print(f"  ├── run      : {runs}")
    print(f"  ├── 視窗     : segment_len={args.segment_len}, stride={args.stride}, "
          f"align={'Euclidean Alignment' if args.align else 'off'}")
    print(f"  ├── 模型     : {args.model}, n_classes={config.N_Class}")
    print(f"  ├── 訓練參數 : epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience}")
    print(f"  ├── 最佳化   : Adam betas=({args.beta1}, 0.999), "
          f"weight_decay={args.weight_decay}, "
          f"dropout={'模型內建 0.5' if args.dropout is None else args.dropout}")
    print(f"  └── 重現性   : seed={args.seed}, deterministic={args.deterministic}")
    print("=" * 78)

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"❌ data_dir 不存在: {args.data_dir}")

    records, missing = discover_runs(args.data_dir, ids, sessions, runs, mi_name)
    if not records:
        raise SystemExit(f"❌ 在 {args.data_dir} 找不到任何 {mi_name}，"
                         f"請確認 --data_dir / --channels / --mi_name")

    by_subject = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)
    available = sorted(by_subject, key=lambda s: int(s))

    print(f"\n{config.TAGS.INFO.value} 找到 {len(records)} 個 run "
          f"({len(available)} 位受試者)，缺少 {len(missing)} 個")
    for sub in available:
        got = by_subject[sub]
        detail = ", ".join(f"{s}:{sum(1 for r in got if r['session'] == s)}" for s in sessions)
        print(f"  ├── subject {sub:>3}: {len(got):>2} runs  ({detail})")

    if args.list:
        n_win = 1 + (2000 - args.segment_len) // args.stride
        print(f"\n{config.TAGS.INFO.value} 以 4 秒 trial 估算，每個 trial 可切 {n_win} 個視窗")
        print(f"{config.TAGS.INFO.value} --list 模式結束，未進行訓練")
        return 0

    # --- 切分 train / val (/ test) ---
    test_subjects = parse_id_list(args.test_subjects, []) if args.test_subjects else []
    if test_subjects and (session_mode or args.no_val):
        raise SystemExit("❌ --test_subjects 只支援受試者切分 (不能搭配 session 切分或 --no_val)")
    unknown = [s for s in test_subjects if s not in available]
    if unknown:
        raise SystemExit(f"❌ --test_subjects 中的 {unknown} 不在可用受試者 {available} 內")
    test_records = [r for r in records if r["subject"] in test_subjects]
    if session_mode:
        # 同一批受試者，用 session 切分：訓練與驗證是同一群人的不同天
        train_subjects = val_subjects = available
        train_records = [r for r in records if r["session"] in train_sessions]
        val_records = [r for r in records if r["session"] in val_sessions]
        if not train_records or not val_records:
            raise SystemExit(f"❌ 找不到 {train_sessions} 或 {val_sessions} 的資料")
        print(f"\n{config.TAGS.INFO.value} session 切分 (受試者內跨 session)")
        print(f"  ├── 受試者 ({len(available)} 位): {available}")
        print(f"  ├── 訓練 {train_sessions}: {len(train_records)} runs")
        print(f"  └── 驗證 {val_sessions}: {len(val_records)} runs")
    elif args.no_val:
        if args.val_subjects:
            raise SystemExit("❌ --no_val 與 --val_subjects 不能同時使用")
        train_subjects, val_subjects = list(available), []
        train_records, val_records = list(records), []
        print(f"\n{config.TAGS.INFO.value} 不保留驗證集 (--no_val)：全部 {len(available)} 位受試者用於訓練，"
              f"固定訓練 {args.epochs} 個 epoch，儲存最後一個 epoch 的權重")
        print(f"  └── 訓練 ({len(train_subjects)} 位): {train_subjects}")
    else:
        rng = random.Random(args.seed)
        if args.val_subjects:
            val_subjects = parse_id_list(args.val_subjects, [])
            unknown = [s for s in val_subjects if s not in available]
            if unknown:
                raise SystemExit(f"❌ --val_subjects 中的 {unknown} 不在可用受試者 {available} 內")
            overlap = [s for s in val_subjects if s in test_subjects]
            if overlap:
                raise SystemExit(f"❌ {overlap} 同時出現在 --val_subjects 與 --test_subjects")
        else:
            candidates = [s for s in available if s not in test_subjects]
            n_val = max(1, int(round(len(candidates) * args.val_frac)))
            val_subjects = sorted(rng.sample(candidates, n_val), key=lambda s: int(s))

        train_subjects = [s for s in available if s not in val_subjects and s not in test_subjects]
        if not train_subjects:
            raise SystemExit("❌ 訓練集沒有受試者，請調小 --val_frac 或改用 --val_subjects")
        if not val_subjects:
            raise SystemExit("❌ 驗證集沒有受試者，held-out 驗證與 early stopping 都需要驗證集")

        print(f"\n{config.TAGS.INFO.value} 受試者切分 (held-out subject group)")
        print(f"  ├── 訓練 ({len(train_subjects)} 位): {train_subjects}")
        print(f"  {'├' if test_subjects else '└'}── 驗證 ({len(val_subjects)} 位): {val_subjects}")
        if test_subjects:
            print(f"  └── 測試 ({len(test_subjects)} 位，只在選定 epoch 後評估一次): {test_subjects}")

        train_records = [r for r in records if r["subject"] in train_subjects]
        val_records = [r for r in records if r["subject"] in val_subjects]

    # --- 記憶體估算 (concatenate 期間會短暫用到兩倍) ---
    n_win_per_trial = 1 + (2000 - args.segment_len) // args.stride
    est_gb = (len(records) * 40 * n_win_per_trial * len(channel_index)
              * args.segment_len * 4) / 1e9
    print(f"\n{config.TAGS.INFO.value} 視窗陣列估計約 {est_gb:.1f} GB "
          f"(合併期間峰值約 {est_gb * 2:.1f} GB)")
    if est_gb > args.max_ram_gb:
        raise SystemExit(f"❌ 估計 {est_gb:.1f} GB 超過 --max_ram_gb ({args.max_ram_gb} GB)。"
                         f"請調大 --stride、減少 --ids/--runs，或調高 --max_ram_gb")

    # --- 載入資料 ---
    X_tr, y_tr, codes_tr, train_subj_order, stats_tr = build_window_arrays(
        train_records, channel_index, args.segment_len, args.stride,
        args.exclude_failed, "訓練集")
    if val_records:
        X_va, y_va, codes_va, val_subj_order, stats_va = build_window_arrays(
            val_records, channel_index, args.segment_len, args.stride,
            args.exclude_failed, "驗證集")
    else:
        X_va = y_va = codes_va = None
        val_subj_order, stats_va = [], None
    if test_records:
        X_te, y_te, codes_te, test_subj_order, stats_te, trials_te = build_window_arrays(
            test_records, channel_index, args.segment_len, args.stride,
            args.exclude_failed, "測試集", return_trials=True)
    else:
        stats_te = None

    if args.align:
        X_tr = euclidean_align(X_tr, codes_tr, len(train_subj_order), "訓練集")
        if X_va is not None:
            X_va = euclidean_align(X_va, codes_va, len(val_subj_order), "驗證集")
        if test_records:
            X_te = euclidean_align(X_te, codes_te, len(test_subj_order), "測試集")

    train_ds = to_dataset(X_tr, y_tr)
    val_ds = to_dataset(X_va, y_va) if X_va is not None else None

    # --- 固定隨機種子 (與 run_braindecode_training 一致) ---
    # cuDNN 預設會 benchmark 各種卷積演算法並挑最快的，選到哪個會隨當下 GPU 狀況改變，
    # 造成相同指令跑兩次結果略有不同。關掉 autotune 並強制決定性演算法即可完全重現。
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 輸出路徑 ---
    # 檔名帶時間戳，與同一次 run 的 log / 學習曲線共用 run_tag，方便三個檔案對應
    out_path = args.out or os.path.join(
        DEFAULT_WEIGHT_DIR, f"pretrained_{args.run_tag}.pth")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        try:
            old = torch.load(out_path, map_location="cpu")["pretrain_meta"]
            print(f"\n⚠️  {out_path} 已存在，將被覆蓋 "
                  f"(舊檔: {old['created']}, held-out acc {old['heldout_overall_acc'] * 100:.2f}%)。"
                  f"若要保留舊檔請改用 --out")
        except Exception:
            print(f"\n⚠️  {out_path} 已存在，將被覆蓋。若要保留舊檔請改用 --out")

    work_dir = os.path.abspath(
        args.work_dir or os.path.join(os.path.dirname(out_path), ".pretrain_work"))
    os.makedirs(work_dir, exist_ok=True)

    # BraindecodeTrainer.train() 會把每個 epoch 的權重寫到 config.EEG_CHECKPOINT_TMP_BASE_FILE
    # (線上系統的暫存資料夾)。這裡刻意改掉這個模組層級變數，避免污染線上 checkpoints/。
    original_tmp_dir = config.EEG_CHECKPOINT_TMP_BASE_FILE
    config.EEG_CHECKPOINT_TMP_BASE_FILE = work_dir + os.sep

    device = torch.device(args.device) if args.device else None
    model_kwargs = build_model_kwargs(args.model, len(channel_index), args.segment_len)

    print(f"\n{config.TAGS.INFO.value} 模型參數: {model_kwargs}")
    print(f"{config.TAGS.INFO.value} 暫存權重: {work_dir}")
    print(f"{config.TAGS.INFO.value} 輸出權重: {out_path}")
    print(f"{config.TAGS.INFO.value} 曲線與 log: {os.path.abspath(args.curve_dir)}")
    if args.no_val:
        print(f"\n🚀 開始訓練 (每個 epoch 一行進度，固定 {args.epochs} 個 epoch，不做 early stopping；"
              f"val_loss/val_acc 欄位為訓練集數值)\n")
    else:
        print(f"\n🚀 開始訓練 (每個 epoch 一行進度，early stopping patience={args.patience})\n")

    try:
        trainer = PerSubjectEvalTrainer(
            dataset=train_ds,
            val_dataset=val_ds,
            model_class=MODEL_DICT[args.model],
            model_kwargs=model_kwargs,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            ft=False,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )
        if val_ds is not None:
            trainer.track_subjects(codes_va, val_subj_order)
        # --no_val 時沒有驗證 loss 可以 early stop，一律跑滿 --epochs
        patience = args.epochs + 1 if args.no_val else args.patience
        hist = trainer.train(tcp_server=ConsoleReporter(args.epochs), patience=patience)
        hist["val_acc_per_subject"] = getattr(trainer, "per_subject_hist", {})
    finally:
        config.EEG_CHECKPOINT_TMP_BASE_FILE = original_tmp_dir

    if args.no_val:
        # --- 沒有驗證集：取最後一個 epoch 的權重 ---
        best_epoch = len(hist["loss"]) - 1
        best_path = os.path.join(work_dir, f"train-epoch{best_epoch}.pth")
        if not os.path.exists(best_path):
            raise SystemExit(f"❌ 找不到最後一個 epoch 的權重 {best_path}")
        trainer.model.load_state_dict(torch.load(best_path, map_location=trainer.device)['model_state_dict'])
        print(f"\n{config.TAGS.INFO.value} 使用最後一個 epoch = {best_epoch + 1}/{len(hist['loss'])} (--no_val)")
        print(f"  └── train loss {hist['loss'][best_epoch]:.4f}, acc {hist['acc'][best_epoch]:.4f}")
        print(f"\n{config.TAGS.INFO.value} 沒有 held-out 受試者，無法估計新使用者準確率；"
              f"請引用相同設定的 held-out 實驗結果")
        per_subject, overall = {}, None
        by_loss = by_acc = None
        test_per_subject, test_overall = {}, None
        test_trial_per_subject, test_trial_overall = {}, None
    else:
        # --- 取回 val_loss 最低的權重 (train() 已存成 train-best.pth) ---
        best_path = os.path.join(work_dir, "train-best.pth")
        if not os.path.exists(best_path):
            raise SystemExit(f"❌ 找不到最佳權重 {best_path}")
        trainer.model.load_state_dict(torch.load(best_path, map_location=trainer.device)['model_state_dict'])

        best_epoch = int(np.argmin(hist["val_loss"]))
        print(f"\n{config.TAGS.INFO.value} 最佳 epoch = {best_epoch + 1}/{len(hist['val_loss'])}"
              f" (依 val_loss)")
        print(f"  ├── train loss {hist['loss'][best_epoch]:.4f}, acc {hist['acc'][best_epoch]:.4f}")
        print(f"  └── val   loss {hist['val_loss'][best_epoch]:.4f}, acc {hist['val_acc'][best_epoch]:.4f}")

        # --- Held-out 受試者逐人準確率 ---
        per_subject, overall = evaluate_per_subject(
            trainer.model, X_va, y_va, codes_va, val_subj_order, trainer.device)
        print(f"\n{config.TAGS.INFO.value} Held-out 受試者準確率 (subject-independent)")
        for sub, acc in per_subject.items():
            bar = "█" * int(round(acc * 30))
            print(f"  ├── subject {sub:>3}: {acc * 100:5.2f}%  {bar}")
        accs = np.array([a for a in per_subject.values() if not np.isnan(a)])
        print(f"  └── 整體 {overall * 100:.2f}%  |  逐人平均 {accs.mean() * 100:.2f}% "
              f"± {accs.std() * 100:.2f}% (n={len(accs)})")

        # --- 另一種選法：val_acc 最高的 epoch (per-epoch 權重由 train() 存成 train-epoch{e}.pth) ---
        by_loss = dict(epoch=best_epoch, per_subject=per_subject, overall=overall)
        acc_epoch = int(np.argmax(hist["val_acc"]))
        acc_state = torch.load(os.path.join(work_dir, f"train-epoch{acc_epoch}.pth"),
                               map_location=trainer.device)['model_state_dict']
        trainer.model.load_state_dict(acc_state)
        acc_per_subject, acc_overall = evaluate_per_subject(
            trainer.model, X_va, y_va, codes_va, val_subj_order, trainer.device)
        by_acc = dict(epoch=acc_epoch, per_subject=acc_per_subject, overall=acc_overall)
        acc_accs = np.array([a for a in acc_per_subject.values() if not np.isnan(a)])
        print(f"\n{config.TAGS.INFO.value} 依 val_acc 選 epoch = {acc_epoch + 1}：整體 {acc_overall * 100:.2f}%  |  "
              f"逐人平均 {acc_accs.mean() * 100:.2f}% ± {acc_accs.std() * 100:.2f}%")

        chosen = by_acc if args.select_by == "val_acc" else by_loss
        best_epoch, per_subject, overall = chosen["epoch"], chosen["per_subject"], chosen["overall"]
        if args.select_by == "val_loss":
            trainer.model.load_state_dict(
                torch.load(best_path, map_location=trainer.device)['model_state_dict'])
        print(f"{config.TAGS.INFO.value} 儲存的權重依 {args.select_by}：epoch {best_epoch + 1}")

        # --- 測試受試者：只用已選定的 epoch 評估一次，不影響任何選擇 ---
        test_per_subject, test_overall = {}, None
        test_trial_per_subject, test_trial_overall = {}, None
        if test_records:
            test_per_subject, test_overall = evaluate_per_subject(
                trainer.model, X_te, y_te, codes_te, test_subj_order, trainer.device)
            test_trial_per_subject, test_trial_overall = evaluate_per_subject_trials(
                trainer.model, X_te, y_te, codes_te, trials_te, test_subj_order, trainer.device)
            print(f"\n{config.TAGS.INFO.value} 測試受試者準確率 (epoch {best_epoch + 1}，未參與任何選擇)")
            for sub, acc in test_per_subject.items():
                print(f"  ├── subject {sub:>3}: {acc * 100:5.2f}%  (trial 多數決 {test_trial_per_subject[sub] * 100:5.2f}%)")
            print(f"  └── 整體 {test_overall * 100:.2f}%  (trial 多數決 {test_trial_overall * 100:.2f}%)")

    # --- 存檔 (鍵名與 generate_random_models / EEGPrediction 相容) ---
    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "n_chans": len(channel_index),
        "n_outputs": config.N_Class,
        "n_times": args.segment_len,
        "model_name": args.model,
        # --- 以下為可追溯性資訊，線上系統不會讀，但方便日後比對 ---
        "pretrain_meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "data_dir": args.data_dir,
            "mi_name": mi_name,
            "channels_mode": args.channels,
            "ch_names": ch_names,
            "raw_channel_index": ch_cfg["raw_channel_index"],
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "excluded_subjects": excluded,
            "split_mode": args.split_mode,
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "sessions": sessions,
            "runs": runs,
            "exclude_failed": args.exclude_failed,
            "segment_len": args.segment_len,
            "stride": args.stride,
            "sfreq": config.SAMPLE_RATE,
            "model_kwargs": model_kwargs,
            "epochs_run": len(hist["val_loss"]),
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "beta1": args.beta1,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "align": args.align,
            "best_epoch": best_epoch,
            "no_val": args.no_val,
            "best_val_loss": None if args.no_val else float(hist["val_loss"][best_epoch]),
            "best_val_acc": None if args.no_val else float(hist["val_acc"][best_epoch]),
            "heldout_overall_acc": overall,
            "heldout_per_subject_acc": per_subject,
            "select_by": args.select_by,
            "by_val_loss": by_loss,
            "by_val_acc": by_acc,
            "test_subjects": test_subjects,
            "test_per_subject": test_per_subject,
            "test_overall": test_overall,
            "test_trial_per_subject": test_trial_per_subject,
            "test_trial_overall": test_trial_overall,
            "test_stats": stats_te,
            "train_stats": stats_tr,
            "val_stats": stats_va,
            "history": hist,
        },
    }
    torch.save(checkpoint, out_path)
    print(f"\n💾 預訓練權重已儲存: {out_path}")

    # --- 學習曲線 ---
    if not args.no_plot:
        png_path = os.path.join(args.curve_dir, f"pretrain_{args.run_tag}.png")
        shown = val_subjects if len(val_subjects) <= 8 else val_subjects[:8]
        held = ", ".join(shown) + ("" if len(val_subjects) <= 8
                                   else f", +{len(val_subjects) - 8} more")
        if session_mode:
            tr_s, va_s = "+".join(train_sessions), "+".join(val_sessions)
            title = (f"{args.model} · {args.channels} ch · within-subject "
                     f"cross-session ({tr_s} → {va_s})")
            subtitle = (f"{len(val_subjects)} subjects ({held}) · train {tr_s} → val {va_s} · "
                        f"{stats_tr['n_windows']:,} train windows · val acc {overall * 100:.2f}%")
            labels = dict(train_label=f"Train ({tr_s})", val_label=f"Validation ({va_s})",
                          per_subject_ylabel=f"{va_s} accuracy\nper subject")
        elif args.no_val:
            title = f"{args.model} · {args.channels} ch · deployable model (all subjects, no held-out set)"
            subtitle = (f"{len(train_subjects)} train subjects · {stats_tr['n_windows']:,} train windows · "
                        f"fixed {args.epochs} epochs, last epoch saved")
            labels = dict(train_label="Train (all subjects)", val_label=None)
        else:
            title = f"{args.model} · {args.channels} ch · cross-subject pre-training"
            subtitle = (f"{len(train_subjects)} train subjects / {len(val_subjects)} held out "
                        f"({held}) · {stats_tr['n_windows']:,} train windows · "
                        f"held-out acc {overall * 100:.2f}%")
            labels = {}
        plot_learning_curve(hist, best_epoch, png_path, title, subtitle, **labels)
        print(f"📈 學習曲線已儲存: {png_path}")

    # --- 驗證線上系統載得進去 ---
    verify_model = MODEL_DICT[args.model](**model_kwargs)
    verify_model.load_state_dict(torch.load(out_path, map_location="cpu")["model_state_dict"])
    print(f"{config.TAGS.INFO.value} ✅ 權重可用 strict=True 重新載入，格式與 EEGPrediction 相容")

    # --- 部署提示 ---
    # 權重存在 Aden/pretrained_result/model_weights，不在線上系統的 checkpoint_main，
    # 所以 Unity 的模型清單 (game_state.set_global_model_name) 不會自動看到它。
    print(f"\n{config.TAGS.INFO.value} 要在線上系統使用這個預訓練模型：")
    print(f"  1. cp {out_path} \\\n       {config.EEG_CHECKPOINT_MAIN_BASE_FILE}")
    print(f"     (複製進 checkpoint_main 後，Unity 的模型清單才會出現它；"
          f"或直接覆蓋 c_000.pth 當作 Calibration fine-tune 的起點)")
    if len(channel_index) != config.N_CHANNELS:
        print(f"  2. ⚠️  把 python/config.json 的 active_channels 從 "
              f"{config.ACTIVE_CHANNEL_MODE} 改成 {args.channels}。")
        print(f"     目前設定是 {config.N_CHANNELS} 通道但模型是 {len(channel_index)} 通道，"
              f"不改的話 EEGPrediction 會判定維度不匹配並用隨機權重覆蓋掉它。")
    else:
        print(f"  2. config.json 的 active_channels ({config.ACTIVE_CHANNEL_MODE}) "
              f"與模型通道數一致，不用改。")

    # --- 清理暫存 ---
    if not args.keep_epoch_checkpoints:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"{config.TAGS.INFO.value} 已清理暫存資料夾 {work_dir} "
              f"(加 --keep_epoch_checkpoints 可保留每個 epoch 的權重)")

    return 0


def build_model_kwargs(model_name, n_chans, n_times, n_classes=None):
    """SCCNet 與 braindecode 的建構參數命名不同，這裡統一轉換。

    注意 sfreq 是取樣率 (500 Hz)，samples 是視窗長度，兩者不要混用
    (tools/quick_trainer.py 就把 samples 寫成 config.SAMPLE_RATE，只有在視窗剛好 500 點時才正確)。
    """
    n_classes = config.N_Class if n_classes is None else n_classes
    if model_name == "SCCNet":
        return dict(n_classes=n_classes, channels=n_chans,
                    samples=n_times, sfreq=config.SAMPLE_RATE)
    return dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)


def main():
    args = parse_args()
    args.run_tag = f"{args.channels}c_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.list:
        return run(args)  # 只列資料，不需要 log 檔

    # log 與學習曲線都放在 curve_dir，預設 Aden/pretrained_result/learning_curve
    os.makedirs(args.curve_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.curve_dir, f"pretrain_{args.run_tag}.txt")
    args.log_file = log_file
    return tee_log(log_file)(run)(args)


if __name__ == "__main__":
    sys.exit(main())
