"""
VR-BCI 深度學習模型訓練與在線校正微調核心 (Braindecode & Online Adaptation Trainer)
包含離線/在線訓練器 BraindecodeTrainer 以及基於 Replay Buffer 的在線增量校正器 OnlineCalibrationTrainer。
"""

import os
import sys
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# 自動路徑解析
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
import main.Utils.global_value as global_value


class BraindecodeTrainer:
    """通用 BCI 深度學習模型訓練器，支援 Early Stopping 與 TCP 即時訓練進度廣播"""

    def __init__(self, dataset, val_dataset=None, model_class=config.USE_MODEL, model_kwargs=None,
                 batch_size=16, num_epochs=100, lr=1e-4, device=None, ft=False):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.ft = ft

        self._load_data()
        self._init_model()

    def _load_data(self):
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False) if self.val_dataset else None

    def _init_model(self):
        """初始化目標模型架構、損失函數與優化器"""
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def load_checkpoint(self, checkpoint_path):
        """載入 Checkpoint，若偵測到通道數或維度不匹配則自動生成隨機權重修復"""
        if not os.path.exists(checkpoint_path):  # 只有檔案不存在才產生隨機權重；維度不匹配會直接報錯，不再覆蓋檔案
            print(f"\n⚠️ [WARNING] 找不到檔案: {checkpoint_path}，產生隨機初始權重。")
            from main.EEG.generate_random_models import create_matching_random_checkpoint
            create_matching_random_checkpoint(
                target_path=checkpoint_path,
                model_class=self.model_class,
                n_chans=config.N_CHANNELS,
                n_outputs=config.N_Class,
                n_times=config.SAMPLE_RATE
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        print(f"{config.TAGS.INFO.value} 成功載入模型權重: {checkpoint_path}")

    def _evaluate(self, loader):
        """評估驗證集 Loss 與 Accuracy"""
        if loader is None:
            return 0.0, 0.0
        self.model.eval()
        total_correct, total_samples, losses = 0, 0, []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device).float(), y_batch.to(self.device).float()
                output = self.model(x_batch.squeeze(1))
                loss = self.criterion(output, y_batch)
                losses.append(loss.item())

                preds = output.argmax(dim=1)
                total_correct += (preds == y_batch.argmax(dim=1)).sum().item()
                total_samples += y_batch.size(0)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
        return mean_loss, accuracy

    def train(self, freeze_layer=False, use_batch_norm=False, tcp_server=None, patience=30):
        """執行模型訓練迴圈"""
        os.makedirs(config.EEG_CHECKPOINT_TMP_BASE_FILE, exist_ok=True)
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        tag = "ft" if self.ft else "train"

        best_val_loss = float('inf')
        early_stop_counter = 0

        # 若指定凍結特徵提取層，僅微調分類層
        if freeze_layer:
            final_layer = getattr(self.model, 'classifier', getattr(self.model, 'final_layer', None))
            if final_layer is None:
                raise AttributeError(f"模型 {self.model_class} 無法找到 'classifier' 或 'final_layer' 屬性進行微調")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in final_layer.parameters():
                param.requires_grad = True
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.lr, betas=(0.5, 0.999))

        for epoch in range(self.num_epochs):
            self.model.train()

            if freeze_layer and use_batch_norm:
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs.squeeze(1))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

            train_loss = running_loss / len(self.dataset) if len(self.dataset) > 0 else 0.0
            train_acc = (correct / total) if total > 0 else 0.0

            val_loss, val_acc = self._evaluate(self.val_loader) if self.val_loader else (train_loss, train_acc)

            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 儲存當前 Epoch Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{tag}-epoch{epoch}.pth"))

            # 透過 TCP 廣播即時進度給 Unity
            if tcp_server:
                tcp_server.broadcast(f"Epoch {epoch + 1}/{self.num_epochs} - "
                                     f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
                                     f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save({'model_state_dict': self.model.state_dict()},
                           os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{tag}-best.pth"))
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                stop_msg = f"Early stopping triggered at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}"
                print(stop_msg)
                if tcp_server:
                    tcp_server.broadcast(stop_msg)
                break

        if tcp_server:
            time.sleep(0.5)
            tcp_server.broadcast(config.TRAINING_FINISH_STR)

        return history


def load_shallowfbcsp_params(dataset):
    """獲取 ShallowFBCSPNet 輸入維度超參數"""
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    return dict(
        n_chans=data_shape[0][1],
        n_outputs=data_shape[1][0],
        n_times=data_shape[0][2]
    )


def load_sccnet_params(dataset):
    """獲取 SCCNet 輸入維度超參數"""
    data_shape = [dataset.__getitem__(0)[0].shape, dataset.__getitem__(0)[1].shape]
    return dict(
        samples=data_shape[0][2],
        channels=data_shape[0][1],
        n_classes=data_shape[1][0],
        sfreq=config.SAMPLE_RATE
    )


class ReplayBuffer:
    """
    單一模型的 Replay Buffer (只存資料，不訓練)：每個類別最多 limit 個 window 的 FIFO，
    被擠出或新進的 window 同時進驗證部分 (上限 limit 的 40%)。失敗 trial 的 window 權重加倍。
    """

    def __init__(self, limit=None, use_val=None, fail_weight=2.0):
        self._limit, self._use_val = limit, use_val
        self.fail_weight = fail_weight
        self.train = {0: [], 1: []}  # label -> [(x, y, weight)]，最舊的在前
        self.val = {0: [], 1: []}

    @property
    def limit(self):
        return self._limit if self._limit is not None else config.REPLAY_BUFFER_LIMIT

    @property
    def use_val(self):
        return self._use_val if self._use_val is not None else config.adaption_use_val

    def add(self, dataset):
        """dataset: TensorDataset(x, y_onehot, failure)。FIFO 規則與原本 OnlineCalibrationTrainer.add_to_buffer 相同"""
        if dataset is None or len(dataset) == 0:
            return
        xs, ys, fails = dataset.tensors
        for i in range(len(xs)):
            label = int(torch.argmax(ys[i]).item())
            item = (xs[i].cpu(), ys[i].cpu(), self.fail_weight if fails[i].item() > 0.5 else 1.0)
            self.train.setdefault(label, []).append(item)
            self.val.setdefault(label, [])
            if len(self.train[label]) > self.limit:  # 超過上限：最舊的移到驗證部分
                oldest = self.train[label].pop(0)
                if self.use_val:
                    self.val[label].append(oldest)
            elif self.use_val:
                self.val[label].append(item)
            if self.use_val and len(self.val[label]) > int(self.limit * 0.4):
                self.val[label].pop(0)

    @staticmethod
    def _to_dataset(parts):
        items = [item for label in parts for item in parts[label]]
        if not items:
            return None
        return TensorDataset(torch.stack([it[0] for it in items]), torch.stack([it[1] for it in items]),
                             torch.tensor([it[2] for it in items], dtype=torch.float32))

    def train_set(self):
        return self._to_dataset(self.train)

    def val_set(self):
        return self._to_dataset(self.val) if self.use_val else None

    def sizes(self):
        """(每個類別的訓練 window 數, 每個類別的驗證 window 數)"""
        return {k: len(v) for k, v in self.train.items()}, {k: len(v) for k, v in self.val.items()}


class OnlineCalibrationTrainer(BraindecodeTrainer):
    """在線校正微調器：只負責訓練 (資料由 ReplayBuffer 提供)，支援失敗經驗加權"""

    def __init__(self, dataset=None, val_dataset=None, model_class=config.USE_MODEL, model_kwargs=None,
                 batch_size=16, num_epochs=100, lr=1e-4, device=None, ft=False,
                 tag=None, seed=None):
        super().__init__(dataset, val_dataset, model_class, model_kwargs,
                         batch_size, num_epochs, lr, device, ft)
        self.online_lr = lr
        self.use_val = config.adaption_use_val
        # 雙模型 A/B 各自的暫存 checkpoint 名稱與 shuffle 種子
        self.tag = tag or ("ft" if ft else "train")
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def _load_data(self):
        pass  # 數據由 train 傳入

    def _init_model(self):
        super()._init_model()
        self.criterion_none = nn.CrossEntropyLoss(reduction='none')

    def train(self, train_set, val_set=None):
        """在 train_set (x, y, weight) 上微調 num_epochs 個 epoch，val_set 上 loss 最低的 epoch 權重存成 {tag}-best.pth"""
        if train_set is None or len(train_set) == 0:
            print(f"{config.TAGS.WARNING.value} 無有效線上訓練資料")
            return 0.0

        start_time = time.perf_counter()
        tag = self.tag
        self.model.train()

        online_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, generator=self.gen)
        optimizer = optim.Adam(self.model.parameters(), lr=self.online_lr)

        val_loader = None
        if self.use_val and val_set is not None and len(val_set) > 0:
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        best_loss = float('inf')
        best_model_state = None
        last_avg_loss = 0.0

        # 微調迴圈
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for inputs, labels, weights in online_loader:
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).float()
                weights = weights.to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(inputs.squeeze(1))
                target_indices = torch.argmax(labels, dim=1)

                loss_per_sample = self.criterion_none(outputs, target_indices)
                weighted_loss = (loss_per_sample * weights).mean()

                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item()

            last_avg_loss = total_loss / len(online_loader) if len(online_loader) > 0 else 0.0
            current_eval_loss = last_avg_loss

            if self.use_val and val_loader is not None:
                self.model.eval()
                val_total_loss = 0.0
                with torch.no_grad():
                    for v_inputs, v_labels, v_weights in val_loader:
                        v_inputs = v_inputs.to(self.device).float()
                        v_labels = v_labels.to(self.device).float()
                        v_weights = v_weights.to(self.device).float()
                        v_outputs = self.model(v_inputs.squeeze(1))
                        v_target = torch.argmax(v_labels, dim=1)
                        v_loss = (self.criterion_none(v_outputs, v_target) * v_weights).mean()
                        val_total_loss += v_loss.item()

                current_eval_loss = val_total_loss / len(val_loader) if len(val_loader) > 0 else current_eval_loss
                self.model.train()

            if current_eval_loss < best_loss:
                best_loss = current_eval_loss
                best_model_state = copy.deepcopy(self.model.state_dict())

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': last_avg_loss,
            }, os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{tag}-epoch{epoch}.pth"))

        print(f"{config.TAGS.INFO.value} 線上微調完成，平均 Loss: {last_avg_loss:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            torch.save({'model_state_dict': best_model_state},
                       os.path.join(config.EEG_CHECKPOINT_TMP_BASE_FILE, f"{tag}-best.pth"))

        if config.verbose:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"{config.TAGS.INFO.value} [Online Adaptation 耗時] {elapsed_ms:.2f} ms")

        return last_avg_loss
