"""
從 group-pool 實驗 (campaign16) 每個 run 存下的 history 重畫學習曲線，不需要重新訓練。
每個 run 一張：train / 驗證受試者的 loss 與 accuracy，以及每位驗證受試者的 accuracy；
測試受試者只在選定 epoch 評估一次，所以寫在副標題，不是一條曲線。

用法：python python/else/pretraining/plot_group_pool_curves.py [--pool HIGH_lr1e-3] [--test 44]
輸出：group_pool/<pool>_lr<lr>/test<id>.png (與 .pth 同一層)
"""
import os
import sys
import glob
import argparse

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)
from pretrain_cross_subject import plot_learning_curve

DEFAULT_DIR = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "group_pool")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--pool", default=None, help="只畫某個資料夾，例如 HIGH_lr1e-3")
    ap.add_argument("--test", default=None, help="只畫某位測試受試者，例如 44")
    args = ap.parse_args()

    pattern = os.path.join(args.dir, args.pool or "*_lr*", f"test{args.test or '*'}.pth")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"❌ 找不到 {pattern}")
    for i, path in enumerate(paths, 1):
        pool, lr = os.path.basename(os.path.dirname(path)).split("_lr")
        m = torch.load(path, map_location="cpu", weights_only=False)["pretrain_meta"]
        test = m["test_subjects"][0]
        best = m["best_epoch"]
        title = f"{pool} pool · lr {lr} · test subject {test}"
        tr = m["train_subjects"]
        tr_txt = ",".join(tr) if len(tr) <= 8 else "all except val and test"
        subtitle = (f"train {len(tr)} ({tr_txt}) · "
                    f"val {','.join(m['val_subjects'])} · epoch {best + 1} chosen by val loss · "
                    f"test subject {test}: {m['test_per_subject'][test] * 100:.2f}% (scored once at that epoch)")
        plot_learning_curve(m["history"], best, path.replace(".pth", ".png"), title, subtitle,
                            train_label=f"Train ({len(m['train_subjects'])} subjects)",
                            val_label="Validation subjects (early stopping)",
                            per_subject_ylabel="Validation accuracy\nper subject")
        if i % 50 == 0 or i == len(paths):
            print(f"{i}/{len(paths)} done", flush=True)


if __name__ == "__main__":
    main()
