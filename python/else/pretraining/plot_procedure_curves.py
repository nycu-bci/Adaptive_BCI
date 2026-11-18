"""
從嚴格流程 (run_procedure) 每個 run 存下的 history 重畫學習曲線，不需要重新訓練。
每個 run 一張：train / 驗證受試者的 loss 與 accuracy，以及每位驗證受試者的 accuracy；
測試受試者只在選定 epoch 評估一次，所以寫在副標題，不是曲線。

用法：python python/else/pretraining/plot_procedure_curves.py [--setting SCCNet_lr1e-3_bs64_dodef_wd0_st100_s42] [--fold 1]
輸出：procedure/runs/<設定>/fold<k>.png (與 .pth 同一層)
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)
from pretrain_cross_subject import plot_learning_curve

DEFAULT_DIR = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "procedure", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--setting", default=None, help="只畫某個設定資料夾")
    ap.add_argument("--fold", default=None, help="只畫某個 fold，例如 1")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.setting or "*", f"fold{args.fold or '*'}.pth")))
    if not paths:
        raise SystemExit(f"❌ 在 {args.dir} 找不到符合的 run")
    for i, path in enumerate(paths, 1):
        setting = os.path.basename(os.path.dirname(path))
        fold = os.path.basename(path)[4:-4]
        m = torch.load(path, map_location="cpu", weights_only=False)["pretrain_meta"]
        best = m["best_epoch"]
        test = m["test_subjects"]
        win = np.mean([m["test_per_subject"][s] for s in test]) * 100
        tri = np.mean([m["test_trial_per_subject"][s] for s in test]) * 100
        title = f"{setting} · fold {fold}"
        subtitle = (f"train {len(m['train_subjects'])} · val {','.join(m['val_subjects'])} · "
                    f"epoch {best + 1} chosen by val loss · test {','.join(test)}: "
                    f"mean {win:.2f}% window, {tri:.2f}% trial (scored once)")
        plot_learning_curve(m["history"], best, path.replace(".pth", ".png"), title, subtitle,
                            train_label=f"Train ({len(m['train_subjects'])} subjects)",
                            val_label="Validation subjects (early stopping)",
                            per_subject_ylabel="Validation accuracy\nper subject")
        if i % 25 == 0 or i == len(paths):
            print(f"{i}/{len(paths)} done", flush=True)


if __name__ == "__main__":
    main()
