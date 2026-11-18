"""
彙整 group-pool 實驗 (campaign16)：每個測試受試者在各訓練池 × 學習率下的準確率。

輸出 (group_pool/ 底下)：
  results.csv   每個 (測試受試者, 池, 學習率) 一列
  summary.txt   池 × 學習率平均、依測試受試者組別、成對比較、依驗證 loss 選出的學習率
  group_pool.png
RAND 池 = 3 次隨機抽樣的平均。實驗未跑完也能執行 (表格會標出 n)。

用法：python python/else/pretraining/analyze_group_pool.py
"""
import os
import csv
import json
import glob
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
DEFAULT_DIR = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "group_pool")

POOLS = ["ALL", "HIGH", "LOW", "RAND"]
TEST_GROUPS = ["HIGH", "LOW", "LH", "HL"]
COLORS = {"ALL": "#2a78d6", "HIGH": "#eb6834", "LOW": "#1baf7a", "RAND": "#eda100"}
MARKERS = {"ALL": "o", "HIGH": "s", "LOW": "^", "RAND": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def load(exp_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "*_lr*", "test*.pth"))):
        pool, lr = os.path.basename(os.path.dirname(path)).split("_lr")
        m = torch.load(path, map_location="cpu", weights_only=False)["pretrain_meta"]
        test = m["test_subjects"][0]
        rows.append(dict(test=test, pool=pool, lr=lr, test_acc=m["test_per_subject"][test],
                         val_loss=m["best_val_loss"], best_epoch=m["best_epoch"] + 1,
                         epochs_run=m["epochs_run"], train_subjects=len(m["train_subjects"])))
    return rows


def table(rows, lrs, subjects):
    """acc[pool][lr][test] (RAND = 3 次抽樣平均)；vloss 同樣結構。"""
    acc = {p: {lr: {} for lr in lrs} for p in POOLS}
    vloss = {p: {lr: {} for lr in lrs} for p in POOLS}
    tmp = {}
    for r in rows:
        fam = "RAND" if r["pool"].startswith("RAND") else r["pool"]
        tmp.setdefault((fam, r["lr"], r["test"]), []).append((r["test_acc"], r["val_loss"]))
    for (fam, lr, test), v in tmp.items():
        if fam == "RAND" and len(v) < 3:
            continue                                   # 3 次抽樣都完成才算
        acc[fam][lr][test] = float(np.mean([a for a, _ in v]))
        vloss[fam][lr][test] = float(np.mean([l for _, l in v]))
    return acc, vloss


def fmt_cell(d, subs):
    vals = [d[s] for s in subs if s in d]
    if not vals:
        return "    —     "
    return f"{np.mean(vals) * 100:5.2f} n{len(vals):<2d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.dir, "groups.json")))
    lrs, groups = info["lrs"], info["groups"]
    subjects = sorted(sum(groups.values(), []), key=int)
    group_of = {s: g for g, ss in groups.items() for s in ss}

    rows = load(args.dir)
    for r in rows:
        r["test_group"] = group_of[r["test"]]
    with open(os.path.join(args.dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "test_group", "pool", "lr", "test_acc", "val_loss",
                                          "best_epoch", "epochs_run", "train_subjects"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (int(r["test"]), r["pool"], lrs.index(r["lr"]))):
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    acc, vloss = table(rows, lrs, subjects)
    out = []
    out.append(f"Group-pool experiment: {len(rows)}/660 runs found. Test accuracy = window accuracy of the held-out "
               f"test subject at the epoch chosen by validation loss (mean over test subjects; n = test subjects).")
    out.append(f"Groups (22-ch within-subject accuracy, medians s1 {info['median_s1']:.3f} s2 {info['median_s2']:.3f}): "
               + "; ".join(f"{g} {groups[g]}" for g in TEST_GROUPS))

    out.append("\n1) Mean test accuracy (%), all test subjects")
    out.append("pool  | " + " | ".join(f"lr {lr:>5s}  " for lr in lrs))
    for p in POOLS:
        out.append(f"{p:5s} | " + " | ".join(fmt_cell(acc[p][lr], subjects) for lr in lrs))

    out.append("\n2) Mean test accuracy (%) by test subject's group")
    for g in TEST_GROUPS:
        out.append(f"-- test group {g} ({len(groups[g])}: {','.join(groups[g])})")
        for p in POOLS:
            out.append(f"   {p:5s} | " + " | ".join(fmt_cell(acc[p][lr], groups[g]) for lr in lrs))

    out.append("\n3) Paired differences (points), same test subjects in both pools; 'better' = test subjects improved")
    for a, b in [("HIGH", "ALL"), ("HIGH", "RAND"), ("LOW", "RAND"), ("LOW", "ALL")]:
        cells = []
        for lr in lrs:
            common = [s for s in subjects if s in acc[a][lr] and s in acc[b][lr]]
            if not common:
                cells.append("        —          ")
                continue
            d = np.array([acc[a][lr][s] - acc[b][lr][s] for s in common]) * 100
            cells.append(f"{d.mean():+5.2f} better {int((d > 0).sum()):2d}/{len(common):<2d}")
        out.append(f"{a:4s}-{b:4s} | " + " | ".join(cells))

    out.append("\n4) Learning rate chosen by mean validation loss (never by test accuracy)")
    for p in POOLS:
        full = [lr for lr in lrs if len(vloss[p][lr]) == len(subjects)]
        if not full:
            out.append(f"{p:5s} | not all test subjects finished yet")
            continue
        best = min(full, key=lambda lr: np.mean(list(vloss[p][lr].values())))
        vl = ", ".join(f"{lr} {np.mean(list(vloss[p][lr].values())):.4f}" for lr in full)
        out.append(f"{p:5s} | chosen lr {best} -> test {np.mean(list(acc[p][best].values())) * 100:.2f}%   "
                   f"(mean val loss: {vl})")

    text = "\n".join(out)
    open(os.path.join(args.dir, "summary.txt"), "w").write(text + "\n")
    print(text)

    # 圖：x = 學習率，線 = 訓練池；左 = 全部測試受試者，右 4 張 = 依測試受試者組別
    panels = [("All test subjects", subjects)] + [(f"Test group {g}", groups[g]) for g in TEST_GROUPS]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), sharey=True)
    x = np.arange(len(lrs))
    for ax, (title, subs) in zip(axes, panels):
        for p in POOLS:
            y = [np.mean([acc[p][lr][s] for s in subs if s in acc[p][lr]]) * 100
                 if any(s in acc[p][lr] for s in subs) else np.nan for lr in lrs]
            ax.plot(x, y, color=COLORS[p], marker=MARKERS[p], ms=7, lw=2, label=p)
        ax.axhline(50, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(lrs)
        ax.set_xlabel("Learning rate", color=INK)
        ax.set_title(f"{title} (n={len(subs)})", color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Test subject accuracy (%)", color=INK)
    axes[0].legend(title="Training pool", frameon=False)
    fig.suptitle("Which training pool gives the best model for an unseen subject? "
                 "(leave-one-subject-out, RAND = mean of 3 draws, same size as HIGH)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "group_pool.png"), dpi=150, facecolor="white")
    print(f"\nsaved {os.path.join(args.dir, 'group_pool.png')}")


if __name__ == "__main__":
    main()
