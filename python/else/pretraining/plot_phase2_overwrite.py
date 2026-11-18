"""
視覺化「校正多快蓋掉起始模型」：pretrained (每位受試者用沒看過他的 fold 模型) vs 10 個 random model。

讀 phase2_calibration_sim.py 的輸出 (calibration_ab.json)，由每個 trial 的對錯重建 previous gate
(每 4 trial 一個 block；adaptive 且 block acc < 0.75 且 run acc < 0.75 就校正)，並核對 n_calibrations。

(a) session 1 run 1-3 每個 4-trial block 的正確率 (22 位受試者平均)；run 1-3 在 A/B 兩種條件下都會校正
(b) pretrained − random 的差距 (灰線為每個 seed)；(c) 同一段時間內已完成的校正次數

用法: python plot_phase2_overwrite.py [--json .../calibration_ab.json]
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Aden")
DEFAULT_JSON = os.path.join(HERE, "pretrained_result", "experiments", "phase2_calibration", "calibration_ab.json")
BLOCK, GATE = 4, 0.75
N_RUNS_A = 3
C_DEPLOY, C_RANDOM, C_SEED = "#2a78d6", "#eb6834", "#b5b4ae"
INK, INK2 = "#0b0b0b", "#52514e"


def blocks_of_run(run):
    """回傳每個 block 的 (正確率, 是否校正)，並核對重建的校正次數。"""
    trials = run["trials"]
    out, correct = [], 0
    for b in range(len(trials) // BLOCK):
        blk = trials[b * BLOCK:(b + 1) * BLOCK]
        correct += sum(blk)
        acc = np.mean(blk)
        calib = run["adaptive"] and acc < GATE and correct / ((b + 1) * BLOCK) < GATE
        out.append((acc, calib))
    assert sum(c for _, c in out) == run["n_calibrations"], "gate reconstruction mismatch"
    return out


def per_record(rec):
    """session 1 run 1-3 的 block 正確率與累積校正次數；所有 run 都核對 gate 重建。"""
    s1 = [r for r in rec["runs"] if r["session"] == "s1"]
    acc_seq, ncal_seq = [], []
    k = 0
    for run in s1:
        for acc, calib in blocks_of_run(run):
            if run["real_run"] <= N_RUNS_A:
                acc_seq.append(acc)
            k += int(calib)
            if run["real_run"] <= N_RUNS_A:
                ncal_seq.append(k)
    for run in [r for r in rec["runs"] if r["session"] == "s2"]:
        blocks_of_run(run)                # 只做核對
    return np.array(acc_seq), np.array(ncal_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args()
    res = json.load(open(args.json))["results"]

    # [arm][seed][subject] -> per_record
    data = defaultdict(lambda: defaultdict(dict))
    for rec in res:
        if rec["arm"] in ("pretrained", "random"):
            data[rec["arm"]][rec["seed"]][rec["subject"]] = per_record(rec)
    subjects = sorted(next(iter(data["pretrained"].values())))
    seeds = sorted(data["random"])

    def seq_mean(arm, seed_list, idx):
        return np.mean([[data[arm][s][u][idx] for u in subjects] for s in seed_list], axis=(0, 1))

    n_blocks = len(seq_mean("random", seeds, 0))
    x = np.arange(1, n_blocks + 1)
    per_run = n_blocks // N_RUNS_A

    fig, (ax_a, ax_g, ax_b) = plt.subplots(3, 1, figsize=(13, 10.5), sharex=True,
                                           gridspec_kw=dict(height_ratios=[3, 2, 1.1], hspace=0.1))
    dep_seeds = sorted(data["pretrained"])

    # (a) block 正確率
    for i, s in enumerate(seeds):
        ax_a.plot(x, seq_mean("random", [s], 0) * 100, color=C_SEED, lw=1, alpha=0.9,
                  label="Random, each of 10 seeds" if i == 0 else None, zorder=1)
    rnd_a = seq_mean("random", seeds, 0) * 100
    dep_a = seq_mean("pretrained", dep_seeds, 0) * 100
    ax_a.plot(x, rnd_a, color=C_RANDOM, lw=2.2, marker="o", ms=4, label="Random, mean of 10 seeds", zorder=3)
    ax_a.plot(x, dep_a, color=C_DEPLOY, lw=2.2, marker="o", ms=4, label="Pretrained (held-out fold model)", zorder=4)
    ax_a.axhline(50, color=INK2, lw=0.8, ls=":")
    ax_a.set_ylabel("Block accuracy (%)\n(4 trials, mean of 22 subjects)", color=INK)
    ax_a.set_title("(a) Accuracy of each 4-trial block, session 1 runs 1–3 (every subject calibrates here under A and B)",
                   loc="left", color=INK, fontsize=11)
    ax_a.legend(loc="lower right", frameon=False, ncol=3)
    ax_a.set_ylim(30, 100)

    # (g) 差距：deployment − 每個 random seed
    for i, s in enumerate(seeds):
        ax_g.plot(x, dep_a - seq_mean("random", [s], 0) * 100, color=C_SEED, lw=1, alpha=0.9, zorder=1,
                  label="Pretrained − random seed i" if i == 0 else None)
    ax_g.plot(x, dep_a - rnd_a, color=C_DEPLOY, lw=2.2, marker="o", ms=4, zorder=3,
              label="Pretrained − random mean")
    ax_g.axhline(0, color=INK, lw=1)
    ax_g.set_ylabel("Accuracy gap\n(points)", color=INK)
    ax_g.set_title("(b) Head start of the pretrained start", loc="left", color=INK, fontsize=11)
    ax_g.legend(loc="upper right", frameon=False, ncol=2)

    # (c) 累積校正次數
    rnd_b = seq_mean("random", seeds, 1)
    dep_b = seq_mean("pretrained", dep_seeds, 1)
    ax_b.step(x, dep_b, where="post", color=C_DEPLOY, lw=2, label="Pretrained")
    ax_b.step(x, rnd_b, where="post", color=C_RANDOM, lw=2, label="Random (mean)")
    ax_b.set_ylabel("Calibrations\ndone (mean)", color=INK)
    ax_b.set_title("(c) Calibrations completed by the end of each block", loc="left", color=INK, fontsize=11)
    ax_b.set_xlabel("4-trial block (10 blocks per run)", color=INK)
    ax_b.legend(loc="upper left", frameon=False)
    for k in [3, 6]:
        j = int(np.argmax(dep_b >= k))
        for ax in (ax_a, ax_g, ax_b):
            ax.axvline(x[j] + 0.5, color=C_DEPLOY, lw=0.8, ls="--", alpha=0.6)
        ax_a.text(x[j] + 0.6, 97, f"≈{k} calibrations\n(pretrained)", fontsize=8, color=INK2, va="top")

    for ax in (ax_a, ax_g, ax_b):
        for r in range(1, N_RUNS_A):
            ax.axvline(r * per_run + 0.5, color=INK, lw=1)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        ax.set_xlim(0.5, n_blocks + 0.5)
    for r in range(N_RUNS_A):
        ax_a.text((r + 0.5) * per_run + 0.5, 88, f"Run {r + 1}", ha="center", color=INK, fontsize=11, weight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str((i % per_run) + 1) for i in range(n_blocks)], fontsize=8)

    fig.suptitle("Does calibration overwrite the starting model?  Pretrained vs 10 random SCCNets "
                 "(22 subjects, each started from the fold model that never saw them; pretrained line = mean of 10 calibration seeds)", color=INK, fontsize=12)
    out = os.path.join(os.path.dirname(args.json), "overwrite_deploy_vs_random.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)

    # 數字摘要
    print("block  deploy  random  gap   deploy_calibs")
    for i in range(n_blocks):
        print(f"{i + 1:5d} {dep_a[i]:7.1f} {rnd_a[i]:7.1f} {dep_a[i] - rnd_a[i]:+5.1f} {dep_b[i]:6.2f}")


if __name__ == "__main__":
    main()
