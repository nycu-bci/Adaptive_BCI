# python python/analysis/gate_study/plot_round10.py --ps <analysis_20261125/round2plus7plus10/per_session.csv> --out <dir>
"""
Round 10 (sprtkb: sprtk with SPRT 0.75 / 0.95 / 0.05 / 0.2, finetune only at 4-trial block ends) figures for Notion
page 11. Style as plot_round7.py: aqua = round 7 sprtk (same SPRT, any trial), blue = round 10 sprtkb, gray =
references. Unit = subject (seeds, then sessions averaged), n = 24.
  r10_budget.png    accuracy vs finetunes per session: clock curve, block, round 7 -> round 10 at the same k
  r10_vs_block.png  paired difference vs block with 95% bootstrap CI, D2 margin (-1 pp) marked
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plot_report import AQUA, BLUE, INK, INK2, MUTED, finish  # noqa: E402  (also sets rcParams)
from plot_round6 import CLOCK, CLOCK_LABEL, boot, by_subject  # noqa: E402

HI = "0.75_0.95_0.05_0.2"
KS = (4, 8, 12)
R7 = {k: f"sprtk_{k}_{HI}" for k in KS}
R10 = {k: f"sprtkb_{k}_{HI}" for k in KS}


def budget(ps, out):
    s = by_subject(ps).groupby("condition").mean()
    pt = lambda c: (s.n_ft[c], 100 * s.acc[c])  # noqa: E731
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot(s.loc[CLOCK, "n_ft"], 100 * s.loc[CLOCK, "acc"], color=MUTED, lw=2, marker="o", ms=7, zorder=2)
    for c in CLOCK:
        if c in ("never", "every_20"):
            continue
        off = (7, 4) if c == "every_6" else (4, 8)
        ax.annotate(
            CLOCK_LABEL[c],
            pt(c),
            xytext=off,
            textcoords="offset points",
            fontsize=9,
            color=INK2,
            ha="right" if off[0] < 0 else "left",
        )
    ax.plot(*pt("block"), marker="s", ms=9, color=INK2, ls="none", zorder=3)
    ax.annotate("block", pt("block"), xytext=(8, -14), textcoords="offset points", fontsize=10, color=INK)
    for k in KS:
        a, b = pt(R7[k]), pt(R10[k])
        ax.annotate("", b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=6, shrinkB=7))
        ax.plot(*a, marker="D", ms=8, color=AQUA, ls="none", zorder=4, mec="white", mew=1.5)
        ax.plot(*b, marker="D", ms=9, color=BLUE, ls="none", zorder=5, mec="white", mew=1.5)
        ax.annotate(f"k={k}", b, xytext=(0, 9), textcoords="offset points", fontsize=10, color=INK, ha="center")
    ax.plot([], [], marker="D", ms=8, color=AQUA, ls="none", label="sprtk：任何 trial 都可微調（第七輪）")
    ax.plot([], [], marker="D", ms=8, color=BLUE, ls="none", label="sprtkb：只在 4-trial block 結束時（第十輪）")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_xlabel("每個 session 的微調次數")
    ax.set_ylabel("線上正確率 (%)")
    ax.set_xlim(14, 72)
    ax.set_ylim(62.8, 66.3)
    finish(
        fig,
        ax,
        "對齊 block 結束讓 k = 4 追上 Block，但仍在時鐘曲線下方",
        "灰線 = 只按固定間隔微調；箭頭 = 同一個 k 從第七輪到第十輪；SPRT 0.75 / 0.95；24 位受試者平均；y 軸從 62.8% 開始",
        os.path.join(out, "r10_budget.png"),
    )


def vs_block(ps, out):
    s = by_subject(ps)["acc"].unstack(0)
    rows = [(f"sprtk {k}（第七輪）", R7[k], AQUA) for k in KS]
    rows += [(f"sprtkb {k}（第十輪）", R10[k], BLUE) for k in KS]
    rows += [("every:12", "every_12", MUTED), ("every:8", "every_8", MUTED), ("every:4", "always", MUTED)]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axvspan(-1, 0, color="#eeeeea", zorder=0)
    ax.axvline(-1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate(
        "D2 容許範圍",
        (-1, -0.4),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK2,
        va="center",
    )
    for i, (lab, c, col) in enumerate(rows):
        d = 100 * (s[c] - s["block"]).dropna()
        lo, hi = boot(d)
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(d.mean(), y, marker="o", ms=9, color=col, mec="white", mew=2, zorder=3)
        ax.annotate(
            f"{d.mean():+.2f}",
            (d.mean(), y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK2,
        )
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("正確率 − block (pp)，點 = 平均，線 = 95% 信賴區間")
    finish(
        fig,
        ax,
        "對齊 block 結束：k = 4 進步最多，但沒有一個 k 贏 Block",
        "每位受試者配對相減（n = 24）；0 的右邊 = 比 block 好；灰底 = 比 block 差不超過 1 pp",
        os.path.join(out, "r10_vs_block.png"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True, help="analysis_20261125/round2plus7plus10/per_session.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ps = pd.read_csv(args.ps, dtype={"subject": str})
    budget(ps, args.out)
    vs_block(ps, args.out)


if __name__ == "__main__":
    main()
