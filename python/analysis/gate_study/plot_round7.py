# python python/analysis/gate_study/plot_round7.py --a <analysis_20261125 dir> --r2 <analysis_20261124/round2_gates_20261123> --out <dir>
"""
Round 7 (sprtk with SPRT 0.75 / 0.95 / 0.05 / 0.2) figures for Notion page 11. Style as plot_report.py / plot_round6.py:
blue = SPRT default without a gap, orange = sprtk with the default SPRT (round 6), aqua = sprtk with 0.75 / 0.95 (round 7),
gray = references. Unit = subject (seeds, then sessions averaged), n = 24.
  r7_budget.png    accuracy vs finetunes per session: clock curve, block, both SPRT settings with and without the gap
  r7_vs_block.png  paired difference vs block with 95% bootstrap CI, D2 margin (-1 pp) marked
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_report import AQUA, BLUE, ORANGE, INK, INK2, MUTED, finish  # noqa: E402  (also sets rcParams)
from plot_round6 import CLOCK, CLOCK_LABEL, boot, by_subject  # noqa: E402

DEF, HI = "0.65_0.85_0.05_0.2", "0.75_0.95_0.05_0.2"
R6 = {k: f"sprtk_{k}_{DEF}" for k in (4, 8, 12)}
R7 = {k: f"sprtk_{k}_{HI}" for k in (4, 8, 12)}


def budget(ps, out):
    s = by_subject(ps).groupby("condition").mean()
    pt = lambda c: (s.n_ft[c], 100 * s.acc[c])  # noqa: E731
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot(s.loc[CLOCK, "n_ft"], 100 * s.loc[CLOCK, "acc"], color=MUTED, lw=2, marker="o", ms=7, zorder=2)
    for c in CLOCK:
        if c == "never":
            continue
        off = (-6, 8) if c == "every_6" else (4, 8)
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
    # 沒有最少間隔的 SPRT：實心 = 預設，空心 = 0.75/0.95
    ax.plot(*pt(f"sprt_{DEF}"), marker="o", ms=10, color=BLUE, ls="none", zorder=4, mec="white", mew=2)
    ax.annotate("SPRT 預設", pt(f"sprt_{DEF}"), xytext=(8, -4), textcoords="offset points", fontsize=10, color=INK)
    ax.plot(*pt(f"sprt_{HI}"), marker="o", ms=10, mfc="white", mec=AQUA, mew=2.2, ls="none", zorder=4)
    ax.annotate(
        "SPRT 0.75/0.95（無間隔）",
        pt(f"sprt_{HI}"),
        xytext=(0, -18),
        textcoords="offset points",
        fontsize=10,
        color=INK,
        ha="center",
    )
    for k in (4, 8, 12):
        a, b = pt(R6[k]), pt(R7[k])
        ax.annotate("", b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=6, shrinkB=7))
        ax.plot(*a, marker="D", ms=8, color=ORANGE, ls="none", zorder=4, mec="white", mew=1.5)
        ax.plot(*b, marker="D", ms=9, color=AQUA, ls="none", zorder=5, mec="white", mew=1.5)
        off, ha = ((-9, 6), "right") if k == 12 else ((0, 9), "center")
        ax.annotate(f"k={k}", b, xytext=off, textcoords="offset points", fontsize=10, color=INK, ha=ha)
    ax.plot([], [], marker="D", ms=8, color=ORANGE, ls="none", label="sprtk，預設參數（第六輪）")
    ax.plot([], [], marker="D", ms=8, color=AQUA, ls="none", label="sprtk，0.75 / 0.95（第七輪）")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_xlabel("每個 session 的微調次數")
    ax.set_ylabel("線上正確率 (%)")
    ax.set_xlim(8, 72)
    ax.set_ylim(61.3, 66.3)
    finish(
        fig,
        ax,
        "換成觸發較多的參數，sprtk 貼近時鐘曲線，但沒有超過",
        "灰線 = 只按固定間隔微調；箭頭 = 同一個 k 從第六輪到第七輪；24 位受試者平均；y 軸從 61% 開始",
        os.path.join(out, "r7_budget.png"),
    )


def vs_block(ps, out):
    s = by_subject(ps)["acc"].unstack(0)
    rows = [(f"sprtk {k}（第六輪）", R6[k], ORANGE) for k in (4, 8, 12)]
    rows += [(f"sprtk {k}（第七輪）", R7[k], AQUA) for k in (4, 8, 12)]
    rows += [("every:12", "every_12", MUTED), ("every:8", "every_8", MUTED)]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axvspan(-1, 0, color="#eeeeea", zorder=0)
    ax.axvline(-1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate(
        "D2 容許範圍",
        (-1, len(rows) - 0.45),
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
            f"{d.mean():+.2f}", (d.mean(), y), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK2
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
        "第七輪的 k = 8 落在 D2 的容許範圍內，但仍比 block 低",
        "每位受試者配對相減（n = 24）；0 的右邊 = 比 block 好；灰底 = 比 block 差不超過 1 pp",
        os.path.join(out, "r7_vs_block.png"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="analysis_20261125 (round6_sprtk_20261124/, round7_sprtk_hi_20261125/)")
    ap.add_argument("--r2", required=True, help="analysis_20261124/round2_gates_20261123")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ps = pd.concat(
        [
            pd.read_csv(os.path.join(args.r2, "per_session.csv")),
            pd.read_csv(os.path.join(args.a, "round6_sprtk_20261124", "per_session.csv")),
            pd.read_csv(os.path.join(args.a, "round7_sprtk_hi_20261125", "per_session.csv")),
        ],
        ignore_index=True,
    )
    budget(ps, args.out)
    vs_block(ps, args.out)


if __name__ == "__main__":
    main()
