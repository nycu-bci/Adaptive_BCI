# python python/analysis/gate_study/plot_v2a.py --replica <v2a per_session.csv> --ref16 <round per_session.csv ...>
#        --out <dir>
"""
Vote check V2 phase a figure for the Notion page: always (= every:4) - X for the 10 SPRT variants, under the
16-window majority (gray, rounds 2 / 4 / 7 / 10) and under the replica of the online vote (blue), seeds 0-2.
Unit = subject (seeds averaged per session, sessions per subject), n = 24; 95% bootstrap CI.
  v2a_gaps.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_report import BLUE, INK, INK2, MUTED, finish  # noqa: E402  (also sets rcParams)
from plot_round6 import boot  # noqa: E402
from v2_decision import VARIANTS, subject_table  # noqa: E402

LABEL = {
    "sprtkb_4_0.75_0.95_0.05_0.2": "sprtkb k=4（第十輪）",
    "sprtkb_8_0.75_0.95_0.05_0.2": "sprtkb k=8",
    "sprtkb_12_0.75_0.95_0.05_0.2": "sprtkb k=12",
    "sprtk_4_0.75_0.95_0.05_0.2": "sprtk k=4（第七輪）",
    "sprtk_8_0.75_0.95_0.05_0.2": "sprtk k=8",
    "sprtk_12_0.75_0.95_0.05_0.2": "sprtk k=12",
    "sprt_0.65_0.85_0.05_0.2": "SPRT 線上預設 0.65/0.85",
    "sprt_0.55_0.95_0.05_0.2": "SPRT 0.55/0.95",
    "sprt_0.75_0.95_0.05_0.2": "SPRT 0.75/0.95",
    "sprtb_0.65_0.85_0.05_0.2": "sprtb（第四輪）",
}


def gaps(T, X):
    s = sorted(set(T["always"]) & set(T[X]))
    return np.array([T["always"][x] - T[X][x] for x in s]) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica", nargs="+", required=True)
    ap.add_argument("--ref16", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = {0, 1, 2}
    R, Q = subject_table(a.replica, seeds), subject_table(a.ref16, seeds)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    y = np.arange(len(VARIANTS))[::-1]
    for yi, X in zip(y, VARIANTS):
        for T, dy, col, lab in (
            (Q, 0.17, MUTED, "16 個 window 多數決（原本的模擬）"),
            (R, -0.17, BLUE, "重建的線上投票（V2）"),
        ):
            d = gaps(T, X)
            lo, hi = boot(d)
            ax.plot([lo, hi], [yi + dy] * 2, color=col, lw=2.2, solid_capstyle="round")
            ax.plot(
                d.mean(), yi + dy, "o", ms=7, color=col, mec="white", mew=1.2, label=lab if X == VARIANTS[0] else None
            )
        ax.annotate(
            f"{gaps(R, X).mean():+.2f}",
            (hi, yi - 0.17),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=BLUE,
            va="center",
        )
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[X] for X in VARIANTS], fontsize=10, color=INK)
    ax.set_xlabel("every:4 − 這個 SPRT 變體（pp，> 0 = every:4 較好）")
    ax.set_xlim(-0.6, 5.2)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    finish(
        fig,
        ax,
        "換成線上投票後，every:4 仍贏過每一種 SPRT",
        "受試者平均（n = 24），seed 0-2，線 = 95% bootstrap CI；藍色數字 = 重建投票下的差",
        os.path.join(a.out, "v2a_gaps.png"),
    )


if __name__ == "__main__":
    main()
