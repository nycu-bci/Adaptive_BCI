# python python/analysis/gate_study/plot_round6.py --a <analysis_20261125 dir> --r2 <analysis_20261124/round2_gates_20261123> --out <dir>
"""
Round 6 (sprtk) figures for the Notion report. Same style as plot_report.py: blue = SPRT default, orange = sprtk,
gray = references (clock, block); every series direct-labeled. Unit = subject (seeds, then sessions averaged), n = 24.
  r6_budget.png    accuracy vs finetunes per session: clock curve, block, SPRT, sprtk 4/8/12
  r6_vs_block.png  paired difference vs block, mean and 95% bootstrap CI over subjects
  r6_gap.png       per finetune: mean gap since the previous finetune vs mean effect on the next 48 trials
  r6_terciles.png  sprtk_8 vs every:12 by subject tercile (never-accuracy): finetunes per session, accuracy
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_report import BLUE, ORANGE, INK, INK2, MUTED, GRID, finish  # noqa: E402,F401  (also sets rcParams)

S = "sprt_0.65_0.85_0.05_0.2"
K = {k: f"sprtk_{k}_0.65_0.85_0.05_0.2" for k in (4, 8, 12)}
CLOCK = ["never", "every_20", "every_12", "every_8", "every_6", "always"]
CLOCK_LABEL = {
    "never": "never",
    "every_20": "every:20",
    "every_12": "every:12",
    "every_8": "every:8",
    "every_6": "every:6",
    "always": "always (every:4)",
}


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def by_subject(ps):
    return ps.groupby(["condition", "subject"])[["acc", "n_ft"]].mean()


def budget(ps, out):
    s = by_subject(ps).groupby("condition").mean()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    cx, cy = s.loc[CLOCK, "n_ft"], 100 * s.loc[CLOCK, "acc"]
    ax.plot(cx, cy, color=MUTED, lw=2, marker="o", ms=7, zorder=2)
    for c in CLOCK:
        off = (-6, 8) if c == "every_6" else (4, 8)
        ax.annotate(
            CLOCK_LABEL[c],
            (s.n_ft[c], 100 * s.acc[c]),
            xytext=off,
            textcoords="offset points",
            fontsize=9,
            color=INK2,
            ha="right" if off[0] < 0 else "left",
        )
    ax.plot(s.n_ft["block"], 100 * s.acc["block"], marker="s", ms=9, color=INK2, ls="none", zorder=3)
    ax.annotate(
        "block（舊機制）",
        (s.n_ft["block"], 100 * s.acc["block"]),
        xytext=(6, -16),
        textcoords="offset points",
        fontsize=10,
        color=INK,
    )
    ax.plot(s.n_ft[S], 100 * s.acc[S], marker="o", ms=10, color=BLUE, ls="none", zorder=4, mec="white", mew=2)
    ax.annotate(
        "SPRT 預設", (s.n_ft[S], 100 * s.acc[S]), xytext=(8, -4), textcoords="offset points", fontsize=10, color=INK
    )
    for k, c in K.items():
        ax.plot(s.n_ft[c], 100 * s.acc[c], marker="D", ms=9, color=ORANGE, ls="none", zorder=4, mec="white", mew=2)
        ax.annotate(
            f"sprtk {k}",
            (s.n_ft[c], 100 * s.acc[c]),
            xytext=(-8, 9 if k != 12 else -16),
            textcoords="offset points",
            fontsize=10,
            color=INK,
            ha="right" if k == 12 else "left",
        )
    ax.set_xlabel("每個 session 的微調次數")
    ax.set_ylabel("線上正確率 (%)")
    ax.set_xlim(-2, 72)
    finish(
        fig,
        ax,
        "sprtk 比 SPRT 好一點，但仍在時鐘曲線下方",
        "灰線 = 只按固定間隔微調；橘色 = SPRT 加最少間隔 k；24 位受試者平均",
        os.path.join(out, "r6_budget.png"),
    )


def vs_block(ps, out):
    s = by_subject(ps)["acc"].unstack(0)
    rows = [
        ("SPRT 預設", S, BLUE),
        ("sprtk 4", K[4], ORANGE),
        ("sprtk 8", K[8], ORANGE),
        ("sprtk 12", K[12], ORANGE),
        ("every:12", "every_12", MUTED),
        ("every:8", "every_8", MUTED),
        ("always", "always", MUTED),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (lab, c, col) in enumerate(rows):
        d = 100 * (s[c] - s["block"]).dropna()
        lo, hi = boot(d)
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(d.mean(), y, marker="o", ms=9, color=col, mec="white", mew=2, zorder=3)
        ax.annotate(
            f"{d.mean():+.2f}", (hi, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2
        )
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("正確率 − block (pp)，點 = 平均，線 = 95% 信賴區間")
    finish(
        fig,
        ax,
        "三個 k 都顯著輸給 block",
        "每位受試者配對相減（n = 24）；0 的右邊 = 比 block 好",
        os.path.join(out, "r6_vs_block.png"),
    )


def gap(pf, out):
    conds = [
        ("SPRT 預設", S, BLUE, "o"),
        ("sprtk 4", K[4], ORANGE, "D"),
        ("sprtk 8", K[8], ORANGE, "D"),
        ("sprtk 12", K[12], ORANGE, "D"),
        ("block", "block", INK2, "s"),
        ("every:8", "every_8", MUTED, "o"),
        ("every:12", "every_12", MUTED, "o"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for lab, c, col, m in conds:
        x = pf[pf.condition == c]
        # 點 = 所有微調合併的平均；CI = 以受試者為單位重抽 (cluster bootstrap)
        sums = x.groupby("subject").effect.agg(["sum", "count"])
        idx = np.random.default_rng(0).integers(0, len(sums), (10000, len(sums)))
        bs = sums["sum"].values[idx].sum(1) / sums["count"].values[idx].sum(1) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        g, e = x.gap.mean(), 100 * x.effect.mean()
        ax.plot([g, g], [lo, hi], color=col, lw=2, alpha=0.6)
        ax.plot(g, e, marker=m, ms=9, color=col, mec="white", mew=2, ls="none", zorder=3)
        off = {"every:12": (8, -12), "sprtk 12": (8, 4), "every:8": (-8, 8), "sprtk 8": (8, -4)}.get(lab, (8, -4))
        ax.annotate(
            lab,
            (g, e),
            xytext=off,
            textcoords="offset points",
            fontsize=10,
            color=INK,
            ha="right" if off[0] < 0 else "left",
        )
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("平均每次微調距離上一次的 trial 數")
    ax.set_ylabel("一次微調的效果 (pp，之後 48 個 trial)")
    finish(
        fig,
        ax,
        "最少間隔讓每次微調更有用，但沒有比時鐘好",
        "每次微調：新模型 − 舊模型在之後 48 個 trial 的正確率，所有微調平均；線 = 95% 信賴區間（依受試者重抽）",
        os.path.join(out, "r6_gap.png"),
    )


def terciles(ps, out):
    s = by_subject(ps)
    nv = s.loc["never", "acc"]
    grp = pd.qcut(nv, 3, labels=["弱", "中", "強"])
    names = [f"{g}（{(grp == g).sum()} 位）" for g in ["弱", "中", "強"]]
    series = [("sprtk 8", K[8], ORANGE), ("every:12", "every_12", MUTED)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    ax = axes[0]  # 微調次數：長條從 0 開始
    w = 0.36
    for i, (lab, c, col) in enumerate(series):
        v = s.loc[c, "n_ft"].groupby(grp, observed=True).mean()
        xs = np.arange(3) + (i - 0.5) * w
        ax.bar(xs, v.values, width=w - 0.03, color=col, label=lab, zorder=2)
        for x_, y_ in zip(xs, v.values):
            ax.annotate(f"{y_:.1f}", (x_, y_), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
                        color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("每個 session 的微調次數")
    ax.set_ylim(0, 29)
    ax.grid(axis="x", visible=False)
    ax.set_title("微調次數", loc="left", fontsize=11, color=INK2)
    ax = axes[1]  # 正確率：啞鈴圖 (軸不從 0 開始，所以不用長條)
    acc = {lab: s.loc[c, "acc"].groupby(grp, observed=True).mean() * 100 for lab, c, _ in series}
    for t in range(3):
        a, b = acc["sprtk 8"].iloc[t], acc["every:12"].iloc[t]
        ax.plot([t, t], [a, b], color=GRID, lw=3, zorder=1)
        ax.annotate(f"{a - b:+.1f} pp", (t, (a + b) / 2), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
    for lab, c, col in series:
        ax.plot(range(3), acc[lab].values, marker="o", ms=10, color=col, ls="none", mec="white", mew=2, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel("線上正確率 (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("正確率（數字 = sprtk 8 − every:12）", loc="left", fontsize=11, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.97))
    fig.suptitle("每一組受試者，sprtk 8 都輸給 every:12", x=0.02, y=0.97, ha="left", fontsize=14, fontweight="bold",
                 color=INK)
    fig.text(0.02, 0.875, "依「完全不微調」時的正確率分三組；弱組 sprtk 8 微調較多卻較差，強組分到的微調太少",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(out, "r6_terciles.png"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(out, "r6_terciles.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="analysis_20261125 (round2plus6/, round6_sprtk_20261124/)")
    ap.add_argument("--r2", required=True, help="analysis_20261124/round2_gates_20261123 (per_finetune.csv)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ps = pd.read_csv(os.path.join(args.a, "round2plus6", "per_session.csv"))
    pf = pd.concat(
        [
            pd.read_csv(os.path.join(args.a, "round6_sprtk_20261124", "per_finetune.csv")),
            pd.read_csv(os.path.join(args.r2, "per_finetune.csv")),
        ]
    )
    pf = pf[pf.condition.isin([S, "block", "every_8", "every_12", *K.values()])]
    pf = pf.sort_values(["subject", "session", "condition", "seed", "after_trial"])
    pf["gap"] = pf.groupby(["subject", "session", "condition", "seed"]).after_trial.diff().fillna(pf.after_trial + 1)
    pf = pf[pf.n_next >= 8]
    budget(ps, args.out)
    vs_block(ps, args.out)
    gap(pf, args.out)
    terciles(ps, args.out)


if __name__ == "__main__":
    main()
