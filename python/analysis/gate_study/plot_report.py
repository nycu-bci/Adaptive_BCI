# python python/analysis/gate_study/plot_report.py --out <dir> [--design <dir>] [--premise <benefit_features.csv>]
#        [--eval <evaluate_gates.py out dir>]
"""
Static PNG figures for the gate-study report (Notion). Palette: validated default categorical slots
(blue #2a78d6, orange #eb6834, aqua #1baf7a), neutral gray for references / controls, direct labels on every series.
  firing_curves.png  finetunes per 40-trial run vs model accuracy (exact, sprt_design_space.py)
  premise.png        mean effect of a finetune by run accuracy so far (benefit_predictors.py)
  budget.png         online accuracy vs finetunes per session, every condition (evaluate_gates.py conditions.csv)
  residual.png       SPRT configurations: accuracy minus the clock curve at the same budget, mean and 95% CI
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8f8e89", "#e6e5e0", "#fcfcfb"
plt.rcParams.update(
    {
        "font.family": ["PingFang TC", "Heiti TC", "Arial Unicode MS", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def finish(fig, ax, title, subtitle, path):
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=24, fontweight="bold")
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", fontsize=10,
                color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def firing_curves(design, out):
    rows = list(csv.DictReader(open(os.path.join(design, "curves.csv"))))

    def series(g):
        pts = sorted((float(r["p"]), float(r["fires_per_run"])) for r in rows if r["gate"] == g)
        return np.array(pts).T

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_axisbelow(True)
    ax.axvspan(0.55, 0.70, color=GRID, alpha=0.6, lw=0)
    ax.text(0.625, 9.75, "第一輪模擬的正確率範圍", ha="center", va="top", fontsize=9, color=INK2)
    x, y = series("every 4")
    ax.plot(x, y, color=MUTED, lw=2, ls="--", label="Always（每個 block）")
    ax.text(0.952, 10, "Always", color=INK2, va="center", fontsize=10)
    for g, c, lab, short, xy in [
        ("block 0.75", ORANGE, "Block gate（舊，門檻 0.75）", "Block 0.75", (0.705, 5.3)),
        ("sprt 0.65,0.85,0.05,0.2", BLUE, "SPRT 預設 0.65 / 0.85 / 0.05 / 0.2", "SPRT 預設", (0.50, 3.6)),
    ]:
        x, y = series(g)
        ax.plot(x, y, color=c, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        ax.text(*xy, short, color=INK, fontsize=10, fontweight="bold")
    ax.text(0.41, 0.5, "p = 0.65 時：Block 每 run 6.5 次、SPRT 2.7 次", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 0.9), labelcolor=INK2)
    ax.set_xlim(0.4, 1.0)
    ax.set_xticks(np.arange(0.4, 0.96, 0.05))
    ax.set_ylim(0, 11)
    ax.set_xlabel("模型的 trial 正確率 p（固定、每個 trial 獨立）")
    ax.set_ylabel("每個 40-trial run 平均微調次數")
    finish(
        fig,
        ax,
        "兩個 gate 的門檻都在 0.75 附近，差別主要是次數",
        "精確計算（動態規劃），不是抽樣；SPRT 分數在 run 開始、微調後、判好後歸零",
        os.path.join(out, "firing_curves.png"),
    )


def premise(path_csv, out):
    rows = list(csv.DictReader(open(path_csv)))
    acc = np.array([float(r["acc_run"]) for r in rows])
    eff = 100 * np.array([float(r["effect"]) for r in rows])
    bins = [
        (0, 0.5, "<0.50"),
        (0.5, 0.6, "0.50–0.60"),
        (0.6, 0.7, "0.60–0.70"),
        (0.7, 0.75, "0.70–0.75"),
        (0.75, 0.8, "0.75–0.80"),
        (0.8, 0.9, "0.80–0.90"),
        (0.9, 1.01, "≥0.90"),
    ]
    m, lo, hi, n = [], [], [], []
    rng = np.random.default_rng(0)
    for a, b, _ in bins:
        e = eff[(acc >= a) & (acc < b)]
        m.append(e.mean())
        bs = e[rng.integers(0, len(e), (2000, len(e)))].mean(1)
        lo.append(np.percentile(bs, 2.5))
        hi.append(np.percentile(bs, 97.5))
        n.append(len(e))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_axisbelow(True)
    xs = np.arange(len(bins))
    ax.bar(xs, m, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(xs, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=0)
    for x, v, k, h in zip(xs, m, n, hi):
        ax.text(x, h + 0.06, f"{v:+.2f}", ha="center", fontsize=9.5, color=INK)
        ax.text(x, -0.38, f"n={k:,}", ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(xs, [b[2] for b in bins])
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlabel("微調當下，這個 run 到目前為止的正確率（Block gate 看的值）")
    ax.set_ylabel("一次微調的效果（pp）")
    finish(
        fig,
        ax,
        "微調的效果幾乎不隨模型表現改變",
        "第一輪 Always：15,730 次微調；效果 = 新模型 − 舊模型在接下來 48 個 trial 的正確率；線段 = 95% CI",
        os.path.join(out, "premise.png"),
    )


def budget(eval_dir, out):
    rows = list(csv.DictReader(open(os.path.join(eval_dir, "conditions.csv"))))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_axisbelow(True)
    get = lambda f: sorted([r for r in rows if r["family"] == f], key=lambda r: float(r["ft_per_session"]))
    clock = get("periodic")
    ax.plot(
        [float(r["ft_per_session"]) for r in clock],
        [100 * float(r["acc"]) for r in clock],
        color=MUTED,
        lw=2,
        marker="s",
        ms=6,
        markeredgecolor=SURFACE,
        label="時鐘（Never、every:k、Always）",
    )
    for r in clock:
        ax.annotate(
            r["condition"].replace("every_", "every:"),
            (float(r["ft_per_session"]), 100 * float(r["acc"])),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8.5,
            color=INK2,
        )
    rnd = get("random")
    if rnd:
        ax.scatter(
            [float(r["ft_per_session"]) for r in rnd],
            [100 * float(r["acc"]) for r in rnd],
            s=40,
            facecolors="none",
            edgecolors=MUTED,
            linewidths=1.5,
            label="隨機（random:q）",
            zorder=3,
        )
    for fam, c, lab in [
        ("block", ORANGE, "Block gate（門檻 0.65 / 0.75 / 0.85）"),
        ("sprt", BLUE, "SPRT（21 組參數）"),
    ]:
        rs = get(fam)
        ax.scatter(
            [float(r["ft_per_session"]) for r in rs],
            [100 * float(r["acc"]) for r in rs],
            s=46,
            color=c,
            edgecolors=SURFACE,
            linewidths=1.5,
            label=lab,
            zorder=4,
        )
    for r in rows:
        if r["condition"] in ("block", "sprt_0.65_0.85_0.05_0.2", "sprt_0.5_0.75_0.05_0.2"):
            name = {
                "block": "Block（舊）",
                "sprt_0.65_0.85_0.05_0.2": "SPRT 預設",
                "sprt_0.5_0.75_0.05_0.2": "SPRT 第一版",
            }[r["condition"]]
            ax.annotate(
                name,
                (float(r["ft_per_session"]), 100 * float(r["acc"])),
                xytext={"block": (-8, 16), "sprt_0.65_0.85_0.05_0.2": (6, 6), "sprt_0.5_0.75_0.05_0.2": (-30, -22)}[
                    r["condition"]
                ],
                textcoords="offset points",
                fontsize=9.5,
                color=INK,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
            )
    orc = [r for r in rows if r["family"] == "oracle"]
    if orc:
        r = orc[0]
        ax.scatter(
            [float(r["ft_per_session"])],
            [100 * float(r["acc"])],
            marker="*",
            s=160,
            color=AQUA,
            edgecolors=INK2,
            linewidths=0.8,
            zorder=5,
            label="Oracle（偷看未來，上限）",
        )
        ax.annotate(
            f"Oracle（訓練 {float(r['trainings_per_session']):.0f} 次、採用 {float(r['ft_per_session']):.0f} 次）",
            (float(r["ft_per_session"]), 100 * float(r["acc"])),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=9.5,
            color=INK,
        )
    ax.set_xlabel("每個 session 的微調次數（採用的）")
    ax.set_ylabel("線上正確率（%，24 位受試者平均）")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right", labelcolor=INK2)
    finish(
        fig,
        ax,
        "正確率主要由微調次數決定",
        "每個點 = 一個條件；灰線 = 只按時鐘微調的對照",
        os.path.join(out, "budget.png"),
    )


def residual(eval_dir, out):
    """SPRT configs: mean residual vs clock curve per subject, with bootstrap CI (needs subject_conditions.csv)"""
    rows = list(csv.DictReader(open(os.path.join(eval_dir, "subject_conditions.csv"))))
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["subject"]] = (float(r["ft_per_session"]), float(r["acc"]))
    clock = [c for c in by if c in ("never", "always") or c.startswith("every_")]
    subjects = sorted(by["never"])
    res = {}
    for c in by:
        if not (c.startswith("sprt_") or c.startswith("block") or c.startswith("random_")):
            continue
        rr = []
        for s in subjects:
            pts = sorted(by[k][s] for k in clock if s in by[k])
            x, y = np.array(pts).T
            if s in by[c]:
                rr.append(by[c][s][1] - np.interp(by[c][s][0], x, y))
        res[c] = 100 * np.array(rr)
    order = sorted(res, key=lambda c: res[c].mean())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(order) + 1.6))
    for i, c in enumerate(order):
        r = res[c]
        bs = r[rng.integers(0, len(r), (4000, len(r)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        col = BLUE if c.startswith("sprt_") else (ORANGE if c.startswith("block") else MUTED)
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(r.mean(), i, "o", color=col, ms=6, markeredgecolor=SURFACE)
    ax.axvline(0, color=INK2, lw=1)
    lab = (
        lambda c: c.replace("sprt_", "SPRT ")
        .replace("_", " / ", 0)
        .replace("_", "/")
        .replace("block/", "Block ")
        .replace("random/", "random ")
    )
    ax.set_yticks(range(len(order)), [lab(c) if c != "block" else "Block（舊）" for c in order], fontsize=8.5)
    ax.set_xlabel("正確率 − 同樣微調次數的時鐘（pp）；> 0 = 時機挑得比時鐘好")
    ax.grid(axis="y", visible=False)
    finish(
        fig,
        ax,
        "gate 挑的時機有沒有比時鐘好？",
        "藍 = SPRT、橘 = Block、灰 = 隨機；點 = 24 位受試者平均，線 = 95% CI",
        os.path.join(out, "residual.png"),
    )


def gap_effect(per_finetune_csv, out, cond="sprt_0.65_0.85_0.05_0.2"):
    """mean effect of a finetune by trials since the previous finetune of the same simulation"""
    import collections

    rows = list(csv.DictReader(open(per_finetune_csv)))
    by = collections.defaultdict(list)
    for r in rows:
        if r["condition"] == cond:
            by[(r["subject"], r["session"], r["seed"])].append((int(r["after_trial"]), 100 * float(r["effect"])))
    gaps, eff = [], []
    for v in by.values():
        v.sort()
        for (t0, _), (t1, e) in zip(v, v[1:]):
            gaps.append(t1 - t0)
            eff.append(e)
    gaps, eff = np.array(gaps), np.array(eff)
    bins = [(1, 2, "1–2"), (3, 4, "3–4"), (5, 8, "5–8"), (9, 16, "9–16"), (17, 40, "17–40"), (41, 10**6, "≥41")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_axisbelow(True)
    rng = np.random.default_rng(0)
    for i, (a, b, lab) in enumerate(bins):
        e = eff[(gaps >= a) & (gaps <= b)]
        bs = e[rng.integers(0, len(e), (2000, len(e)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.bar(i, e.mean(), width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.plot([i, i], [lo, hi], color=INK2, lw=1.2)
        ax.text(i, max(hi, 0) + 0.12, f"{e.mean():+.2f}", ha="center", fontsize=9.5, color=INK)
        ax.text(
            i,
            -1.1,
            f"n={len(e):,}\n佔 {100 * len(e) / len(eff):.0f}%",
            ha="center",
            fontsize=8.5,
            color=INK2,
            va="top",
        )
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(bins)), [b[2] for b in bins])
    ax.set_ylim(-2.4, 6.4)
    ax.set_xlabel("距離上一次微調幾個 trial（= buffer 裡多了幾個新 trial）")
    ax.set_ylabel("一次微調的效果（pp）")
    finish(
        fig,
        ax,
        "緊接著上一次的微調幾乎沒用：SPRT 有 33% 是這種",
        "SPRT 預設，第二輪 5 seeds × 48 session；不含 session 第一次微調；線段 = 95% CI（以微調為單位）",
        os.path.join(out, "gap_effect.png"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--design")
    ap.add_argument("--premise")
    ap.add_argument("--eval")
    ap.add_argument("--per_finetune")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.design:
        firing_curves(args.design, args.out)
    if args.premise:
        premise(args.premise, args.out)
    if args.eval:
        budget(args.eval, args.out)
        residual(args.eval, args.out)
    if args.per_finetune:
        gap_effect(args.per_finetune, args.out)


if __name__ == "__main__":
    main()
