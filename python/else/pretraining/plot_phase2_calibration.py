"""
Phase 2 校正模擬的視覺化：部署模型 (預訓練) vs 隨機起點
=====================================================

上圖：全部 22 位受試者，每人用「沒看過他」的 fold 模型當起點 (等同部署配方的估計)，平均 ± 1 SE
下排：部署模型 (fold1) 真正 held out 的 5 位受試者，逐人畫

x 軸：s1 真實 run 1..7，接著 s2 真實 run 1..7。每個點是該 run 40 個 trial 的模型真實判斷正確率
      (每個 trial 用當下的模型判斷，校正在每 4 個 trial 之後才發生)，多個 seed 取平均；只有 6 個 run 的 session 缺真實 run 3
校正依 metrics_summary.csv 的 A/B 條件：condition A 只在真實 run 1-3 校正，condition B 七個 run 都校正；
      s2 接續 s1 校正完的模型 (replay buffer 每個 session 清空)
顏色 = session (s1 藍、s2 橘)；線型 = 起點模型 (實線 = 部署模型、虛線 = 隨機)
另存 *_by_condition.png：依 condition A / B 分兩張圖

用法：python python/else/pretraining/plot_phase2_calibration.py [--result calibration_ab.json] [--out deploy_vs_random_ab.png]
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Aden")
EXP_DIR = os.path.join(_script_dir, "pretrained_result", "experiments", "phase2_calibration")
DEPLOY_FOLD = "fold1.pth"
N_RUNS = 7

# dataviz 預設類別色 slot 1/2 (已驗證 CVD 與對比)。顏色表示 session，線型表示起點模型，
# 兩個維度各用一種編碼，彼此不重疊
C_S1, C_S2 = "#2a78d6", "#eb6834"
C_SURFACE, C_INK, C_INK2, C_GRID, C_REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0", "#8a8984"
LS = {"pretrained": "-", "random": (0, (5, 2))}
LABEL = {"pretrained": "Deployment model start", "random": "Random start (current system)"}


def series_one(r):
    """一個 (arm, subject) 的 14 個點：s1 run1..7 + s2 run1..7，模型真正的最後一刀正確率 (%)。缺的 run 補 nan。"""
    out = np.full(2 * N_RUNS, np.nan)
    for u in r["runs"]:
        k = u.get("real_run") or int(u["run"][3:])          # 以真實 run 編號放位置，缺的 run 留空
        out[(0 if u["session"] == "s1" else N_RUNS) + k - 1] = u["true_acc"] * 100
    return out


def series(recs):
    """同一位受試者、同一個起點的所有 seed 取平均。"""
    recs = recs if isinstance(recs, list) else [recs]
    return np.nanmean(np.vstack([series_one(r) for r in recs]), 0)


def style_ax(ax):
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_INK2, labelsize=9, length=0)
    ax.axhline(50, color=C_REF, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(N_RUNS - 0.5, color=C_GRID, linewidth=1.2, zorder=1)   # s1 | s2 分隔


def draw(ax, v, arm, lw, band=None, markers=True):
    """把 14 個點畫成兩段：s1 段藍色、s2 段橘色，同一個起點模型用同一種線型。"""
    x = np.arange(2 * N_RUNS)
    for seg, colour in ((slice(0, N_RUNS), C_S1), (slice(N_RUNS, 2 * N_RUNS), C_S2)):
        if band is not None:
            ax.fill_between(x[seg], (v - band)[seg], (v + band)[seg], color=colour, alpha=0.10, linewidth=0)
        ax.plot(x[seg], v[seg], color=colour, linestyle=LS[arm], linewidth=lw, zorder=3)
        if markers:
            ax.plot(x[seg], v[seg], "o" if arm == "pretrained" else "s", color=colour, markersize=5,
                    markerfacecolor=colour if arm == "pretrained" else C_SURFACE,
                    markeredgewidth=1.4, zorder=4)


def plot_by_condition(by, subjects, args_, n_seeds, out_png):
    """每位受試者各有一個 condition A session 與一個 B session；依條件分兩張圖，x = 真實 run 1-7。"""
    arms = [("pretrained", C_S1, "-", "o", "Deployment model start, calibrated"),
            ("random", C_S2, (0, (5, 2)), "s", f"Random start, calibrated (mean of {n_seeds} seeds)"),
            ("static", C_REF, (0, (1, 1.5)), "", "Deployment model, never calibrated")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    fig.patch.set_facecolor(C_SURFACE)
    x = np.arange(1, N_RUNS + 1)
    for ax, cond in zip(axes, ("A", "B")):
        style_ax(ax)
        ax.lines[-1].remove()                              # style_ax 的 s1|s2 分隔線這裡用不到
        if cond == "A":
            ax.axvspan(3.5, 7.5, color=C_GRID, alpha=0.45, zorder=0, linewidth=0)
            ax.text(5.5, 0.97, "runs 4–7: no calibration", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color=C_INK2)
        for arm, colour, ls, mk, label in arms:
            if not all(arm in by[s] for s in subjects):
                continue
            rows = []
            for s in subjects:
                v = series(by[s][arm])
                sess = [u["session"] for u in by[s][arm][0]["runs"] if u.get("condition") == cond]
                if not sess:
                    continue
                off = 0 if sess[0] == "s1" else N_RUNS
                rows.append(v[off:off + N_RUNS])
            M = np.vstack(rows)
            m = np.nanmean(M, 0)
            se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), 0))
            if arm != "static":
                ax.fill_between(x, m - se, m + se, color=colour, alpha=0.12, linewidth=0)
            ax.plot(x, m, color=colour, linestyle=ls, linewidth=2.2, marker=mk or None, markersize=5,
                    markerfacecolor=colour if mk == "o" else C_SURFACE, markeredgewidth=1.4, label=label, zorder=3)
        ax.set_xticks(x)
        ax.set_xlim(0.6, 7.4)
        ax.set_xlabel("Run (real run number)", fontsize=10, color=C_INK2)
        n = len(rows)
        ax.set_title(f"Condition {cond} sessions · {n} subjects · "
                     + ("calibration in runs 1–3 only" if cond == "A" else "calibration in every run"),
                     loc="left", fontsize=10.5, color=C_INK)
    axes[0].set_ylabel("Trial accuracy (%)", fontsize=10, color=C_INK2)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.995), frameon=False, fontsize=9,
               labelcolor=C_INK2, ncol=1)
    fig.text(0.012, 0.985, "Deployment model vs random start, split by condition", fontsize=13.5,
             fontweight="bold", color=C_INK, va="top")
    fig.text(0.012, 0.935, f"Mean ± 1 SE over subjects. Decision: {args_.get('decision')}; gate: {args_.get('gate')}. "
             "Each subject has one session of each condition.", fontsize=9.3, color=C_INK2, va="top")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.12, wspace=0.06)
    fig.savefig(out_png, dpi=170, facecolor=C_SURFACE)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=os.path.join(EXP_DIR, "calibration_ab.json"))
    ap.add_argument("--out", default=os.path.join(EXP_DIR, "deploy_vs_random_ab.png"))
    args = ap.parse_args()

    data = json.load(open(args.result))
    sham_on = data["game"]["sham"]
    res = data["results"]
    by = {}
    for r in res:
        by.setdefault(r["subject"], {}).setdefault(r["arm"], []).append(r)
    subjects = sorted(by, key=int)
    deploy_subjects = [s for s in subjects if by[s]["pretrained"][0]["fold"] == DEPLOY_FOLD]
    n_seeds = len({r.get("seed") for r in res if r["arm"] == "random"})
    args_ = data["args"]
    x = np.arange(2 * N_RUNS)
    xt = [str(k) for k in range(1, N_RUNS + 1)] * 2

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(C_SURFACE)
    gs = fig.add_gridspec(2, len(deploy_subjects), height_ratios=[1.35, 1], hspace=0.5, wspace=0.12)

    # --- 上圖：22 位平均 ± 1 SE ---
    ax = fig.add_subplot(gs[0, :])
    style_ax(ax)
    means = {}
    for arm in ("random", "pretrained"):
        M = np.vstack([series(by[s][arm]) for s in subjects])
        m = np.nanmean(M, 0)
        se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), 0))
        draw(ax, m, arm, 2.2, band=se)
        means[arm] = m
    for k in (0, N_RUNS - 1, N_RUNS):   # 只標第一個 run、s1 最後一個 run、s2 第一個 run
        for arm, dy in (("pretrained", 9), ("random", -15)):
            ax.annotate(f"{means[arm][k]:.1f}", (x[k], means[arm][k]), xytext=(0, dy),
                        textcoords="offset points", ha="center", fontsize=8.5, color=C_INK, fontweight="bold")
    ax.set_xticks(x, xt)
    if not args_.get("no_ab_rule"):
        ax.text(0.99, 1.02, "Each session is condition A for 11 subjects (calibrate runs 1–3 only) and B for 11 (all runs)",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=C_INK2)
    ax.set_xlim(-0.5, 2 * N_RUNS - 0.5)
    lo = np.nanmin([means[a] for a in means]) - 4
    hi = np.nanmax([means[a] for a in means]) + 4
    ax.set_ylim(min(lo, 47), max(hi, 65))
    ax.set_ylabel("Trial accuracy (%)", fontsize=10, color=C_INK2)
    ax.text(0.0, -0.13, "Session 1 · runs 1–7", transform=ax.transAxes, fontsize=9.5,
            color=C_S1, fontweight="bold")
    ax.text(0.5, -0.13, "Session 2 (next day) · continues from the session-1 model",
            transform=ax.transAxes, fontsize=9.5, color=C_S2, fontweight="bold")
    ax.set_title(f"All {len(subjects)} subjects · each tested by the fold model that never saw them · mean ± 1 SE",
                 loc="left", fontsize=10.5, color=C_INK2, pad=8)
    handles = [plt.Line2D([], [], color=C_INK2, linestyle=LS["pretrained"], marker="o", markersize=5,
                          linewidth=2, label=LABEL["pretrained"]),
               plt.Line2D([], [], color=C_INK2, linestyle=LS["random"], marker="s", markersize=5,
                          markerfacecolor=C_SURFACE, linewidth=2, label=LABEL["random"]),
               plt.Line2D([], [], color=C_REF, linewidth=1, linestyle=(0, (4, 3)), label="chance")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=C_INK2, ncol=3)

    # --- 下排：部署模型 (fold1) 真正沒看過的 5 位 ---
    for i, s in enumerate(deploy_subjects):
        a = fig.add_subplot(gs[1, i])
        style_ax(a)
        for arm in ("random", "pretrained"):
            draw(a, series(by[s][arm]), arm, 1.6, markers=False)
        a.set_ylim(35, 100)
        a.set_xlim(-0.5, 2 * N_RUNS - 0.5)
        a.set_xticks([0, 6, 7, 13], ["1", "7", "1", "7"])
        if i:
            a.tick_params(labelleft=False)
        else:
            a.set_ylabel("Trial accuracy (%)", fontsize=10, color=C_INK2)
        pre, rnd = series(by[s]["pretrained"]), series(by[s]["random"])
        a.set_title(f"Subject {s}", loc="left", fontsize=10, color=C_INK, fontweight="bold", pad=4)
        a.text(0.02, 0.97, f"s1 {np.nanmean(pre[:N_RUNS]):.1f} vs {np.nanmean(rnd[:N_RUNS]):.1f}\n"
                           f"s2 {np.nanmean(pre[N_RUNS:]):.1f} vs {np.nanmean(rnd[N_RUNS:]):.1f}",
               transform=a.transAxes, fontsize=8.3, color=C_INK2, va="top")
    fig.text(0.012, 0.43, "Deployment model (fold 1) · the 5 subjects it never saw · "
             "numbers: session mean, deployment vs random", fontsize=10.5, color=C_INK2)

    fig.text(0.012, 0.985, "Calibration from the deployment model vs from random weights",
             fontsize=13.5, fontweight="bold", color=C_INK, va="top")
    dec_txt = ("majority vote over the trial's 16 windows" if args_.get("decision") == "trial_mode"
               else "weighted vote at the brain-controlled last cut")
    fig.text(0.012, 0.953, f"Model's own trial decision ({dec_txt}), per run of 40 trials (sham help "
             f"excluded). Decision: {args_.get('decision', 'last_cut')}; gate: {args_.get('gate', 'live')}; "
             f"A/B rule {'off' if args_.get('no_ab_rule') else 'on'}; mean of {n_seeds} seeds per subject.",
             fontsize=9.3, color=C_INK2, va="top")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.06)
    fig.savefig(args.out, dpi=170, facecolor=C_SURFACE)
    print(f"saved {args.out}")
    if not args_.get("no_ab_rule"):
        plot_by_condition(by, subjects, args_, n_seeds, args.out.replace(".png", "_by_condition.png"))


if __name__ == "__main__":
    main()
