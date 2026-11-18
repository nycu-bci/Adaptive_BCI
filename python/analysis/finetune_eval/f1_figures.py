# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f1_figures.py
"""
Step 3 (figures) from results/finetune_eval/trial_checkpoint.csv.gz, finetune_results.csv and f1_pooled.csv:
  heatmap_<sub>_<model>.png  every checkpoint x blocks of 4 trials (vote accuracy), in-sample cells hatched, live chain outlined, side panel
  chain_<sub>_<model>.png    rolling accuracy of the live chain vs frozen c?_000 / c?_001, chain - frozen with a bootstrap band, child - parent per finetune
  f1_pooled.png              per-finetune W1/W2 effects and pooled estimates with CIs, split live / not live
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_common as F
from f1_analyze import SessionData
from wald_sprt_explainer import SURFACE, INK, INK2, MUTED, GRID, AXIS, MODEL_COLOR, RULE_COLOR, style, header

BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#0d366b"])
LIVE_C, NOTLIVE_C = RULE_COLOR[2], RULE_COLOR[3]
ROLL, MIN_PERIODS = 24, 12


def column_groups(sess, block=F.DISPLAY_BLOCK):
    """Trial indices of each display column: 4 consecutive trials within a run (session order)."""
    cols = []
    for r in sess.runs:
        idx = [t["seq"] for t in r.trials]
        cols += [idx[i:i + block] for i in range(0, len(idx), block)]
    return cols


def run_edges(sess):
    e, n = [], 0
    for r in sess.runs[:-1]:
        n += len(r.trials)
        e.append(n)
    return e


def ck_names(sess, m):
    return [F.ckpt_name(c["path"]) for c in sess.chain[m]]


# ---------------- figure 1: heatmap ----------------
def wrap(width_in, text):
    import textwrap
    return "\n".join(textwrap.fill(p, int(width_in * 10.2)) for p in text.split("\n"))


def fig_heatmap(sd, m, out):
    s = sd.sess
    names = ck_names(s, m)
    nck = len(names)
    cols = column_groups(s)
    ok, ins, oc = sd.ok[m], sd.insample[m], sd.on_chain[m]
    acc = np.full((nck, len(cols)), np.nan)
    frac_in = np.zeros((nck, len(cols)))
    live_cell = np.zeros((nck, len(cols)), bool)
    for j, idx in enumerate(cols):
        acc[:, j] = np.nanmean(ok[idx, :], axis=0)
        frac_in[:, j] = ins[idx, :].mean(0)
        live_cell[:, j] = oc[idx, :].any(0)
    rgba = BLUES(np.nan_to_num(acc, nan=0.0))[..., :3]
    surface = np.array(to_rgb(SURFACE))
    shade = np.where(frac_in >= 0.5, 0.45, 1.0)[..., None]
    img = surface + (rgba - surface) * shade
    img[np.isnan(acc)] = to_rgb(GRID)

    w = min(17.5, max(11.5, 4.5 + 0.13 * len(cols)))
    h = max(5.2, 2.5 + 0.23 * nck)
    fig = plt.figure(figsize=(w, h))
    left, right, top, bottom = 0.07, 0.985, 1 - 1.6 / h, 0.8 / h
    strip_h = 0.16 / h * 1.2
    main_w = 0.80
    ax = fig.add_axes([left, bottom + strip_h + 0.012, main_w * (right - left), top - bottom - strip_h - 0.012])
    ax_live = fig.add_axes([left, bottom, main_w * (right - left), strip_h], sharex=ax)
    ax_side = fig.add_axes([left + main_w * (right - left) + 0.015, bottom + strip_h + 0.012, (1 - main_w) * (right - left) - 0.015, top - bottom - strip_h - 0.012], sharey=ax)
    ax.imshow(img, extent=(0, len(cols), nck, 0), aspect="auto", interpolation="nearest")
    ax.grid(False)
    hatch = [Rectangle((j, k), 1, 1) for k in range(nck) for j in range(len(cols)) if frac_in[k, j] >= 0.5]
    if hatch:
        ax.add_collection(PatchCollection(hatch, facecolor="none", edgecolor=INK2, hatch="//////", linewidth=0, alpha=0.55))
    outline = [Rectangle((j, k), 1, 1) for k in range(nck) for j in range(len(cols)) if live_cell[k, j]]
    ax.add_collection(PatchCollection(outline, facecolor="none", edgecolor=INK, linewidth=1.5))
    edges = np.cumsum([0] + [len(c) for c in cols])
    for e in run_edges(s):
        j = int(np.searchsorted(edges, e))
        ax.axvline(j, color=INK, lw=1.2)
        ax_live.axvline(j, color=INK, lw=1.2)
    starts = [0] + [int(np.searchsorted(edges, e)) for e in run_edges(s)] + [len(cols)]
    for r, a, b in zip(s.runs, starts[:-1], starts[1:]):
        ax.text((a + b) / 2, -0.35, f"run {r.run}" + (" (test only)" if r.test_only else ""), ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_yticks(np.arange(nck) + 0.5)
    ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(axis="y", length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(nck, 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j, idx in enumerate(cols):
        n_a = sum(s.live[i] == "A" for i in idx)
        ax_live.add_patch(Rectangle((j, 0), 1, 1, facecolor=MODEL_COLOR["A" if n_a * 2 >= len(idx) else "B"], edgecolor="none"))
    ax_live.set_ylim(0, 1)
    ax_live.set_yticks([])
    ax_live.grid(False)
    ax_live.set_ylabel("live", rotation=0, ha="right", va="center", fontsize=8.5, color=INK2)
    for sp in ax_live.spines.values():
        sp.set_visible(False)
    xt = np.arange(0, len(cols) + 1, 12)
    ax_live.set_xticks(xt)
    ax_live.set_xticklabels([str(int(edges[min(x, len(cols))])) for x in xt], fontsize=8)
    ax_live.set_xlabel("trials so far in the session (columns = 4 consecutive trials; live strip: blue = A, orange = B)", fontsize=9)
    # side panel: accuracy per checkpoint on trials after it went live (out-of-sample) vs before (in-sample)
    acc_out = np.array([np.nanmean(ok[~ins[:, k], k]) if (~ins[:, k]).any() else np.nan for k in range(nck)])
    acc_in = np.array([np.nanmean(ok[ins[:, k], k]) if ins[:, k].any() else np.nan for k in range(nck)])
    y = np.arange(nck) + 0.5
    ax_side.axvline(0.5, color=AXIS, lw=0.8)
    ax_side.scatter(acc_out, y, s=26, color=BLUES(0.85), zorder=3)
    ax_side.scatter(acc_in, y, s=26, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2, zorder=3)
    ax_side.set_xlim(0, 1.04)
    ax_side.set_xticks([0, 0.5, 1])
    ax_side.set_xticklabels(["0", "0.5", "1"], fontsize=8)
    plt.setp(ax_side.get_yticklabels(), visible=False)
    ax_side.tick_params(axis="y", length=0)
    ax_side.grid(axis="y", visible=False)
    for txt, off, col, wt in (("filled: later trials", 3, BLUES(0.85), "normal"), ("open: earlier (in-sample)", 14, INK2, "normal"), ("accuracy by checkpoint", 26, INK, "semibold")):
        ax_side.annotate(txt, (0, 1), xytext=(0, off), textcoords="offset points", fontsize=8.5 if wt == "normal" else 9, color=col, fontweight=wt, va="bottom", ha="left", annotation_clip=False)
    header(fig, f"{s.sub}, model {m}: vote accuracy of every checkpoint on every block of 4 trials",
           wrap(w, "Colour = share of the 4 votes right (light = 0, dark = 1). Hatched, faded cells: the trials came before the checkpoint existed (in-sample: it or its ancestors may have trained on them).\n"
           "Black outline: the checkpoint that was live at the block. Rows: c?_000 (top) to the last checkpoint; vertical lines: run boundaries. "
           + ("c?_000 of B is a random network: its row is the chance reference." if m == "B" else "c?_000 of A is the pretrained base.")))
    fig.savefig(os.path.join(out, f"heatmap_{s.sub}_{m}.png"), dpi=140)
    plt.close(fig)


# ---------------- figure 2: chain view ----------------
def rolling(x):
    return pd.Series(x).rolling(ROLL, min_periods=MIN_PERIODS).mean().to_numpy()


def rolling_diff_band(d, rng, n_boot=300):
    """Rolling mean of paired differences d (nan where undefined) with a 95% bootstrap band over the trials of each window."""
    n = len(d)
    mean, lo, hi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        w = d[max(0, t - ROLL + 1):t + 1]
        w = w[~np.isnan(w)]
        if len(w) < MIN_PERIODS:
            continue
        bs = w[rng.integers(0, len(w), (n_boot, len(w)))].mean(1)
        mean[t], (lo[t], hi[t]) = w.mean(), np.percentile(bs, [2.5, 97.5])
    return mean, lo, hi


def fig_chain(sd, m, ft_df, out, seed=0):
    s = sd.sess
    names = ck_names(s, m)
    n = len(s.trials)
    ok, oc = sd.ok[m], sd.on_chain[m]
    chain = np.array([ok[i, np.argmax(oc[i])] for i in range(n)])
    c0 = ok[:, 0].copy()
    after = s.ats > s.chain[m][1]["time"]
    c1 = np.where(after, ok[:, 1], np.nan)
    x = np.arange(1, n + 1)
    rng = np.random.default_rng(seed)
    fts = ft_df[(ft_df.session == s.sub) & (ft_df.model == m)]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.2), sharex=False, gridspec_kw={"height_ratios": [3, 2, 2.1]})
    edges = run_edges(s)
    ax = axes[0]
    ax.plot(x, rolling(c0), color=MUTED, lw=1.6, label=f"{names[0]} frozen")
    ax.plot(x, rolling(c1), color="#6250d6", lw=1.6, label=f"{names[1]} frozen (trials after it went live)")
    ax.plot(x, rolling(chain), color=MODEL_COLOR[m], lw=2.2, label="live chain (checkpoint live at each trial)")
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"vote accuracy, last {ROLL} trials")
    for xs in fts.swap_seq:
        ax.plot([xs, xs], [0, 0.04], color=INK, lw=1.0)
    ax.text(n + 0.5, 0.02, "ticks: finetunes", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.legend(loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.07), frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95)
    ax.set_title(f"Rolling vote accuracy ({ROLL}-trial window, shown from {MIN_PERIODS} trials)", fontsize=11)
    ax = axes[1]
    d1 = np.where(after, chain - ok[:, 1], np.nan)
    d0 = chain - c0
    for d, col, lab in ((d1, "#6250d6", f"chain - {names[1]}"), (d0, MUTED, f"chain - {names[0]}")):
        if d is d0 and m == "B":
            continue
        mean, lo, hi = rolling_diff_band(d, rng)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, mean, color=col, lw=1.6, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("difference in accuracy")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95)
    ax.set_title("Live chain minus frozen checkpoint (band: 95% bootstrap over the trials of each window; windows overlap)", fontsize=11)
    ax = axes[2]
    ax.axhline(0, color=INK, lw=0.8)
    for _, f in fts.iterrows():
        if f["w2_all_n"] == 0:
            continue
        d = f["w2_all_acc_c"] - f["w2_all_acc_p"]
        col = LIVE_C if f["w2_live_frac"] > 0.5 else NOTLIVE_C
        ax.scatter(f["swap_seq"], d, s=34 if f["w2_all_n"] >= 24 else 22, color=col, alpha=0.9 if f["w2_all_n"] >= 24 else 0.45, zorder=3)
    ax.set_ylabel("child - parent accuracy")
    ax.set_title(f"One dot per finetune: child minus parent on the next 48 trials (W2), at the swap; faded = fewer than 24 trials left", fontsize=11)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=LIVE_C, label=f"{m} live during W2 (majority)"),
                       Line2D([], [], marker="o", ls="", color=NOTLIVE_C, label=f"{m} not live during W2")], loc="upper left", fontsize=8.5, ncol=2, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95)
    lo_hi = [f["w2_all_acc_c"] - f["w2_all_acc_p"] for _, f in fts.iterrows() if f["w2_all_n"] > 0]
    ax.set_ylim(min(lo_hi + [-0.1]) - 0.03, max(lo_hi + [0.1]) + 0.09)
    for a in axes:
        for e in edges:
            a.axvline(e + 0.5, color=INK, lw=0.9, ls="--")
        a.set_xlim(0, n + 1)
    axes[2].set_xlabel("trial number within the session (dashed: run boundaries)")
    for r, st in zip(s.runs, [0] + edges):
        axes[0].text(st + 1, 1.005, f"run {r.run}", fontsize=8, color=INK2, va="bottom")
    header(fig, f"{s.sub}, model {m}: does the live chain beat a frozen checkpoint?",
           f"Chain = checkpoint of {m} live at each trial, scored on that trial's own vote. Frozen = one fixed checkpoint scored on the same trials. Descriptive, "
           "one session.\nDots in panel 3: positive = the finetune helped on the next trials.")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.06, hspace=0.38)
    fig.savefig(os.path.join(out, f"chain_{s.sub}_{m}.png"), dpi=140)
    plt.close(fig)


# ---------------- figure 3: pooled F1 ----------------
def fig_pooled(ft_df, pooled_df, out):
    groups = ["A", "B", "A+B"]
    lanes = [("notlive", "not live"), ("live", "live"), ("all", "all")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 9.6), sharex="col")
    rng = np.random.default_rng(1)
    for gi, g in enumerate(groups):
        d = ft_df if g == "A+B" else ft_df[ft_df.model == g]
        for wi, w in enumerate(("w1", "w2")):
            ax = axes[gi, wi]
            ax.axvline(0, color=INK, lw=0.8)
            for li, (s, lab) in enumerate(lanes):
                n = d[f"{w}_{s}_n"].to_numpy()
                eff = (d[f"{w}_{s}_acc_c"] - d[f"{w}_{s}_acc_p"]).to_numpy()
                v = n > 0
                col = {"live": LIVE_C, "notlive": NOTLIVE_C, "all": MUTED}[s]
                ax.scatter(eff[v], li + rng.uniform(-0.28, 0.28, v.sum()), s=np.clip(np.sqrt(n[v]) * 5, 6, 50), color=col, alpha=0.35, lw=0)
                p = pooled_df[(pooled_df.model == g) & (pooled_df.window == w) & (pooled_df.subset == s)]
                if len(p) and p.iloc[0]["n"] > 0:
                    p = p.iloc[0]
                    ax.errorbar([p["delta"]], [li], xerr=[[p["delta"] - p["ci_lo"]], [p["ci_hi"] - p["delta"]]], fmt="D", color=INK, ms=7, lw=1.8, capsize=3, zorder=4)
                    ax.text(0.01, li - 0.47, f"{p['delta']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]  n={int(p['n'])} ({int(p['n_ft'])} ft)" +
                            (f"  sign p={p['p']:.3f}" if w == "w1" else ""), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.2, color=INK)
                else:
                    ax.text(0.01, li - 0.47, "no trials", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.2, color=MUTED)
            ax.set_yticks(range(3))
            ax.set_yticklabels([lab for _, lab in lanes], fontsize=9)
            ax.set_ylim(2.5, -0.75)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(-0.62, 0.62)
            ax.set_title(f"{g}: {'W1 (served until the next swap)' if w == 'w1' else 'W2 (next 48 trials)'}", fontsize=10.5)
    for ax in axes[-1]:
        ax.set_xlabel("child - parent accuracy")
    header(fig, "F1: does one finetune help on the trials after it? Per finetune (dots) and pooled (diamonds, 95% CI)",
           "Dot size ~ sqrt(trials in the window); dots beyond +-0.62 not shown. Pooled = trial-weighted; CI = bootstrap over sessions then finetunes (few sessions: wide and unstable). "
           "'live' = the finetuned model\nwas the live model on the test trials. W2 windows overlap between finetunes: their CIs are too narrow. Sign test (W1) on trials where parent and child votes differ. Sessions: " + ", ".join(sorted(set(ft_df.session))) + ".")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.06, hspace=0.30, wspace=0.12)
    fig.savefig(os.path.join(out, "f1_pooled.png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(F.OUT_DIR, "trial_checkpoint.csv.gz"))
    ap.add_argument("--dir", default=F.OUT_DIR, help="folder with finetune_results.csv and f1_pooled.csv; figures go to <dir>/figures")
    ap.add_argument("--only", default="", help="comma list of sessions")
    ap.add_argument("--suffix", default="", help="'_all' for the sensitivity run (inputs finetune_results_all.csv, f1_pooled_all.csv; figures_all/)")
    args = ap.parse_args(argv)
    out = os.path.join(args.dir, "figures" + args.suffix)
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(args.table)
    ft_df = pd.read_csv(os.path.join(args.dir, f"finetune_results{args.suffix}.csv"))
    pooled_df = pd.read_csv(os.path.join(args.dir, f"f1_pooled{args.suffix}.csv"))
    style()
    subs = [s for s in F.SESSIONS if s in set(df.session) and (not args.only or s in args.only.split(","))]
    for sub in subs:
        sd = SessionData(sub, df)
        for m in "AB":
            if len(sd.sess.chain[m]) < 2:
                print(f"{sub} {m}: only the base checkpoint, skipped")
                continue
            fig_heatmap(sd, m, out)
            fig_chain(sd, m, ft_df, out)
            print(f"{sub} {m}: figures written", flush=True)
    fig_pooled(ft_df, pooled_df, out)
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
