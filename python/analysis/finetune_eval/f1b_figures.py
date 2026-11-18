# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f1b_figures.py
"""
Figures for step 0b and F1b (inputs: f1b/step0b/step0b_per_finetune.csv, f1b/f1b_per_finetune.csv, f1b/f1b_pooled.csv; outputs in f1b/figures/):
  step0b_agreement.png   per finetune: vote agreement retrained vs saved child and seed vs seed (W2 top, W1 bottom)
  f1b_pooled_w2.png, f1b_pooled_w1.png   per variant pooled delta vs parent (diamonds, 95% CI over finetunes) and per-finetune deltas (dots), like f1_pooled.png
  f1b_early_A.png        A's early finetunes (run 1): accuracy per finetune on W2 and on the late run for parent, saved, retrain, V0, V1, V2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_common as R
from wald_sprt_explainer import SURFACE, INK, INK2, MUTED, GRID, AXIS, MODEL_COLOR, RULE_COLOR, style, header
F = R.F

OUT = os.path.join(F.OUT_DIR, "f1b")
VCOL = {"saved": INK2, "asran": MUTED, "v0": RULE_COLOR[0], "v1": RULE_COLOR[1], "v2": RULE_COLOR[2]}
VLAB = {"saved": "saved child (as ran)", "asran": "as-ran retrain", "v0": "V0 corrected", "v1": "V1 held-out + keep parent", "v2": "V2 validate on next block"}
LIVE_C, NOTLIVE_C = RULE_COLOR[2], RULE_COLOR[3]


def fig_step0b(d, out):
    panels = [(s, m) for s in F.SESSIONS for m in "AB" if len(d[(d.session == s) & (d.model == m)])]
    widths = [max(4.0, 0.16 * d[(d.session == s) & (d.model == m) & (d.window == "w2")].shape[0] + 1.5) for s, m in panels]
    fig, axes = plt.subplots(2, len(panels), figsize=(sum(widths) + 1.4, 7.6), gridspec_kw={"width_ratios": widths}, squeeze=False)
    for ci, (s, m) in enumerate(panels):
        for ri, w in enumerate(("w2", "w1")):
            ax = axes[ri, ci]
            g = d[(d.session == s) & (d.model == m) & (d.window == w)].sort_values("k")
            x = np.arange(len(g))
            ax.vlines(x, g.agree_seeds, g.agree_saved, color=GRID, lw=1.2, zorder=1)
            ax.scatter(x, g.agree_seeds, marker="D", s=22, facecolor=SURFACE, edgecolor=INK2, lw=1.1, zorder=3, label="seed vs seed (mean of 10 pairs)")
            ax.scatter(x, g.agree_saved, marker="o", s=26, color=MODEL_COLOR[m], zorder=4, label="retrained vs saved child (mean of 5 seeds)")
            ax.axhline(g.agree_seeds.median(), color=INK2, lw=0.9, ls="--", zorder=2)
            ax.axhline(g.agree_saved.median(), color=MODEL_COLOR[m], lw=0.9, zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels(g.k.astype(int), fontsize=7)
            ax.set_ylim(0.45, 1.02)
            ax.set_xlim(-0.8, len(g) - 0.2)
            ax.set_title(f"{s} {m}: {w.upper()}  (median {g.agree_saved.median():.3f} vs {g.agree_seeds.median():.3f}, n={len(g)})", fontsize=9.6)
            if ci == 0:
                ax.set_ylabel("vote agreement")
            if ri == 1:
                ax.set_xlabel("finetune k (chain index)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8.5, bbox_to_anchor=(0.99, 0.935))
    header(fig, "Step 0b: does retraining from the saved parent on the original data reproduce the saved finetune?",
           "Per finetune, share of trials where two checkpoints give the same offline Unity vote. Filled dot = retrained seeds vs the saved child; open diamond = the retrained seeds against each other\n"
           "(noise floor). Lines = medians. Gate: median(filled) >= median(open) - 0.02 on W2. W1 windows can be very short (2-4 trials), so single W1 points are coarse.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.08, hspace=0.32, wspace=0.12)
    fig.savefig(os.path.join(out, "step0b_agreement.png"), dpi=140)
    plt.close(fig)


def fig_pooled(df, pooled, w, out):
    variants = [v for v in ("saved", "v0", "v1", "v2") if v in set(df.variant)]
    groups = ["A", "B", "A+B"]
    lanes = [("notlive", "not live"), ("live", "live"), ("all", "all")]
    fig, axes = plt.subplots(3, len(variants), figsize=(4.1 * len(variants) + 0.9, 9.4), sharex=True, squeeze=False)
    rng = np.random.default_rng(1)
    for gi, g in enumerate(groups):
        for vi, v in enumerate(variants):
            ax = axes[gi, vi]
            d = df[(df.variant == v)] if g == "A+B" else df[(df.variant == v) & (df.model == g)]
            ax.axvline(0, color=INK, lw=0.8)
            for li, (s, lab) in enumerate(lanes):
                n = d[f"{w}_{s}_n"].to_numpy()
                eff = (d[f"{w}_{s}_acc_c"] - d[f"{w}_{s}_acc_p"]).to_numpy()
                ok = n > 0
                col = {"live": LIVE_C, "notlive": NOTLIVE_C, "all": MUTED}[s]
                ax.scatter(eff[ok], li * 1.3 + rng.uniform(-0.28, 0.28, ok.sum()), s=np.clip(np.sqrt(n[ok]) * 4, 5, 40), color=col, alpha=0.35, lw=0)
                p = pooled[(pooled.model == g) & (pooled.variant == v) & (pooled.window == w) & (pooled.subset == s)]
                if len(p) and p.iloc[0]["n"] > 0:
                    p = p.iloc[0]
                    ax.errorbar([p["delta"]], [li * 1.3], xerr=[[max(p["delta"] - p["ci_lo"], 0)], [max(p["ci_hi"] - p["delta"], 0)]], fmt="D", color=INK, ms=6, lw=1.6, capsize=3, zorder=4)
                    ax.text(0.015, li * 1.3 + 0.56, f"{p['delta']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] n={int(p['n'])}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                            fontsize=7.4, color=INK)
                else:
                    ax.text(0.015, li * 1.3 + 0.56, "no trials", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.4, color=MUTED)
            ax.set_yticks([0, 1.3, 2.6])
            ax.set_yticklabels([lab for _, lab in lanes] if vi == 0 else [], fontsize=8.5)
            ax.set_ylim(3.35, -0.6)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(-0.3, 0.3)
            ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
            if gi == 0:
                ax.set_title(VLAB[v], fontsize=10, color=VCOL[v])
            if vi == 0:
                ax.set_ylabel(g, fontsize=11, rotation=0, labelpad=26, va="center")
    for ax in axes[-1]:
        ax.set_xlabel("child' - parent accuracy")
    wt = {"w1": "W1 (served until the next swap)", "w2": "W2 (next 48 trials after the swap)"}[w]
    header(fig, f"F1b, {wt}: does a finetune help? Pooled delta per variant (diamond, 95% CI over finetunes) and per-finetune deltas (dots)",
           "Children retrained on the H4-correct cut, 5 seeds averaged per finetune; 'saved child' = the checkpoint the online system made (F1). Dot size ~ sqrt(trials); dots beyond +-0.3 not shown.\n"
           "Pooled = trial-weighted. 'live' = the finetuned model was the live model on the test trials. Few sessions: CIs describe finetunes, not subjects; W2 windows overlap, so W2 CIs are too narrow.")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.865, bottom=0.06, hspace=0.20, wspace=0.07)
    fig.savefig(os.path.join(out, f"f1b_pooled_{w}.png"), dpi=140)
    plt.close(fig)


def fig_early(early, df, out, model="A"):
    sessions = [s for s in F.SESSIONS if len(early[(early.session == s) & (early.model == model)])]
    if not sessions:
        return
    fig, axes = plt.subplots(len(sessions), 3, figsize=(15.2, 3.7 * len(sessions) + 2.4), squeeze=False)
    for ri, s in enumerate(sessions):
        e = early[(early.session == s) & (early.model == model)]
        ks = sorted(e.k.unique())
        for ci, (col, title) in enumerate((("w2", "W2: next 48 trials after the swap"), ("late", "late run (common test set)"))):
            ax = axes[ri, ci]
            if e[f"{col}_n"].max() == 0:
                ax.text(0.5, 0.5, "no late run in this session", ha="center", va="center", transform=ax.transAxes, color=MUTED)
                ax.set_axis_off()
                continue
            par = e[e.variant == e.variant.iloc[0]].set_index("k")[f"{col}_parent"].reindex(ks)
            ax.plot(ks, par.values, color=INK, lw=1.5, marker="s", ms=4, label="parent")
            for v in ("saved", "asran", "v0", "v1", "v2"):
                g = e[e.variant == v].set_index("k")[f"{col}_acc"].reindex(ks)
                if g.notna().any():
                    ax.plot(ks, g.values, color=VCOL[v], lw=1.1, marker="o", ms=3.6, alpha=0.95, ls="--" if v in ("saved", "asran") else "-", label=VLAB[v])
            ax.set_xticks(ks)
            ax.set_ylim(0.3, 1.0)
            ax.set_title(f"{s} {model}: {title}", fontsize=10)
            ax.set_xlabel("early finetune k (chain index)")
            if ci == 0:
                ax.set_ylabel("vote accuracy")
        ax = axes[ri, 2]
        ax.axvline(0, color=INK, lw=0.8)
        rng = np.random.default_rng(2)
        rows = []
        for vi, v in enumerate(("saved", "asran", "v0", "v1", "v2")):
            g = e[e.variant == v]
            for li, col in enumerate(("w2", "late")):
                x = (g[f"{col}_acc"] - g[f"{col}_parent"]).dropna().to_numpy()
                if len(x) == 0:
                    continue
                y = vi * 2.3 + li * 0.9
                ax.scatter(x, y + rng.uniform(-0.18, 0.18, len(x)), s=14, color=VCOL[v], alpha=0.4, lw=0)
                ax.scatter([x.mean()], [y], marker="D", s=40, color=VCOL[v], edgecolor=INK, lw=0.8, zorder=4)
                ax.text(0.985, y, f"{x.mean():+.3f}", va="center", ha="right", fontsize=7.6, color=INK, transform=ax.get_yaxis_transform())
                rows.append((y, f"{VLAB[v].split(' (')[0]} {'W2' if col == 'w2' else 'late'}", VCOL[v]))
        ax.set_yticks([r[0] for r in rows])
        ax.set_yticklabels([r[1] for r in rows], fontsize=7.6)
        for tl, r in zip(ax.get_yticklabels(), rows):
            tl.set_color(r[2])
        ax.set_ylim(max(r[0] for r in rows) + 0.8, -0.8)
        ax.set_xlim(-0.35, 0.58)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("child' - parent accuracy (diamond = mean over early finetunes)")
        ax.set_title("mean effect over the early finetunes", fontsize=10)
        if ri == 0:
            h, l = axes[0, 0].get_legend_handles_labels()
    figh = fig.get_size_inches()[1]
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 1 - 0.98 / figh))
    header(fig, f"F1b: model {model}'s early finetunes (trained during run 1, the first song(s)): do the corrected variants avoid the drop?",
           "Each line is one variant's child' per early finetune, every child retrained from the real parent of that finetune (not a chain). Accuracy = vote accuracy on the window, 5 seeds averaged.\n"
           "Late run = the session's last run, the same trials for every early finetune (parent and children ran on trials they never trained on).")
    fig.subplots_adjust(left=0.055, right=0.99, top=1 - 1.75 / figh, bottom=0.6 / figh + 0.03, hspace=0.45, wspace=0.42)
    fig.savefig(os.path.join(out, f"f1b_early_{model}.png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    args = ap.parse_args(argv)
    out = os.path.join(OUT, "figures" + args.name)
    os.makedirs(out, exist_ok=True)
    style()
    s0 = os.path.join(OUT, "step0b", f"step0b_per_finetune{args.name}.csv")
    if os.path.exists(s0):
        fig_step0b(pd.read_csv(s0), out)
    pf = os.path.join(OUT, f"f1b_per_finetune{args.name}.csv")
    if os.path.exists(pf):
        df = pd.read_csv(pf)
        pooled = pd.read_csv(os.path.join(OUT, f"f1b_pooled{args.name}.csv"))
        for w in ("w2", "w1"):
            fig_pooled(df, pooled, w, out)
        early = pd.read_csv(os.path.join(OUT, f"f1b_early{args.name}.csv"))
        early = early.rename(columns={"w2_acc": "w2_acc", "late_acc": "late_acc"})
        for m in "AB":
            fig_early(early, df, out, m)


if __name__ == "__main__":
    main()
