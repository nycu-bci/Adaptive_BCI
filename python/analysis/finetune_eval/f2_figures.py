# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f2_figures.py [--name _sub02-05]
"""
Figures for F2 (inputs: f2/f2_per_decision.csv, f2/f2_pooled.csv; outputs in f2/figures/):
  f2_timeline.png         per session and model: effect of a finetune at every block in session order, fired (filled) vs skipped (open)
  f2_fired_vs_skipped.png per model: per-decision effects and the pooled effect (95% CI) of fired and skipped blocks
  f2_gate_inputs.png      effect vs the gate's inputs (errors in the block, run accuracy) and the parent's accuracy on the previous 24 trials
  f2_paired.png           per block: accuracy if skipped (model kept) vs if finetuned, split by what the gate did
  f2_accuracy_timeline.png per session and model: both accuracies at every block, with the blocks the gate fired on
  f2_inuse_paired.png     (f2_inuse.py) keep vs finetune on the trials each choice is in use, per session x model: wrong / missed fires
  f2_inuse_gained.png     (f2_inuse.py) running total of trials gained at fired and at skipped blocks
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import f2_analyze as A
from wald_sprt_explainer import SURFACE, INK, INK2, MUTED, GRID, MODEL_COLOR, style, header

OUT = os.path.join(A.OUT, "figures")
FIRED_C, SKIP_C = "#1baf7a", "#6250d6"


def fig_timeline(df, out):
    panels = [(s, m) for s in df.session.unique() for m in "AB" if len(df[(df.session == s) & (df.model == m)])]
    widths = [max(3.5, 0.11 * len(df[(df.session == s) & (df.model == m)]) + 1) for s, m in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(sum(widths) + 1.2, 4.9), gridspec_kw={"width_ratios": widths}, squeeze=False, sharey=True)
    for ax, (s, m) in zip(axes[0], panels):
        g = df[(df.session == s) & (df.model == m)].sort_values("d")
        seeds = g[[f"all_effect_s{i}" for i in range(5)]].to_numpy()
        ax.axhline(0, color=INK, lw=0.8)
        ax.vlines(g.d, seeds.min(1), seeds.max(1), color=GRID, lw=1.4, zorder=1)
        f, k = g[g.fired == 1], g[g.fired == 0]
        ax.scatter(k.d, k.all_effect, s=26, facecolor=SURFACE, edgecolor=SKIP_C, lw=1.2, zorder=3, label="skipped by the gate")
        ax.scatter(f.d, f.all_effect, s=28, color=FIRED_C, zorder=4, label="fired by the gate")
        for r, h in g.groupby("run"):
            if r != g.run.min():
                ax.axvline(h.d.min() - 0.5, color=MUTED, lw=0.7, ls=":")
            ax.text(h.d.mean(), 0.33, f"run {r}", ha="center", fontsize=7.5, color=MUTED)
        ax.set_ylim(-0.35, 0.37)
        ax.set_title(f"{s} {m}: fired {len(f)}, skipped {len(k)}", fontsize=10)
        ax.set_xlabel("block (session order)")
    axes[0, 0].set_ylabel("child - parent accuracy (next 48 trials)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8.5, bbox_to_anchor=(0.99, 0.9))
    header(fig, "F2: what a finetune would have done at every block, fired or not",
           "Dot = 5-seed mean effect of a counterfactual V0 finetune (H4-correct trials) of the checkpoint the model had at that block; bar = seed range.\n"
           "Each dot is judged on the 48 trials after its block, so neighbouring dots share most trials. One dot has ~6 pp of test-trial noise.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.12, wspace=0.06)
    fig.savefig(os.path.join(out, "f2_timeline.png"), dpi=140)
    plt.close(fig)


def fig_fired_vs_skipped(df, pool, out):
    groups = [("A", ("A",)), ("B", ("B",)), ("A+B", ("A", "B"))]
    fig, axes = plt.subplots(len(groups), 1, figsize=(9.5, 7.2), sharex=True)
    rng = np.random.default_rng(1)
    for ax, (g, models) in zip(axes, groups):
        d = df[df.model.isin(models)]
        p = pool[(pool.group == g) & (pool.scope == "all blocks") & (pool.subset == "all")]
        ax.axvline(0, color=INK, lw=0.8)
        for li, (fired, lab, col) in enumerate(((1, "fired", FIRED_C), (0, "skipped", SKIP_C))):
            h = d[d.fired == fired]
            ax.scatter(h.all_effect, li + rng.uniform(-0.22, 0.22, len(h)), s=14, color=col, alpha=0.45, lw=0)
            if len(p):
                r = p.iloc[0]
                e, lo, hi = (r.eff_fired, r.ci_fired_lo, r.ci_fired_hi) if fired else (r.eff_skipped, r.ci_skipped_lo, r.ci_skipped_hi)
                if not np.isnan(e):
                    ax.errorbar([e], [li], xerr=[[max(e - lo, 0) if not np.isnan(lo) else 0], [max(hi - e, 0) if not np.isnan(hi) else 0]], fmt="D", color=INK,
                                ms=6, lw=1.6, capsize=3, zorder=4)
                    ax.text(0.99, li, f"{lab}: n {len(h)}, {e:+.3f} [{lo:+.3f}, {hi:+.3f}]", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
        if len(p):
            r = p.iloc[0]
            ax.text(0.99, 1.62, f"fired - skipped {r['diff']:+.3f} [{r.ci_diff_lo:+.3f}, {r.ci_diff_hi:+.3f}]", transform=ax.get_yaxis_transform(), ha="right",
                    fontsize=8.5, color=INK, fontweight="semibold")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["fired", "skipped"])
        ax.set_ylim(1.9, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_ylabel(g, rotation=0, labelpad=22, fontsize=11, va="center")
        ax.set_xlim(-0.3, 0.3)
    axes[-1].set_xlabel("child - parent accuracy on the next 48 trials")
    header(fig, "F2: does the gate fire where a finetune helps? Fired vs skipped blocks",
           "Dots = decisions (5-seed mean effect); diamond = pooled effect (trial-weighted) with a 95% CI over decisions. A good gate: fired > 0 and\n"
           "skipped <= 0. Windows of neighbouring blocks overlap, so the CIs are too narrow.")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.08, hspace=0.25)
    fig.savefig(os.path.join(out, "f2_fired_vs_skipped.png"), dpi=140)
    plt.close(fig)


def binned(x, y, n, bins):
    x = np.asarray(x, float)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        i = np.flatnonzero((x >= lo) & (x < hi))
        if len(i):
            out.append((x[i].mean(), A.pooled(y[i], n[i]), len(i)))
    return out


def fig_gate_inputs(df, out):
    d = df[df.is_cal == 0]
    cols = [("errors", "errors in the block (of 4)", None), ("run_acc", "run accuracy at the decision", A.RUN_ACC_BINS),
            ("parent_prev_acc", "parent accuracy, previous 24 trials", A.PREV_BINS)]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    rng = np.random.default_rng(3)
    for ax, (col, lab, bins) in zip(axes, cols):
        ax.axhline(0, color=INK, lw=0.8)
        for m in "AB":
            g = d[d.model == m]
            x = g[col].to_numpy(float) + (rng.uniform(-0.12, 0.12, len(g)) if col == "errors" else 0)
            ax.scatter(x, g.all_effect, s=np.where(g.fired == 1, 26, 18), facecolor=np.where(g.fired == 1, MODEL_COLOR[m], SURFACE), edgecolor=MODEL_COLOR[m],
                       lw=1.0, alpha=0.7, label=f"{m} (filled = fired)")
        y, n = d.all_effect.to_numpy(), d.all_n.to_numpy()
        if col == "errors":
            pts = [(e, A.pooled(y[d.errors == e], n[d.errors == e]), int((d.errors == e).sum())) for e in range(5) if (d.errors == e).any()]
        else:
            pts = binned(d[col].to_numpy(), y, n, bins)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-D", color=INK, ms=5, lw=1.4, label="pooled mean (A+B)")
        for px, py, pn in pts:
            ax.text(px, 0.31, f"n={pn}", ha="center", fontsize=7, color=MUTED)
        if col == "run_acc":
            ax.axvline(A.CF.THRESHOLD, color=INK2, lw=1, ls="--")
            ax.text(A.CF.THRESHOLD - 0.01, -0.33, "gate fires left of 0.75\n(if the block has an error)", ha="right", fontsize=7.5, color=INK2)
        if col == "errors":
            ax.text(0.5, -0.33, "gate needs >= 1", ha="center", fontsize=7.5, color=INK2)
        ax.set_xlabel(lab)
        ax.set_ylim(-0.35, 0.35)
    axes[0].set_ylabel("child - parent accuracy (next 48 trials)")
    axes[0].legend(fontsize=8, loc="lower right")
    header(fig, "F2: do the gate's inputs predict whether a finetune helps? (4-trial blocks; the calibration block is left out)",
           "If the gate's logic holds, the black line should rise with errors and fall with accuracy. Points = decisions (5-seed mean effect), filled = the gate fired.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.12, wspace=0.06)
    fig.savefig(os.path.join(out, "f2_gate_inputs.png"), dpi=140)
    plt.close(fig)


UP_C, DOWN_C, SAME_C = "#2a78d6", "#e34948", "#898781"
ONLINE_C = "#eb6834"
MIN_DRAW_N = 24  # blocks near the session end have few test trials: not drawn one by one (still in the means)


def fig_paired(df, out):
    """Per decision: accuracy on the next 48 trials if the block is skipped (the model is kept = parent) and if it is finetuned (child)."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6), sharey=True)
    for ri, m in enumerate("AB"):
        for ci, (fired, title) in enumerate(((1, "blocks the gate FIRED on"), (0, "blocks the gate SKIPPED"))):
            ax = axes[ri, ci]
            g = df[(df.model == m) & (df.fired == fired) & (df.all_n > 0)]
            p, c, n = g.all_parent_acc.to_numpy(), g.all_child_acc.to_numpy(), g.all_n.to_numpy()
            up, down = c > p, c < p
            for pi, ci_, u, d, ni in zip(p, c, up, down, n):
                if ni >= MIN_DRAW_N:
                    ax.plot([0, 1], [pi, ci_], color=UP_C if u else DOWN_C if d else SAME_C, ls="-" if u or not d else "--", lw=0.9, alpha=0.4, zorder=2)
            if len(g):
                mp, mc = A.pooled(p, n), A.pooled(c, n)
                ax.plot([0, 1], [mp, mc], color=INK, lw=2.6, marker="o", ms=8, zorder=4)
                ax.text(-0.06, mp, f"{mp:.3f}", ha="right", va="center", fontsize=9.5, color=INK, fontweight="semibold")
                ax.text(1.06, mc, f"{mc:.3f}", ha="left", va="center", fontsize=9.5, color=INK, fontweight="semibold")
                ax.text(0.5, 0.995, f"{len(g)} blocks: {int(up.sum())} up, {int(down.sum())} down, {int((~up & ~down).sum())} same\n"
                        f"mean change {mc - mp:+.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK2)
            else:
                ax.text(0.5, 0.5, "no blocks", transform=ax.transAxes, ha="center", color=MUTED)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["if skipped\n(model kept)", "if finetuned"])
            ax.set_xlim(-0.35, 1.35)
            ax.set_ylim(0.3, 1.02)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"model {m}: {title}", fontsize=10.5)
            if ci == 0:
                ax.set_ylabel("vote accuracy on the next 48 trials")
    h = [plt.Line2D([], [], color=UP_C, lw=1.5), plt.Line2D([], [], color=DOWN_C, lw=1.5, ls="--"), plt.Line2D([], [], color=INK, lw=2.6, marker="o")]
    fig.legend(h, ["finetune raises accuracy", "finetune lowers accuracy", "mean over blocks (trial-weighted)"], loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    header(fig, "F2: accuracy after each block if it is skipped vs if it is finetuned",
           "Each line is one block: left = the model as it was (what skipping gives), right = after a finetune (5-seed mean, H4-correct trials), both on the\n"
           "same 48 trials after the block. Left column = what the gate's finetunes did; right column = what the skipped blocks would have gained or lost.\n"
           f"Blocks with fewer than {MIN_DRAW_N} later trials (session end) are not drawn but count in the mean and the up/down counts.")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.1, hspace=0.3, wspace=0.12)
    fig.savefig(os.path.join(out, "f2_paired.png"), dpi=140)
    plt.close(fig)


def fig_accuracy_timeline(df, out):
    """Per session and model: accuracy on the next 48 trials if skipped (parent) and if finetuned (child) at every block; ticks = the gate fired."""
    panels = [(s, m) for s in df.session.unique() for m in "AB" if len(df[(df.session == s) & (df.model == m)])]
    ncol = 2
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow + 1.6), squeeze=False)
    for ax, (s, m) in zip(axes.flat, panels):
        g = df[(df.session == s) & (df.model == m) & (df.all_n >= MIN_DRAW_N)].sort_values("d")
        x, p, c = g.d.to_numpy(), g.all_parent_acc.to_numpy(), g.all_child_acc.to_numpy()
        ax.fill_between(x, p, c, where=c >= p, color=UP_C, alpha=0.18, lw=0, interpolate=True)
        ax.fill_between(x, p, c, where=c < p, color=DOWN_C, alpha=0.18, lw=0, interpolate=True)
        ax.plot(x, p, color=INK, lw=1.6, label="if skipped (model kept)")
        ax.plot(x, c, color=UP_C, lw=1.6, label="if finetuned")
        if "online_acc" in g:
            ax.plot(x, g.online_acc.to_numpy(), color=ONLINE_C, lw=1.4, ls=(0, (4, 2)), label="online (model actually in use)")
        f = g[g.fired == 1]
        ax.plot(f.d, np.full(len(f), 0.33), "|", color=INK2, ms=9, mew=1.6, label="gate fired")
        for r, h in g.groupby("run"):
            if r != g.run.min():
                ax.axvline(h.d.min() - 0.5, color=MUTED, lw=0.7, ls=":")
            ax.text(h.d.mean(), 0.985, f"run {r}", ha="center", va="top", fontsize=7.5, color=MUTED)
        ax.set_ylim(0.3, 1.0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f"{s} {m}", fontsize=10)
        ax.set_xlabel("block (session order)")
        ax.set_ylabel("accuracy, next 48 trials")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_axis_off()
    h, l = axes[0, 0].get_legend_handles_labels()
    figh = fig.get_size_inches()[1]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.1 / figh))
    fig.text(0.02, 1 - 0.2 / figh, "F2: accuracy over the session if each block is skipped vs finetuned", fontsize=12.5, fontweight="semibold", va="top")
    fig.text(0.02, 1 - 0.55 / figh,
           "Black = the model kept at that block; blue = the same model after a finetune there. Blue shading = finetuning would raise accuracy, red = lower it.\n"
           "Orange dashed = online: the checkpoint actually in use at each of those trials (includes the gate's later finetunes, trained on the ~1 s late cut).\n"
           "Ticks at the bottom = blocks where the gate fired. Each point is the accuracy on the 48 trials after its block, so neighbouring points share most trials.\n"
           f"Blocks with fewer than {MIN_DRAW_N} later trials (the session end) are left out.", fontsize=9.2, color=INK2, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 1.75 / figh, bottom=1.0 / figh, hspace=0.55, wspace=0.12)
    fig.savefig(os.path.join(out, "f2_accuracy_timeline.png"), dpi=140)
    plt.close(fig)


def spread(ys, gap=0.022):
    """Label heights pushed apart so neighbouring labels do not collide (order kept)."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def fig_inuse_paired(df, out):
    """In-use windows: per session x model, pooled accuracy if the model is kept vs if finetuned; fired / skipped blocks, both window versions."""
    import f2_inuse as I
    units = [(s, m) for s in df.session.unique() for m in "AB"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4), sharey=True)
    for ri, (pre, vname) in enumerate(I.VERSIONS):
        for ci, (fired, title, bad, badname) in enumerate(((1, "blocks the gate FIRED on", -1, "wrong fires"),
                                                           (0, "blocks the gate SKIPPED", 1, "missed fires"))):
            ax = axes[ri, ci]
            g = df[(df.fired == fired) & (df[pre + "n"] > 0)]
            ends, labs = [], []
            for s, m in units:
                h = g[(g.session == s) & (g.model == m)]
                if h.empty:
                    continue
                n = h[pre + "n"]
                k, f = A.pooled(h[pre + "keep_acc"], n), A.pooled(h[pre + "ft_acc"], n)
                up, down = f > k, f < k
                ax.plot([0, 1], [k, f], color=UP_C if up else DOWN_C if down else SAME_C, ls="--" if down else "-", lw=1.6, marker="o", ms=4, zorder=3)
                ends.append(f)
                labs.append(f"{s} {m} ({len(h)}: {int((h[pre + 'direction'] == bad).sum())} {badname.split()[0]})")
            for y, t in zip(spread(ends), [labs[i] for i in range(len(labs))]):
                ax.text(1.05, y, t, va="center", fontsize=7.6, color=INK2)
            if len(g):
                n = g[pre + "n"]
                mk, mf = A.pooled(g[pre + "keep_acc"], n), A.pooled(g[pre + "ft_acc"], n)
                ax.plot([0, 1], [mk, mf], color=INK, lw=2.8, marker="D", ms=7, zorder=4)
                ax.text(-0.06, mk, f"{mk:.3f}", ha="right", va="center", fontsize=9.5, color=INK, fontweight="semibold")
                nb = int((g[pre + "direction"] == bad).sum())
                ax.text(0.5, 0.995, f"all: {len(g)} blocks, {int(n.sum())} trials, change {mf - mk:+.3f}\n{badname}: {nb}/{len(g)}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["keep the\nold model", "finetune\nat the block"])
            ax.set_xlim(-0.3, 1.9)
            ax.set_ylim(0.35, 1.02)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{title}: {vname}", fontsize=10)
            if ci == 0:
                ax.set_ylabel("vote accuracy over the trials the choice is in use")
    h = [plt.Line2D([], [], color=UP_C, lw=1.6, marker="o", ms=4), plt.Line2D([], [], color=DOWN_C, lw=1.6, ls="--", marker="o", ms=4),
         plt.Line2D([], [], color=INK, lw=2.8, marker="D")]
    fig.legend(h, ["one session x model: finetune better", "one session x model: keep better", "all, pooled (trial-weighted)"], loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.005))
    header(fig, "F2: wrong fires and missed fires, judged on the trials each choice is in use",
           "Each block is scored from the next block to the end of the next block the gate fired on (or the session end; at most 48 trials).\n"
           "Left: blue = the gate's fire helped; a red line or a fired block counted as 'wrong' = keeping the old model was better. Right: blue = a fire was missed.\n"
           "Bottom row drops the window's last block when it is the next fire (chosen for the live model's errors). Labels: blocks, clear errors (>= 1 trial, >= 4/5 seeds).")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.08, hspace=0.3, wspace=0.08)
    fig.savefig(os.path.join(out, "f2_inuse_paired.png"), dpi=140)
    plt.close(fig)


def fig_inuse_gained(df, out):
    """In-use windows: running total over the session of (finetuned - kept) correct trials, at fired and at skipped blocks."""
    import f2_inuse as I
    panels = [(s, m) for s in df.session.unique() for m in "AB" if len(df[(df.session == s) & (df.model == m)])]
    ncol = 2
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.0 * nrow + 1.8), squeeze=False)
    for ax, (s, m) in zip(axes.flat, panels):
        g = df[(df.session == s) & (df.model == m)].sort_values("d")
        ax.axhline(0, color=INK, lw=0.8)
        for fired, col, name in ((1, FIRED_C, "fired blocks: what the gate's fires gained"), (0, SKIP_C, "skipped blocks: what a fire would have gained")):
            for pre, vname in I.VERSIONS:
                h = g[(g.fired == fired) & (g[pre + "n"] > 0)]
                if h.empty:
                    continue
                ax.step(np.r_[g.d.min() - 1, h.d], np.r_[0, h[pre + "effect_trials"].cumsum()], where="post", color=col, lw=1.6,
                        ls="-" if pre == "" else (0, (4, 2)), label=f"{name} ({vname})")
        for r, h in g.groupby("run"):
            if r != g.run.min():
                ax.axvline(h.d.min() - 0.5, color=MUTED, lw=0.7, ls=":")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f"{s} {m}", fontsize=10)
        ax.set_xlabel("block (session order)")
        ax.set_ylabel("trials gained (running total)")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_axis_off()
    hs, ls = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls:
                hs.append(hh)
                ls.append(ll)
    figh = fig.get_size_inches()[1]
    fig.legend(hs, ls, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.05 / figh))
    fig.text(0.02, 1 - 0.2 / figh, "F2: trials gained by finetuning (finetuned - kept correct trials), added up over the session", fontsize=12.5,
             fontweight="semibold", va="top")
    fig.text(0.02, 1 - 0.55 / figh,
             "Green rising = the gate's fires paid off; green falling = wrong fires. Purple rising = fires the gate missed; purple falling = skips that were right.\n"
             "Solid = full in-use window; dashed = without the window's last block when it is the next fire. Skipped windows overlap, so purple double-counts trials.",
             fontsize=9.2, color=INK2, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 1.25 / figh, bottom=1.25 / figh, hspace=0.6, wspace=0.15)
    fig.savefig(os.path.join(out, "f2_inuse_gained.png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="suffix of f2_analyze's outputs; figures go to figures<name>/")
    args = ap.parse_args(argv)
    out = OUT + args.name
    os.makedirs(out, exist_ok=True)
    style()
    df = pd.read_csv(os.path.join(A.OUT, f"f2_per_decision{args.name}.csv"))
    pool = pd.read_csv(os.path.join(A.OUT, f"f2_pooled{args.name}.csv"))
    fig_timeline(df, out)
    fig_fired_vs_skipped(df, pool, out)
    fig_gate_inputs(df, out)
    fig_paired(df, out)
    fig_accuracy_timeline(df, out)
    inuse = os.path.join(A.OUT, f"inuse_per_decision{args.name}.csv")
    if os.path.exists(inuse):
        iu = pd.read_csv(inuse)
        fig_inuse_paired(iu, out)
        fig_inuse_gained(iu, out)


if __name__ == "__main__":
    main()
