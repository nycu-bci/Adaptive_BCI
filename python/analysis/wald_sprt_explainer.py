# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/wald_sprt_explainer.py --session python/main/real_time_data/sub01/ses1 --out python/analysis/results/wald_sprt
"""
Figures that explain the Wald SPRT live-model switching rule (known issue H1), using a recorded session.

Data per run{N}/:
  log_*.txt      'Trial i END: ... LABEL: y MODEL_PREDICTION: A=a B=b LIVE=L ...' -> paired per-trial correctness
  ab_blocks.csv  n_trials per block -> where Python made its block decisions

The replay uses the logged votes (Python's replica of Unity's vote, for both models) and keeps the
recorded order of trials, runs and blocks. It only shows what each rule would have decided; it can't
show what the subject would have seen after a switch.
"""
import argparse
import csv
import glob
import math
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {"A": "#2a78d6", "B": "#eb6834"}
RULE_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
NEUTRAL_DARK, NEUTRAL_LIGHT = "#c3c2b7", "#e1e0d9"

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE_RE = re.compile(r"Trial (\d+) END: [\d.]+ LABEL: (\d) MODEL_PREDICTION: A=(\w+) B=(\w+) LIVE=(\w)")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def header(fig, title, sub):
    """Figure title + grey subtitle above the plot area (keeps them from overlapping the axes title slot)."""
    fig.text(0.02, 0.975, title, fontsize=12.5, fontweight="semibold", va="top")
    fig.text(0.02, 0.925, sub, fontsize=9.2, color=INK2, va="top", linespacing=1.4)


# ---------------- data ----------------
def load_session(session_dir):
    """-> list of runs; each run = {'name', 'trials': [(a_ok, b_ok)], 'blocks': [n_trials, ...]}"""
    runs = []
    run_dirs = sorted(glob.glob(os.path.join(session_dir, "run*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
    for d in run_dirs:
        logs = glob.glob(os.path.join(d, "log_*.txt"))
        blocks_path = os.path.join(d, "ab_blocks.csv")
        if not logs or not os.path.exists(blocks_path):
            continue
        trials = []
        for log in sorted(logs):
            with open(log, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = LINE_RE.search(line)
                    if not m or m.group(3) == "NA" or m.group(4) == "NA":
                        continue
                    y = m.group(2)
                    trials.append((m.group(3) == y, m.group(4) == y))
        with open(blocks_path, encoding="utf-8") as f:
            blocks = [int(r["n_trials"]) for r in csv.DictReader(f)]
        if sum(blocks) != len(trials):
            raise ValueError(f"{d}: blocks sum to {sum(blocks)} but the log has {len(trials)} trials")
        runs.append({"name": os.path.basename(d), "trials": trials, "blocks": blocks})
    return runs


# ---------------- rules ----------------
class SPRT:
    def __init__(self, p1=0.7, alpha=0.05, beta=0.2):
        self.p1, self.alpha, self.beta = p1, alpha, beta
        self.win = math.log(p1 / 0.5)
        self.loss = math.log((1 - p1) / 0.5)
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))

    def label(self):
        return f"SPRT p1={self.p1:g}, α={self.alpha:g}, β={self.beta:g}"


def replay(runs, sprt, start_live="A"):
    """Trial-by-trial SPRT walk, decisions at block ends. Returns per-trial trace and decision events."""
    live, llr = start_live, 0.0
    trace, events, episodes = [], [], []
    episode = [(0, 0)]  # (challenger losses, challenger wins) path of the current episode
    t = 0
    run_starts = []
    for run in runs:
        run_starts.append(t)
        i = 0
        for size in run["blocks"]:
            for a_ok, b_ok in run["trials"][i:i + size]:
                ok = {"A": a_ok, "B": b_ok}
                chal = "B" if live == "A" else "A"
                step = 0.0
                if ok[chal] != ok[live]:
                    step = sprt.win if ok[chal] else sprt.loss
                    w, l = episode[-1][1], episode[-1][0]
                    episode.append((l + (0 if ok[chal] else 1), w + (1 if ok[chal] else 0)))
                llr += step
                trace.append({"t": t, "llr": llr, "live": live, "step": step})
                t += 1
            i += size
            if llr >= sprt.upper:
                events.append({"t": t, "kind": "switch", "from": live, "llr": llr})
                episodes.append({"path": episode, "end": "switch", "challenger": "B" if live == "A" else "A"})
                live, llr, episode = ("B" if live == "A" else "A"), 0.0, [(0, 0)]
            elif llr <= sprt.lower:
                events.append({"t": t, "kind": "reset", "from": live, "llr": llr})
                episodes.append({"path": episode, "end": "reset", "challenger": "B" if live == "A" else "A"})
                llr, episode = 0.0, [(0, 0)]
    episodes.append({"path": episode, "end": "open", "challenger": "B" if live == "A" else "A"})
    return trace, events, episodes, run_starts


def replay_streak(runs, k, start_live="A"):
    """Current rule: switch when the other model has strictly higher block accuracy k blocks in a row."""
    live, streak, t, events = start_live, 0, 0, []
    for run in runs:
        i = 0
        for size in run["blocks"]:
            blk = run["trials"][i:i + size]
            i += size
            t += size
            acc = {"A": sum(a for a, _ in blk), "B": sum(b for _, b in blk)}
            other = "B" if live == "A" else "A"
            streak = streak + 1 if acc[other] > acc[live] else 0
            if streak >= k:
                events.append({"t": t, "from": live})
                live, streak = other, 0
    return events


# ---------------- figures ----------------
def fig_evidence(runs, out):
    """Fig 1: only the trials where A and B disagree carry evidence about which is better."""
    names = [r["name"].replace("run", "Run ") for r in runs]
    cats = [("Both right", NEUTRAL_DARK), ("Both wrong", NEUTRAL_LIGHT), ("A right, B wrong", MODEL_COLOR["A"]),
            ("B right, A wrong", MODEL_COLOR["B"])]
    counts = np.zeros((len(runs), 4), int)
    for i, r in enumerate(runs):
        for a, b in r["trials"]:
            counts[i, 0 if a and b else 1 if not a and not b else 2 if a else 3] += 1
    frac = counts / counts.sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(runs))[::-1]
    left = np.zeros(len(runs))
    for j, (lab, col) in enumerate(cats):
        ax.barh(y, frac[:, j], left=left, color=col, height=0.62, edgecolor=SURFACE, linewidth=2, label=lab)
        for yi, l, f, c in zip(y, left, frac[:, j], counts[:, j]):
            if f > 0.06:
                ax.text(l + f / 2, yi, str(c), ha="center", va="center", fontsize=9,
                        color="white" if j >= 2 else INK)
        left += frac[:, j]
    ax.set_yticks(y, [f"{n}  (n={counts[i].sum()})" for i, n in enumerate(names)])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of trials (numbers are trial counts)")
    tot = counts.sum(0)
    disc = tot[2] + tot[3]
    header(fig, "1. Only disagreements are evidence",
           f"Gray trials look the same for both models. Only the {disc} of {tot.sum()} trials "
           f"({disc / tot.sum():.0%}) where A and B disagree say which model is better.")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.28), fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.84, bottom=0.22)
    fig.savefig(os.path.join(out, "fig1_disagreements.png"), dpi=150)
    plt.close(fig)
    return counts


def fig_scoring(sprt, out):
    """Fig 2: how one disagreement moves the score, and where the two lines come from."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.bar([0, 1], [sprt.win, sprt.loss], color=[MODEL_COLOR["B"], MODEL_COLOR["A"]], width=0.5)
    ax1.axhline(0, color=AXIS, lw=1)
    ax1.set_xticks([0, 1], ["Challenger right,\nlive model wrong", "Live model right,\nchallenger wrong"])
    ax1.grid(axis="x", visible=False)
    for x, v in [(0, sprt.win), (1, sprt.loss)]:
        ax1.text(x, v + (0.03 if v > 0 else -0.03), f"{v:+.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=10)
    ax1.set_ylim(-0.75, 0.6)
    ax1.set_ylabel("Change in score (log-likelihood ratio)")
    ax1.set_title("2a. One disagreement = one step")
    ax1.text(0.5, -0.72, f"+ln({sprt.p1:g}/0.5)  and  ln({1 - sprt.p1:.1f}/0.5)\nAgreements: 0", ha="center",
             va="bottom", fontsize=8.5, color=INK2)

    # 2b: the two lines; how many straight wins needed for different p1
    ax2.axhline(sprt.upper, color=INK, lw=1.4, ls="--")
    ax2.axhline(sprt.lower, color=INK2, lw=1.4, ls=":")
    ax2.axhline(0, color=AXIS, lw=1)
    ax2.text(0.2, sprt.upper + 0.12, f"Switch line  ln((1−β)/α) = {sprt.upper:.2f}", fontsize=9)
    ax2.text(0.2, sprt.lower - 0.3, f"Reset line  ln(β/(1−α)) = {sprt.lower:.2f}", fontsize=9, color=INK2)
    seq = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # made-up: mostly challenger wins
    s = np.concatenate([[0], np.cumsum([sprt.win if w else sprt.loss for w in seq])])
    hit = int(np.argmax(s >= sprt.upper)) if (s >= sprt.upper).any() else len(s) - 1
    x = np.arange(hit + 1)
    ax2.plot(x, s[:hit + 1], "-", color=MODEL_COLOR["B"], lw=1.5)
    for xi, a, b in zip(x[1:], s[:hit], s[1:hit + 1]):
        ax2.plot(xi, b, "o", color=MODEL_COLOR["B"] if b > a else MODEL_COLOR["A"], ms=6, mec=SURFACE, mew=1.5, zorder=4)
    '''
    ax2.annotate(f"switch after {hit} disagreements\n({sum(seq[:hit])} wins, {hit - sum(seq[:hit])} losses)",
                 (hit, s[hit]), xytext=(hit - 4.5, -0.9), fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    '''
    ax2.set_xticks(range(0, hit + 2, 2))
    ax2.set_xlabel("Disagreements seen (made-up example)")
    ax2.set_ylabel("Running score")
    ax2.set_ylim(-2.2, 3.6)
    ax2.set_title("2b. Add the steps until a line is crossed")
    need = math.ceil(sprt.upper / sprt.win)
    fig.text(0.08, 0.015, f"With p1={sprt.p1:g}: switching needs at least {need} challenger wins in a row; "
             f"each live-model win cancels {abs(sprt.loss) / sprt.win:.1f} of them.\n"
             "α = chance of a false switch between equal models; β = chance of missing a challenger that "
             f"really wins {sprt.p1:.0%} of disagreements.", fontsize=8.5, color=INK2, linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.24, wspace=0.3)
    fig.savefig(os.path.join(out, "fig2_scoring.png"), dpi=150)
    plt.close(fig)


def fig_walk(runs, sprt, trace, events, run_starts, streak_events, out):
    """Fig 3: the SPRT score over the whole recorded session, with the current streak rule for comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.array([d["t"] for d in trace]) + 1
    llr = np.array([d["llr"] for d in trace])
    live = [d["live"] for d in trace]
    # colour the line by which model is the challenger (the one the score is "for")
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or live[i] != live[start]:
            chal = "B" if live[start] == "A" else "A"
            ax.step(np.r_[t[start] - 1, t[start:i]], np.r_[0 if start == 0 else llr[start - 1], llr[start:i]],
                    where="post", color=MODEL_COLOR[chal], lw=1.6)
            start = i
    ax.axhline(sprt.upper, color=INK, lw=1.2, ls="--")
    ax.axhline(sprt.lower, color=INK2, lw=1.2, ls=":")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.text(t[-1] + 4, sprt.upper + 0.08, "switch", va="bottom", fontsize=9)
    ax.text(t[-1] + 4, sprt.lower + 0.08, "reset", va="bottom", fontsize=9, color=INK2)
    for s, r in zip(run_starts, runs):
        ax.axvline(s, color=AXIS, lw=0.8)
        ax.text(s + 2, 4.05, r["name"].replace("run", "Run "), fontsize=8.5, color=MUTED)
    ax.text(24, -6.0, "calibration song:\none 48-trial block", fontsize=8, color=MUTED, ha="center")
    for e in events:
        if e["kind"] == "switch":
            to = "B" if e["from"] == "A" else "A"
            ax.plot(e["t"], e["llr"], "o", ms=9, color=MODEL_COLOR[to], mec=SURFACE, mew=2, zorder=5)
            ax.annotate(f"switch {e['from']}→{to}\n(trial {e['t']})", (e["t"], e["llr"]), xytext=(e["t"] - 95, 3.3),
                        fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        else:
            ax.plot(e["t"], e["llr"], "x", ms=6, color=INK2, mew=1.5, zorder=5)
    for e in streak_events:
        ax.plot(e["t"], -6.5, "v", ms=7, color=INK, zorder=5)
        ax.text(e["t"], -6.15, f"t={e['t']}", fontsize=8, ha="center", va="bottom")
    ax.text(t[-1] + 4, -5.2, "▼ current rule\n(2 blocks in a row)\nwould switch", fontsize=8.5, va="center")
    ax.set_ylim(-6.9, 4.4)
    ax.set_xlim(0, t[-1] + 50)
    ax.set_xlabel("Trial (2026-10-15 recording, all runs in order)")
    ax.set_ylabel("SPRT score for the challenger")
    n_sw = sum(e["kind"] == "switch" for e in events)
    n_rs = sum(e["kind"] == "reset" for e in events)
    header(fig, f"3. Replay on the real data: {sprt.label()}",
           f"Orange line = B challenging A; blue line = A challenging B. SPRT: {n_sw} switch, {n_rs} resets (×), "
           f"score set back to 0 after each. Current rule: {len(streak_events)} switches (▼). "
           "Decisions are checked at block ends, as in Python.")
    fig.subplots_adjust(left=0.065, right=0.9, top=0.86, bottom=0.11)
    fig.savefig(os.path.join(out, "fig3_replay_walk.png"), dpi=150)
    plt.close(fig)


def fig_plane(sprt, episodes, out):
    """Fig 4: the same rule drawn as two parallel lines in the (losses, wins) plane."""
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    maxl = max(max(p[0] for p in e["path"]) for e in episodes) + 3
    L = np.linspace(0, maxl, 50)
    up = (sprt.upper - sprt.loss * L) / sprt.win
    lo = (sprt.lower - sprt.loss * L) / sprt.win
    ax.fill_between(L, up, up.max() + 20, color=MODEL_COLOR["B"], alpha=0.08, lw=0)
    ax.fill_between(L, -5, lo, color=INK2, alpha=0.07, lw=0)
    ax.plot(L, up, color=INK, ls="--", lw=1.2)
    ax.plot(L, lo, color=INK2, ls=":", lw=1.2)
    for e in episodes:
        p = np.array(e["path"])
        if len(p) < 2:
            continue
        col = MODEL_COLOR[e["challenger"]]
        strong = e["end"] == "switch"
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.2 if strong else 1.1, alpha=1 if strong else 0.55,
                drawstyle="steps-post" if False else "default", zorder=4 if strong else 3)
        ax.plot(p[-1, 0], p[-1, 1], "o" if strong else ("x" if e["end"] == "reset" else "s"), color=col,
                ms=8 if strong else 5, mec=SURFACE if strong else col, zorder=5)
    ymax = max(max(pt[1] for pt in e["path"]) for e in episodes) + 4
    ax.set_xlim(0, maxl)
    ax.set_ylim(0, max(ymax, up[0] + 4))
    ax.text(0.4, up[0] + 0.6 + 0.0, "SWITCH region", fontsize=9, color=INK, va="bottom")
    ax.text(maxl * 0.55, (sprt.lower - sprt.loss * maxl * 0.55) / sprt.win - 1.2, "RESET region", fontsize=9, color=INK2, va="top")
    ax.set_xlabel("Challenger losses (live model right, challenger wrong)")
    ax.set_ylabel("Challenger wins (challenger right, live model wrong)")
    slope = -sprt.loss / sprt.win
    header(fig, "4. The same rule as a map",
           f"Each comparison starts at (0, 0) and moves right on a loss, up on a win. Both lines have slope {slope:.2f}:\n"
           f"the challenger needs more than {slope:.1f} wins per loss to move towards a switch.\n"
           "Thick = ended in a switch, × = reset, ■ = still open at the end of the recording.")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.1)
    fig.savefig(os.path.join(out, "fig4_wins_losses_map.png"), dpi=150)
    plt.close(fig)


def simulate(p_grid, d, n_blocks, rules, n_sim=20000, block=4, seed=0):
    """Trial model: each trial is a disagreement with prob d; the challenger wins a disagreement with prob p.
    Returns {rule: (P(switch within n_blocks), median trials to first switch or nan)} per p."""
    rng = np.random.default_rng(seed)
    out = {name: (np.zeros(len(p_grid)), np.full(len(p_grid), np.nan)) for name, _ in rules}
    for pi, p in enumerate(p_grid):
        disc = rng.random((n_sim, n_blocks, block)) < d
        win = rng.random((n_sim, n_blocks, block)) < p
        wins = (disc & win).sum(2)
        losses = (disc & ~win).sum(2)
        for name, rule in rules:
            first = np.full(n_sim, -1)
            if isinstance(rule, SPRT):
                llr = np.zeros(n_sim)
                for b in range(n_blocks):
                    llr += wins[:, b] * rule.win + losses[:, b] * rule.loss
                    hit = (llr >= rule.upper) & (first < 0)
                    first[hit] = b
                    llr[llr <= rule.lower] = 0.0
                    llr[first >= 0] = -1e9  # stop after the first switch
            else:
                streak = np.zeros(n_sim, int)
                for b in range(n_blocks):
                    streak = np.where(wins[:, b] > losses[:, b], streak + 1, 0)
                    hit = (streak >= rule) & (first < 0)
                    first[hit] = b
            sw = first >= 0
            out[name][0][pi] = sw.mean()
            if sw.mean() >= 0.5:
                out[name][1][pi] = np.median(np.where(sw, (first + 1) * block, np.inf))
    return out


def fig_simulation(d, rules, observed, out):
    """Fig 5: how often each rule switches, as a function of how much better the challenger really is."""
    p_grid = np.round(np.arange(0.3, 0.901, 0.025), 3)
    one_run = simulate(p_grid, d, 12, rules)
    long = simulate(p_grid, d, 120, rules, n_sim=5000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.9))
    for k, (name, _) in enumerate(rules):
        c = RULE_COLOR[k]
        ax1.plot(p_grid, one_run[name][0], color=c, lw=2, label=name)
        ax2.plot(p_grid, long[name][1], color=c, lw=2, label=name)
    for ax in (ax1, ax2):
        ax.axvline(0.5, color=INK2, lw=1, ls=":")
        for lab, pv in observed:
            ax.axvline(pv, color=MUTED, lw=0.8)
    ax1.text(0.495, 0.45, "equal models", fontsize=8.5, color=INK2, rotation=90, ha="right", va="bottom")
    for lab, pv in observed:
        ax1.text(pv - 0.006, 0.45, f"B in {lab}", fontsize=8, color=MUTED, rotation=90, ha="right", va="bottom")
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_xlabel("Challenger's true win rate among disagreements (p)")
    ax1.set_ylabel("Chance of at least one switch")
    ax1.set_title("5a. Switches within one run (48 trials)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax2.set_xlabel("Challenger's true win rate among disagreements (p)")
    ax2.set_ylabel("Median trials until the first switch")
    ax2.set_title("5b. How long a switch takes (up to 480 trials)")
    ax2.set_ylim(0, 480)
    ax2.text(0.31, 460, "no line = fewer than half of the\nsimulated sessions switch within 480 trials",
             fontsize=8.5, color=INK2, va="top")
    fig.text(0.06, 0.015, f"Simulation: each trial is a disagreement with probability {d:.2f} (the rate in the recording); "
             "blocks of 4 trials; independent trials; models do not change. Left of the dotted line every switch is a "
             "wrong switch.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.17, wspace=0.22)
    fig.savefig(os.path.join(out, "fig5_simulation.png"), dpi=150)
    plt.close(fig)
    return p_grid, one_run, long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join(PY_DIR, "main", "real_time_data", "sub01", "ses1"))
    ap.add_argument("--out", default=os.path.join(PY_DIR, "analysis", "results", "wald_sprt"))
    ap.add_argument("--p1", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()

    runs = load_session(args.session)
    sprt = SPRT(args.p1, args.alpha, args.beta)
    counts = fig_evidence(runs, args.out)
    fig_scoring(sprt, args.out)
    trace, events, episodes, run_starts = replay(runs, sprt)
    streak_events = replay_streak(runs, 2)
    fig_walk(runs, sprt, trace, events, run_starts, streak_events, args.out)
    fig_plane(sprt, episodes, args.out)

    tot = counts.sum(0)
    d = (tot[2] + tot[3]) / tot.sum()
    b_rest = counts[1:, 3].sum() / counts[1:, 2:].sum()
    b_run1 = counts[0, 3] / counts[0, 2:].sum()
    rules = [("Current: 2 blocks in a row", 2), ("3 blocks in a row", 3), (sprt.label(), sprt),
             ("SPRT p1=0.65, α=0.05, β=0.2", SPRT(0.65, 0.05, 0.2))]
    p_grid, one_run, long = fig_simulation(d, rules, [(f"run 1: {b_run1:.2f}", b_run1), (f"runs 2–7: {b_rest:.2f}", b_rest)],
                                           args.out)

    lines = [f"session: {args.session}", f"rule: {sprt.label()}  win {sprt.win:+.3f}  loss {sprt.loss:+.3f}  "
             f"upper {sprt.upper:.3f}  lower {sprt.lower:.3f}", ""]
    lines.append("run, n, both_right, both_wrong, A_only, B_only")
    for r, c in zip(runs, counts):
        lines.append(f"{r['name']}, {c.sum()}, {c[0]}, {c[1]}, {c[2]}, {c[3]}")
    lines.append(f"disagreement rate d = {d:.3f}; B win rate among disagreements: run1 {b_run1:.3f}, runs2-7 {b_rest:.3f}")
    lines.append("")
    lines.append("SPRT replay events (t = trials since session start, at block end):")
    for e in events:
        lines.append(f"  t={e['t']:4d}  {e['kind']:6s}  live before={e['from']}  score={e['llr']:+.2f}")
    lines.append(f"final open score {trace[-1]['llr']:+.2f}, live {trace[-1]['live']}")
    lines.append("streak-2 replay switches: " + ", ".join(f"t={e['t']} ({e['from']}->{'B' if e['from'] == 'A' else 'A'})"
                                                          for e in streak_events))
    lines.append("")
    lines.append("simulation, P(>=1 switch within 48 trials) at p = 0.5 / 0.6 / 0.7:")
    for name, _ in rules:
        vals = [one_run[name][0][np.argmin(abs(p_grid - v))] for v in (0.5, 0.6, 0.7)]
        lines.append(f"  {name:32s} " + " / ".join(f"{v:.3f}" for v in vals))
    lines.append("simulation, P(>=1 switch within 480 trials) at p = 0.5:")
    for name, _ in rules:
        lines.append(f"  {name:32s} {long[name][0][np.argmin(abs(p_grid - 0.5))]:.3f}")
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
