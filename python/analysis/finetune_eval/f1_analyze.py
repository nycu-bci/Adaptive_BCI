# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f1_analyze.py
"""
Step 3 (numbers): F1 chain experiment from results/finetune_eval/trial_checkpoint.csv.gz.

For every finetune (parent = chain checkpoint before the child), parent and child vote on the same later trials:
  W1 = trials with swap_time < AT <= the model's next swap (or session end), W2 = the first 48 trials after the swap (may span runs).
Each test trial is tagged by whether the finetuned model was live (LIVE= in the log). Subsets: not live, live, all.
Pooled per model and overall: exact two-sided sign test on the discordant W1 trials, accuracy change child - parent with a 95% bootstrap
CI (sessions resampled, then finetunes within a session; trial-weighted pooled change), share of finetunes worse / equal / better on W2,
effect vs the parent's accuracy on the (up to) 12 trials before the swap.

Outputs in results/finetune_eval/: finetune_results.csv (one row per finetune), summary.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_common as F

EXPECTED_FT = {"sub01": (31, 20), "sub02": (12, 11), "sub03": (28, 20), "sub04": (17, 17), "sub05": (0, 10)}
SUBSETS = ("all", "live", "notlive")
WINDOWS = ("w1", "w2")
MIN_PRE = 4


# ---------------- data ----------------
class SessionData:
    """One session: Session (chains, trials) + arrays ok/margin/in_sample (n_trials, n_ckpts) per model from the table."""

    def __init__(self, sub, df):
        self.sub = sub
        self.sess = F.Session(sub, load_eeg=False)
        s = self.sess
        self.ok, self.margin, self.insample, self.on_chain = {}, {}, {}, {}
        for m in "AB":
            d = df[(df.session == sub) & (df.model == m)]
            n_ck = len(s.chain[m])
            assert d.ckpt_index.max() + 1 == n_ck and len(d) == n_ck * len(s.trials), f"{sub} {m}: table does not cover chain x trials"
            shape = (len(s.trials), n_ck)
            self.ok[m], self.margin[m] = np.full(shape, np.nan), np.full(shape, np.nan)
            self.insample[m], self.on_chain[m] = np.zeros(shape, bool), np.zeros(shape, bool)
            i, k = d.trial_seq.to_numpy(), d.ckpt_index.to_numpy()
            self.ok[m][i, k] = d.vote_ok.to_numpy()
            self.margin[m][i, k] = d.margin.to_numpy()
            self.insample[m][i, k] = d.in_sample.to_numpy()
            self.on_chain[m][i, k] = d.on_chain.to_numpy()


# ---------------- per finetune ----------------
def window_stats(ok_p, ok_c, mg_p, mg_c):
    """n, parent/child accuracy, b (only parent right), c (only child right), mean margin diff over trials where both votes exist."""
    v = ~np.isnan(ok_p) & ~np.isnan(ok_c)
    n = int(v.sum())
    if n == 0:
        return {"n": 0, "sum_p": 0.0, "sum_c": 0.0, "acc_p": np.nan, "acc_c": np.nan, "b": 0, "c": 0, "mdiff": np.nan}
    p, c = ok_p[v] == 1, ok_c[v] == 1
    md = mg_c[v] - mg_p[v]
    return {"n": n, "sum_p": float(p.sum()), "sum_c": float(c.sum()), "acc_p": p.mean(), "acc_c": c.mean(), "b": int((p & ~c).sum()),
            "c": int((~p & c).sum()), "mdiff": float(np.nanmean(md)) if np.isfinite(md).any() else np.nan}


def analyse_finetune(sd, ft):
    s, m = sd.sess, ft["model"]
    pi, ci = ft["parent_index"], ft["child_index"]
    chain = s.chain[m]
    nxt = chain[ci + 1]["time"] if ci + 1 < len(chain) else None
    w1, w2 = F.windows(s.ats, ft["swap_time"], nxt)
    tag = F.live_tag(s.live, m)
    rec = dict(ft)
    rec["session"] = sd.sub
    rec["next_swap_time"] = np.nan if nxt is None else nxt
    first = int(np.flatnonzero(s.ats > ft["swap_time"])[0]) if (s.ats > ft["swap_time"]).any() else -1
    rec["swap_seq"] = int((s.ats <= ft["swap_time"]).sum())
    rec["run_at_swap"] = s.trials[first]["run"] if first >= 0 else s.trials[-1]["run"]
    for wname, idx in (("w1", w1), ("w2", w2)):
        for sub_name, sel in (("all", np.ones(len(idx), bool)), ("live", tag[idx]), ("notlive", ~tag[idx])):
            j = idx[sel]
            st = window_stats(sd.ok[m][j, pi], sd.ok[m][j, ci], sd.margin[m][j, pi], sd.margin[m][j, ci])
            for k_, v in st.items():
                rec[f"{wname}_{sub_name}_{k_}"] = v
    rec["w2_live_frac"] = float(tag[w2].mean()) if len(w2) else np.nan
    pre = F.pre_trials(s.ats, ft["swap_time"], chain[pi]["time"])
    v = sd.ok[m][pre, pi]
    v = v[~np.isnan(v)]
    rec["n_pre"] = len(v)
    rec["pre_acc"] = float(v.mean()) if len(v) else np.nan
    return rec


# ---------------- statistics ----------------
def sign_test(b, c):
    return 1.0 if b + c == 0 else float(stats.binomtest(int(c), int(b + c), 0.5).pvalue)


def pooled(recs, w, s):
    n = sum(r[f"{w}_{s}_n"] for r in recs)
    sp = sum(r[f"{w}_{s}_sum_p"] for r in recs)
    sc = sum(r[f"{w}_{s}_sum_c"] for r in recs)
    b = sum(r[f"{w}_{s}_b"] for r in recs)
    c = sum(r[f"{w}_{s}_c"] for r in recs)
    return {"n": n, "acc_p": sp / n if n else np.nan, "acc_c": sc / n if n else np.nan, "delta": (sc - sp) / n if n else np.nan, "b": b, "c": c,
            "p": sign_test(b, c)}


def boot_ci(recs, w, s, n_boot, rng):
    """95% percentile CI of the trial-weighted pooled change child - parent: resample sessions, then finetunes within each session."""
    by = {}
    for r in recs:
        by.setdefault(r["session"], []).append([r[f"{w}_{s}_n"], r[f"{w}_{s}_sum_p"], r[f"{w}_{s}_sum_c"]])
    clusters = [np.array(v, float) for v in by.values()]
    if not clusters:
        return (np.nan, np.nan)
    out = []
    for _ in range(n_boot):
        tot = np.zeros(3)
        for ci in rng.integers(0, len(clusters), len(clusters)):
            cl = clusters[ci]
            tot += cl[rng.integers(0, len(cl), len(cl))].sum(0)
        if tot[0] > 0:
            out.append((tot[2] - tot[1]) / tot[0])
    return tuple(np.percentile(out, [2.5, 97.5])) if len(out) > 20 else (np.nan, np.nan)


def boot_ci_finetunes(recs, w, s, n_boot, rng):
    """Sensitivity: resample finetunes only (ignores that finetunes of one session share a subject and day)."""
    a = np.array([[r[f"{w}_{s}_n"], r[f"{w}_{s}_sum_p"], r[f"{w}_{s}_sum_c"]] for r in recs], float)
    out = []
    for _ in range(n_boot):
        t = a[rng.integers(0, len(a), len(a))].sum(0)
        if t[0] > 0:
            out.append((t[2] - t[1]) / t[0])
    return tuple(np.percentile(out, [2.5, 97.5])) if len(out) > 20 else (np.nan, np.nan)


def share_w2(recs, s, min_n=1):
    """(worse, equal, better, total) finetunes by child - parent accuracy on W2 (subset s) among finetunes with at least min_n trials."""
    d = [r[f"w2_{s}_acc_c"] - r[f"w2_{s}_acc_p"] for r in recs if r[f"w2_{s}_n"] >= min_n]
    d = np.array(d)
    return int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum()), len(d)


# ---------------- report ----------------
def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report(df, ft_df, sdata, n_boot, seed):
    rng = np.random.default_rng(seed)
    recs = ft_df.to_dict("records")
    L = []
    say = L.append
    say("Finetune evaluation (F1 chain experiment + overview numbers). Unity vote recomputed offline; sources: trial_checkpoint.csv.gz, ab_blocks.csv of each run")
    say(f"sessions in the table: {', '.join(sdata)}; trials {sum(len(sd.sess.trials) for sd in sdata.values())}; bootstrap {n_boot} resamples, seed {seed}")
    say("")
    say("== Finetunes per session (ab_blocks rows with ft_X = 1) vs the expected counts ==")
    for sub, sd in sdata.items():
        nA, nB = len(sd.sess.chain["A"]) - 1, len(sd.sess.chain["B"]) - 1
        say(f"  {sub}: A {nA} B {nB}  (expected {EXPECTED_FT[sub][0]}/{EXPECTED_FT[sub][1]}: {'ok' if (nA, nB) == EXPECTED_FT[sub] else 'MISMATCH'}); trials {len(sd.sess.trials)}, "
            f"live A {int((sd.sess.live == 'A').sum())} B {int((sd.sess.live == 'B').sum())}")
    say("")
    say("Definitions: W1 = trials with swap_time < AT <= next swap of the same model (or session end); W2 = first 48 trials after the swap (may span runs, fewer at session end).")
    say("  live = the finetuned model was LIVE= on the trial; notlive = the other model was live. b = only parent right, c = only child right (trials where both votes exist).")
    say("  delta = child - parent accuracy (pooled over trials). p = exact two-sided sign test on b vs c (trials in a session are correlated: p is indicative).")
    say("")

    for w, title in (("w1", "W1 (primary)"), ("w2", "W2 (first 48 trials after the swap; windows overlap between finetunes)")):
        say(f"== Per session and model, {title} ==")
        say("session model subset   n_ft  n_trials  parent_acc  child_acc   delta    b    c   p_sign   mean_margin_diff")
        for sub in sdata:
            for m in "AB":
                rr = [r for r in recs if r["session"] == sub and r["model"] == m]
                if not rr:
                    say(f"{sub}  {m}     (no finetunes)")
                    continue
                for s in SUBSETS:
                    p = pooled(rr, w, s)
                    md = np.nanmean([r[f"{w}_{s}_mdiff"] for r in rr if r[f"{w}_{s}_n"] > 0]) if p["n"] else np.nan
                    say(f"{sub}  {m}     {s:7s} {len(rr):4d}  {p['n']:7d}   {p['acc_p']:.3f}       {p['acc_c']:.3f}      {p['delta']:+.3f}   {p['b']:4d} {p['c']:4d}  {p['p']:.3f}    {md:+.3f}")
        say("")

    say("== Pooled (all sessions) ==")
    say("CI 'S>F' = bootstrap resampling sessions then finetunes (primary); 'F' = finetunes only (sensitivity). Both trial-weighted. With 4-5 sessions the S>F interval is itself unstable.")
    say("W2 windows overlap, so the W2 intervals are too narrow (descriptive).")
    groups = [("A", [r for r in recs if r["model"] == "A"]), ("B", [r for r in recs if r["model"] == "B"]), ("A+B", recs)]
    pooled_rows = []
    for w in WINDOWS:
        say("")
        say(f"-- {w.upper()} --")
        say("model subset   n_ft  n_trials parent_acc child_acc  delta    CI S>F               CI F                 b     c    p_sign")
        for g, rr in groups:
            for s in SUBSETS:
                p = pooled(rr, w, s)
                nft = sum(1 for r in rr if r[f"{w}_{s}_n"] > 0)
                ci1 = boot_ci(rr, w, s, n_boot, rng)
                ci2 = boot_ci_finetunes(rr, w, s, n_boot, rng)
                say(f"{g:4s}  {s:7s} {nft:4d}  {p['n']:7d}  {p['acc_p']:.3f}     {p['acc_c']:.3f}    {p['delta']:+.3f}  {fmt_ci(ci1)}  {fmt_ci(ci2)}  {p['b']:4d} {p['c']:4d}  {p['p']:.4f}")
                pooled_rows.append({"model": g, "window": w, "subset": s, "n_ft": nft, **p, "ci_lo": ci1[0], "ci_hi": ci1[1], "ci_f_lo": ci2[0], "ci_f_hi": ci2[1]})
    say("")
    say("== Share of finetunes by child - parent accuracy on W2 ==")
    say("model subset   min_n  finetunes  worse  equal  better   share_worse  share_better")
    for g, rr in groups:
        for s in SUBSETS:
            for mn in (1, 24):
                wq, eq, be, tot = share_w2(rr, s, mn)
                if tot:
                    say(f"{g:4s}  {s:7s} {mn:5d}  {tot:8d}  {wq:5d}  {eq:5d}  {be:6d}    {wq / tot:.3f}        {be / tot:.3f}")
    say("")
    say(f"== Effect vs the parent's accuracy on the last {F.PRE_TRIALS} trials before the swap (out-of-sample for the parent; finetunes with >= {MIN_PRE} such trials) ==")
    say("model window  n_ft  spearman_rho  p      slope(delta per +1.0 pre_acc)   mean delta if pre_acc<=0.5 (n)   mean delta if pre_acc>0.5 (n)")
    for g, rr in groups:
        for w in WINDOWS:
            x = np.array([r["pre_acc"] for r in rr if r["n_pre"] >= MIN_PRE and r[f"{w}_all_n"] > 0])
            y = np.array([r[f"{w}_all_acc_c"] - r[f"{w}_all_acc_p"] for r in rr if r["n_pre"] >= MIN_PRE and r[f"{w}_all_n"] > 0])
            if len(x) < 5:
                continue
            rho, p = stats.spearmanr(x, y)
            sl = stats.linregress(x, y).slope
            lo, hi = y[x <= 0.5], y[x > 0.5]
            say(f"{g:4s}  {w}     {len(x):4d}  {rho:+.3f}        {p:.3f}  {sl:+.3f}                        "
                f"{(np.mean(lo) if len(lo) else np.nan):+.3f} ({len(lo)})                    {(np.mean(hi) if len(hi) else np.nan):+.3f} ({len(hi)})")
    say("")
    say("== Overview (descriptive): each session and model, accuracy of the live chain vs frozen checkpoints ==")
    say("chain = the checkpoint live at each trial (single-checkpoint vote, table 'on_chain'); c000 = base; c001 = first finetuned checkpoint, trials after it went live only")
    say("session model  n_ckpt  trials  chain_acc  c000_acc | trials_after_c001  chain_acc  c001_acc  c000_acc | logged_vote_acc")
    for sub, sd in sdata.items():
        for m in "AB":
            ok, oc = sd.ok[m], sd.on_chain[m]
            chain_ok = np.where(oc, ok, np.nan)
            chain_ok = np.nanmean(chain_ok, axis=1)
            nck = ok.shape[1]
            logged = df[(df.session == sub) & (df.model == m) & (df.ckpt_index == 0)].sort_values("trial_seq")
            lv = (logged.logged_vote.to_numpy() == logged.label.to_numpy()).mean()
            if nck > 1:
                after = sd.sess.ats > sd.sess.chain[m][1]["time"]
                say(f"{sub}  {m}     {nck:4d}   {len(chain_ok):4d}    {np.nanmean(chain_ok):.3f}      {np.nanmean(ok[:, 0]):.3f}   | {int(after.sum()):6d}            "
                    f"{np.nanmean(chain_ok[after]):.3f}     {np.nanmean(ok[after, 1]):.3f}     {np.nanmean(ok[after, 0]):.3f}   | {lv:.3f}")
            else:
                say(f"{sub}  {m}     {nck:4d}   {len(chain_ok):4d}    {np.nanmean(chain_ok):.3f}      {np.nanmean(ok[:, 0]):.3f}   | (no finetuned checkpoint)                         | {lv:.3f}")
    say("")
    say("== Per-finetune list: see finetune_results.csv ==")
    return L, pooled_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(F.OUT_DIR, "trial_checkpoint.csv.gz"))
    ap.add_argument("--out", default=F.OUT_DIR)
    ap.add_argument("--name", default="")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.table)
    subs = [s for s in F.SESSIONS if s in set(df.session)]
    sdata = {sub: SessionData(sub, df) for sub in subs}
    rows = [analyse_finetune(sd, ft) for sd in sdata.values() for ft in sd.sess.finetunes]
    ft_df = pd.DataFrame(rows)
    ft_df.to_csv(os.path.join(args.out, f"finetune_results{args.name}.csv"), index=False)
    lines, pooled_rows = report(df, ft_df, sdata, args.n_boot, args.seed)
    pd.DataFrame(pooled_rows).to_csv(os.path.join(args.out, f"f1_pooled{args.name}.csv"), index=False)
    with open(os.path.join(args.out, f"summary{args.name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
