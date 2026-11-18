# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f2_analyze.py --subs sub01,sub05
# other sessions: ... f2_analyze.py --subs sub02,sub03,sub04,sub05 --name _sub02-05
"""
F2 analysis: how good is the current finetune gate? Inputs: results/finetune_eval/f2/decisions.csv and f2/raw/f2_*.csv (f2_counterfactual.py).

Per decision (block x model): effect = child - parent vote accuracy on the 48 trials after the block, the child's accuracy being the mean over the
5 seeds (trials where the parent and all seeds have a vote). Groups are pooled trial-weighted; CI = bootstrap over decisions (consecutive windows
overlap, so CIs are too narrow; two sessions are not resampled).
A: fired vs skipped (and the difference). B: effect vs the gate inputs (errors in the block, run accuracy) and the parent's accuracy on the 24
trials up to the block. C: fired / skipped x help / hurt (seed-mean effect sign; strict = exact sign test of the seed-majority child vs the parent,
p < 0.05). Live split: the model was LIVE= on the test trial.
online_acc = accuracy on the same 48 trials of the checkpoint actually in use at each trial (the online chain, incl. later finetunes; offline
vote from trial_checkpoint_all.csv.gz, on_chain rows).
Outputs in results/finetune_eval/f2/: f2_per_decision.csv, f2_pooled.csv, summary.txt (with --name X: f2_per_decisionX.csv, ...)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_common as R
import f2_counterfactual as CF
F = R.F

OUT = CF.OUT
N_BOOT = 2000
SUBSETS = ("all", "live", "notlive")
GROUPS = (("A", ("A",)), ("B", ("B",)), ("A+B", ("A", "B")))
SCOPES = (("all blocks", None), ("without the calibration block", 0), ("calibration block only", 1))
RUN_ACC_BINS = [0.0, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0001]
PREV_BINS = [0.0, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0001]


# ---------------- per decision ----------------
def decision_effect(labels, parent, children):
    """labels (n,), parent votes (n,), children votes (n_seeds, n); nan = no vote. Trials with every vote present.
    -> dict n, parent_acc, child_acc (seed mean), effect, effect per seed, b (only parent right), c (only seed-majority child right)."""
    labels, parent, children = np.asarray(labels, float), np.asarray(parent, float), np.atleast_2d(np.asarray(children, float))
    ok = ~np.isnan(parent) & ~np.isnan(children).any(0)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "parent_acc": np.nan, "child_acc": np.nan, "effect": np.nan, "seed_effects": np.full(len(children), np.nan), "b": 0, "c": 0}
    p_ok = parent[ok] == labels[ok]
    c_ok = children[:, ok] == labels[ok][None, :]
    maj = c_ok.mean(0) > 0.5
    seed_eff = c_ok.mean(1) - p_ok.mean()
    return {"n": n, "parent_acc": p_ok.mean(), "child_acc": c_ok.mean(), "effect": c_ok.mean() - p_ok.mean(), "seed_effects": seed_eff,
            "b": int((p_ok & ~maj).sum()), "c": int((~p_ok & maj).sum())}


def sign_p(b, c):
    return 1.0 if b + c == 0 else float(stats.binomtest(int(c), int(b + c), 0.5).pvalue)


def decision_votes(subs):
    """Per decision with test trials and all seeds: (d row, labels, live tag, parent votes (nT,), child votes (seeds, nT), online vote_ok (nT,));
    nan = not scored."""
    dec = pd.read_csv(os.path.join(OUT, "decisions.csv"))
    votes, meta = R.read_results(CF.RAW, "f2")
    chain = pd.read_csv(os.path.join(F.OUT_DIR, "trial_checkpoint_all.csv.gz"))
    chain = chain[chain.on_chain == 1]
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        nT, lab = len(sess.trials), sess.labels.astype(float)
        for m in "AB":
            tag = F.live_tag(sess.live, m)
            vm = votes[(votes.session == sub) & (votes.model == m)]
            ch = chain[(chain.session == sub) & (chain.model == m)]
            online_ok = np.full(nT, np.nan)
            online_ok[ch.trial_seq.to_numpy()] = ch.vote_ok.to_numpy(float)
            for _, d in dec[(dec.session == sub) & (dec.model == m)].iterrows():
                if d.n_test == 0:
                    continue
                g = vm[vm.k == d.d]
                pv = np.full(nT, np.nan)
                gp = g[g.variant == "parent"]
                pv[gp.seq.to_numpy()] = gp.vote.to_numpy()
                gc = g[g.variant == "cf"]
                seeds = sorted(gc.seed.unique())
                if len(seeds) < R.N_SEEDS or gp.empty:
                    continue
                M = np.full((len(seeds), nT), np.nan)
                for si, s in enumerate(seeds):
                    x = gc[gc.seed == s]
                    M[si, x.seq.to_numpy()] = x.vote.to_numpy()
                yield d, lab, tag, pv, M, online_ok


def build(subs):
    recs = []
    for d, lab, tag, pv, M, online_ok in decision_votes(subs):
        test, prev = CF.test_seqs(int(d.last_seq), len(lab)), CF.prev_seqs(int(d.last_seq))
        rec = d.to_dict()
        pp = pv[prev]
        rec["parent_prev_n"] = int((~np.isnan(pp)).sum())
        rec["parent_prev_acc"] = float(np.mean(pp[~np.isnan(pp)] == lab[prev][~np.isnan(pp)])) if rec["parent_prev_n"] else np.nan
        oo = online_ok[test]
        rec["online_n"] = int((~np.isnan(oo)).sum())
        rec["online_acc"] = float(np.nanmean(oo)) if rec["online_n"] else np.nan
        for sname, sel in (("all", np.ones(len(test), bool)), ("live", tag[test]), ("notlive", ~tag[test])):
            j = test[sel]
            e = decision_effect(lab[j], pv[j], M[:, j])
            pre = f"{sname}_"
            rec.update({pre + "n": e["n"], pre + "parent_acc": e["parent_acc"], pre + "child_acc": e["child_acc"], pre + "effect": e["effect"],
                        pre + "b": e["b"], pre + "c": e["c"]})
            for si, v in enumerate(e["seed_effects"]):
                rec[f"{pre}effect_s{si}"] = v
        rec["p_sign"] = sign_p(rec["all_b"], rec["all_c"])
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------------- pooling ----------------
def pooled(eff, n):
    eff, n = np.asarray(eff, float), np.asarray(n, float)
    ok = n > 0
    return float(np.sum(eff[ok] * n[ok]) / np.sum(n[ok])) if ok.any() else np.nan


def boot_ci(eff, n, rng, n_boot=N_BOOT):
    eff, n = np.asarray(eff, float), np.asarray(n, float)
    ok = n > 0
    eff, n = eff[ok], n[ok]
    if len(eff) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(eff), (n_boot, len(eff)))
    bs = (eff[idx] * n[idx]).sum(1) / n[idx].sum(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def boot_diff_ci(e1, n1, e2, n2, rng, n_boot=N_BOOT):
    """CI of pooled(e1) - pooled(e2), each group resampled on its own."""
    a = [np.asarray(x, float) for x in (e1, n1, e2, n2)]
    ok1, ok2 = a[1] > 0, a[3] > 0
    e1, n1, e2, n2 = a[0][ok1], a[1][ok1], a[2][ok2], a[3][ok2]
    if len(e1) < 2 or len(e2) < 2:
        return np.nan, np.nan
    i1, i2 = rng.integers(0, len(e1), (n_boot, len(e1))), rng.integers(0, len(e2), (n_boot, len(e2)))
    d = (e1[i1] * n1[i1]).sum(1) / n1[i1].sum(1) - (e2[i2] * n2[i2]).sum(1) / n2[i2].sum(1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def seed_sd(df, subset):
    cols = [f"{subset}_effect_s{s}" for s in range(R.N_SEEDS)]
    return float(np.std([pooled(df[c], df[f"{subset}_n"]) for c in cols], ddof=1)) if len(df) else np.nan


def select(df, models, scope):
    g = df[df.model.isin(models)]
    return g if scope is None else g[g.is_cal == scope]


def fmt(x, sign=True):
    return "  nan " if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.3f}" if sign else f"{x:.3f}")


# ---------------- report ----------------
def report(df, subs):
    rng = np.random.default_rng(0)
    L, rows = [], []
    L.append("F2: how good is the current finetune gate? Counterfactual V0 finetune at every block (fired or not), H4-correct trials, 5 seeds.")
    L.append(f"gate: >= 1 wrong trial in the block and run accuracy (incl. the block) < {CF.THRESHOLD}; sessions {', '.join(subs)}; "
             f"bootstrap {N_BOOT} over decisions, seed 0")
    L.append("effect = child - parent vote accuracy on the 48 trials after the block (child = mean of 5 seeds); pooled trial-weighted. Consecutive windows "
             "overlap (44 of 48 trials shared): CIs are too narrow. 'seed sd' = SD over seeds of the pooled effect.")
    L.append("")
    L.append("== Decisions ==")
    for sub in subs:
        for m in "AB":
            g = df[(df.session == sub) & (df.model == m)]
            L.append(f"  {sub} {m}: {len(g)} decisions with test trials, fired {int(g.fired.sum())}, skipped {int((g.fired == 0).sum())}; "
                     f"gate recomputed = logged decision in {int((g.gate == g.fired).sum())}/{len(g)}")
    L.append("")
    L.append("== A. Fired vs skipped: pooled effect (child - parent) ==")
    L.append(f"{'model':5} {'scope':32} {'subset':8} {'fired: n_dec n_tr effect  CI':38} {'skipped: n_dec n_tr effect  CI':40} {'fired - skipped  CI':28} seed sd f/s")
    for gname, models in GROUPS:
        for scope_name, scope in SCOPES:
            g = select(df, models, scope)
            for sub in SUBSETS:
                f, s = g[(g.fired == 1) & (g[f"{sub}_n"] > 0)], g[(g.fired == 0) & (g[f"{sub}_n"] > 0)]
                if f.empty and s.empty:
                    continue
                ef, es = pooled(f[f"{sub}_effect"], f[f"{sub}_n"]), pooled(s[f"{sub}_effect"], s[f"{sub}_n"])
                cf, cs = boot_ci(f[f"{sub}_effect"], f[f"{sub}_n"], rng), boot_ci(s[f"{sub}_effect"], s[f"{sub}_n"], rng)
                cd = boot_diff_ci(f[f"{sub}_effect"], f[f"{sub}_n"], s[f"{sub}_effect"], s[f"{sub}_n"], rng)
                rows.append(dict(group=gname, scope=scope_name, subset=sub, n_fired=len(f), n_tr_fired=int(f[f"{sub}_n"].sum()), eff_fired=ef, ci_fired_lo=cf[0],
                                 ci_fired_hi=cf[1], n_skipped=len(s), n_tr_skipped=int(s[f"{sub}_n"].sum()), eff_skipped=es, ci_skipped_lo=cs[0], ci_skipped_hi=cs[1],
                                 diff=ef - es, ci_diff_lo=cd[0], ci_diff_hi=cd[1], seed_sd_fired=seed_sd(f, sub), seed_sd_skipped=seed_sd(s, sub)))
                r = rows[-1]
                L.append(f"{gname:5} {scope_name:32} {sub:8} {len(f):4} {r['n_tr_fired']:5} {fmt(ef)} [{fmt(cf[0])},{fmt(cf[1])}]   "
                         f"{len(s):4} {r['n_tr_skipped']:5} {fmt(es)} [{fmt(cs[0])},{fmt(cs[1])}]   {fmt(ef - es)} [{fmt(cd[0])},{fmt(cd[1])}]   "
                         f"{r['seed_sd_fired']:.3f}/{r['seed_sd_skipped']:.3f}")
    L.append("")
    L.append("== A per session (all blocks, all trials) ==")
    for sub in subs:
        for m in "AB":
            g = df[(df.session == sub) & (df.model == m)]
            f, s = g[g.fired == 1], g[g.fired == 0]
            L.append(f"  {sub} {m}: fired {len(f):3} effect {fmt(pooled(f.all_effect, f.all_n))} | skipped {len(s):3} effect {fmt(pooled(s.all_effect, s.all_n))} | "
                     f"all {len(g):3} effect {fmt(pooled(g.all_effect, g.all_n))}")
    L.append("")
    L.append("== B. Effect vs the gate's inputs (all trials; normal 4-trial blocks only, the calibration block is excluded) ==")
    nb = df[df.is_cal == 0]
    for gname, models in GROUPS:
        g = nb[nb.model.isin(models)]
        L.append(f"-- {gname} ({len(g)} decisions) --")
        L.append("  errors in the block: " + " | ".join(f"{e}: n {int((g.errors == e).sum())} eff {fmt(pooled(g[g.errors == e].all_effect, g[g.errors == e].all_n))}"
                                                        for e in range(5)))
        for col, bins, name in (("run_acc", RUN_ACC_BINS, "run accuracy at the decision"), ("parent_prev_acc", PREV_BINS, "parent accuracy, previous 24 trials")):
            cut = pd.cut(g[col], bins, right=False)
            parts = []
            for iv, h in g.groupby(cut, observed=False):
                parts.append(f"[{iv.left:.2f},{min(iv.right, 1):.2f}{']' if iv.right > 1 else ')'}: n {len(h)} eff {fmt(pooled(h.all_effect, h.all_n))}")
            L.append(f"  {name}: " + " | ".join(parts))
        for col, name in (("errors", "errors"), ("run_acc", "run accuracy"), ("parent_prev_acc", "parent prev-24 accuracy")):
            h = g[[col, "all_effect", "all_n"]].dropna()
            h = h[h.all_n > 0]
            if len(h) < 3:
                continue
            rho, p = stats.spearmanr(h[col], h.all_effect)
            slope = np.polyfit(h[col], h.all_effect, 1)[0]
            idx = rng.integers(0, len(h), (N_BOOT, len(h)))
            x, y = h[col].to_numpy(), h.all_effect.to_numpy()
            bs = [np.polyfit(x[i], y[i], 1)[0] if np.ptp(x[i]) > 0 else np.nan for i in idx]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            L.append(f"  {name:24}: Spearman rho {rho:+.3f} (p {p:.3f}); slope {slope:+.3f} per unit [{lo:+.3f}, {hi:+.3f}]")
        cond = [("errors >= 1", g.errors >= 1), ("errors = 0", g.errors == 0), (f"run acc < {CF.THRESHOLD}", g.run_acc < CF.THRESHOLD),
                (f"run acc >= {CF.THRESHOLD}", g.run_acc >= CF.THRESHOLD)]
        L.append("  each condition of the gate on its own: " + " | ".join(f"{n}: n {int(c.sum())} eff {fmt(pooled(g[c].all_effect, g[c].all_n))}" for n, c in cond))
    L.append("")
    L.append("== C. Fired / skipped x help / hurt (all blocks, all trials) ==")
    L.append("  help = seed-mean effect > 0, hurt < 0, zero = 0; strict = exact sign test (seed-majority child vs parent) p < 0.05")
    for gname, models in GROUPS:
        g = df[df.model.isin(models)]
        for fired, name in ((1, "fired"), (0, "skipped")):
            h = g[g.fired == fired]
            L.append(f"  {gname:3} {name:7}: n {len(h):3} | help {int((h.all_effect > 0).sum()):3} zero {int((h.all_effect == 0).sum()):3} hurt {int((h.all_effect < 0).sum()):3} "
                     f"| strict help {int(((h.p_sign < 0.05) & (h.all_c > h.all_b)).sum()):2} strict hurt {int(((h.p_sign < 0.05) & (h.all_c < h.all_b)).sum()):2}")
        hp = g.all_effect > 0
        L.append(f"  {gname:3} share of 'help' decisions the gate fired on: {int((hp & (g.fired == 1)).sum())}/{int(hp.sum())}; "
                 f"share of fired decisions that help: {int((hp & (g.fired == 1)).sum())}/{int((g.fired == 1).sum())}")
    L.append("")
    L.append("== Noise per decision ==")
    dis = (df.all_b + df.all_c) / df.all_n
    se = np.sqrt(df.all_b + df.all_c) / df.all_n
    L.append(f"  discordant share (seed-majority child vs parent): median {dis.median():.3f}; implied SE of one decision's effect: median {se.median():.3f}; "
             f"mean |effect| {df.all_effect.abs().mean():.3f}; SD of effects over decisions {df.all_effect.std():.3f}")
    return L, pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub01,sub05")
    ap.add_argument("--name", default="", help="suffix of the output files, e.g. _sub02-05")
    args = ap.parse_args(argv)
    subs = args.subs.split(",")
    df = build(subs)
    df.to_csv(os.path.join(OUT, f"f2_per_decision{args.name}.csv"), index=False)
    lines, pool = report(df, subs)
    pool.to_csv(os.path.join(OUT, f"f2_pooled{args.name}.csv"), index=False)
    with open(os.path.join(OUT, f"summary{args.name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
