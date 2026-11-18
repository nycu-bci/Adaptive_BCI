# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f1b_analyze.py --subs sub01,sub05
"""
F1b analysis: does a finetune help when it is trained on the H4-correct trials? Inputs: results/finetune_eval/f1b/raw/f1b_*.csv (v0, v1, v2 children),
f1b/step0b/step0b_*.csv (as-ran retrains), trial_checkpoint.csv.gz (parent and saved child votes of F1).

For every finetune and variant the votes of the 5 seed children on the finetune's windows are averaged first (per finetune: accuracy of the seed
mean); then pooled over finetunes, trial-weighted, as in f1_analyze: delta = child' - parent accuracy.
Windows: W1 (served until the next swap), W2 (next 48 trials), LATE (the session's last run, for finetunes of run 1 in a multi-run session).
Subsets: all, live (the finetuned model was LIVE= on the trial), notlive.
CI: bootstrap over finetunes (primary for the main set: two sessions cannot be resampled); with --subs of 3+ sessions also sessions -> finetunes.
Outputs in results/finetune_eval/f1b/: f1b_per_finetune.csv, f1b_pooled.csv, f1b_early.csv, summary.txt (or summary_<name>.txt with --name)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_common as R
import f1_analyze as A1
F = R.F

OUT = os.path.join(F.OUT_DIR, "f1b")
RAW = os.path.join(OUT, "raw")
S0B = os.path.join(OUT, "step0b")
WINS = ("w1", "w2", "late")
SUBSETS = ("all", "live", "notlive")
VARS = ("saved", "asran", "v0", "v1", "v2")
LABEL = {"saved": "saved child (as ran)", "asran": "as-ran retrain (step 0b)", "v0": "V0 corrected", "v1": "V1 held-out trials + keep parent", "v2": "V2 validate on next block"}


# ---------------- per finetune records ----------------
def vote_matrix(vdf, n_trials, seeds):
    """(n_seeds, n_trials) votes (nan = no vote / not scored) from a votes DataFrame of one finetune and variant."""
    M = np.full((len(seeds), n_trials), np.nan)
    for si, s in enumerate(seeds):
        g = vdf[vdf.seed == s]
        M[si, g.seq.to_numpy()] = g.vote.to_numpy()
    return M


def build_records(subs):
    tab = pd.read_csv(os.path.join(F.OUT_DIR, "trial_checkpoint.csv.gz"))
    votes, meta = R.read_results(RAW, "f1b")
    s_votes, s_meta = R.read_results(S0B, "step0b") if os.path.isdir(S0B) else (None, None)
    recs = []
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        nT, lab = len(sess.trials), sess.labels.astype(float)
        for ft in sess.finetunes:
            m, k = ft["model"], ft["k"]
            tm = tab[(tab.session == sub) & (tab.model == m)]
            pv = tm[tm.ckpt_index == ft["parent_index"]].sort_values("trial_seq").vote.to_numpy()
            sv = tm[tm.ckpt_index == ft["child_index"]].sort_values("trial_seq").vote.to_numpy()
            w1, w2, late = R.ft_windows(sess, ft)
            tag = F.live_tag(sess.live, m)
            run_at = sess.trials[int(w2[0])]["run"] if len(w2) else sess.trials[-1]["run"]
            mats = {"saved": sv[None, :]}
            for variant, src in (("v0", votes), ("v1", votes), ("v2", votes), ("asran", s_votes)):
                if src is None:
                    continue
                g = src[(src.session == sub) & (src.model == m) & (src.k == k) & (src.variant == variant)]
                seeds = sorted(g.seed.unique())
                if len(seeds) == R.N_SEEDS:
                    mats[variant] = vote_matrix(g, nT, seeds)
            for variant, M in mats.items():
                rec = dict(session=sub, model=m, k=k, child=ft["child"], variant=variant, run_at_swap=run_at, early=R.trained_in_first_run(sess, ft), swap_seq=int((sess.ats <= ft["swap_time"]).sum()),
                           n_seeds=M.shape[0])
                for wname, idx in zip(WINS, (w1, w2, late)):
                    for sname, sel in (("all", np.ones(len(idx), bool)), ("live", tag[idx]), ("notlive", ~tag[idx])):
                        j = idx[sel]
                        valid = ~np.isnan(pv[j]) & ~np.isnan(M[:, j]).any(0)
                        j = j[valid]
                        n = len(j)
                        p_ok = (pv[j] == lab[j])
                        c_ok = (M[:, j] == lab[j][None, :])
                        pre = f"{wname}_{sname}_"
                        rec[pre + "n"] = n
                        rec[pre + "sum_p"] = float(p_ok.sum())
                        rec[pre + "sum_c"] = float(c_ok.sum(1).mean()) if n else 0.0
                        for si in range(R.N_SEEDS):
                            rec[pre + f"sum_c_s{si}"] = float(c_ok[si].sum()) if (n and si < M.shape[0]) else np.nan
                        rec[pre + "acc_p"] = float(p_ok.mean()) if n else np.nan
                        rec[pre + "acc_c"] = float(c_ok.mean()) if n else np.nan
                recs.append(rec)
    return pd.DataFrame(recs), meta, (s_meta if s_meta is not None else pd.DataFrame())


# ---------------- statistics ----------------
def delta(recs, w, s, a="sum_c", b="sum_p"):
    n = sum(r[f"{w}_{s}_n"] for r in recs)
    return (sum(r[f"{w}_{s}_{a}"] for r in recs) - sum(r[f"{w}_{s}_{b}"] for r in recs)) / n if n else np.nan


def ci_boot(recs, w, s, rng, n_boot, by_session=False, a="sum_c", b="sum_p"):
    """95% percentile CI of the trial-weighted pooled delta; resample finetunes (by_session: sessions first, then finetunes within a session)."""
    rr = [{"session": r["session"], f"{w}_{s}_n": r[f"{w}_{s}_n"], f"{w}_{s}_sum_p": r[f"{w}_{s}_{b}"], f"{w}_{s}_sum_c": r[f"{w}_{s}_{a}"]} for r in recs
          if r[f"{w}_{s}_n"] > 0]
    if len(rr) < 3:
        return (np.nan, np.nan)
    return A1.boot_ci(rr, w, s, n_boot, rng) if by_session else A1.boot_ci_finetunes(rr, w, s, n_boot, rng)


def fmt(x, d=3):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def seed_sd(recs, w, s):
    """SD over the 5 training seeds of the pooled delta (a single seed's child' - parent): the training-noise level of the pooled estimate."""
    d = []
    for si in range(R.N_SEEDS):
        n = sum(r[f"{w}_{s}_n"] for r in recs)
        if not n:
            return np.nan
        d.append((sum(r[f"{w}_{s}_sum_c_s{si}"] for r in recs) - sum(r[f"{w}_{s}_sum_p"] for r in recs)) / n)
    return float(np.std(d, ddof=1))


def to_dicts(df, variant):
    d = df[df.variant == variant]
    return d.to_dict("records")


def pair(df, va, vb):
    """Records of finetunes present in both variants: a = va, b = vb (delta of va - vb)."""
    A, B = {(r["session"], r["model"], r["k"]): r for r in to_dicts(df, va)}, {(r["session"], r["model"], r["k"]): r for r in to_dicts(df, vb)}
    out = []
    for key in A:
        if key in B:
            r = dict(A[key])
            for w in WINS:
                for s in SUBSETS:
                    pre = f"{w}_{s}_"
                    assert A[key][pre + "n"] == B[key][pre + "n"], (key, w, s)
                    r[pre + "sum_p"] = B[key][pre + "sum_c"]
                    r[pre + "acc_p"] = B[key][pre + "acc_c"]
                    for si in range(R.N_SEEDS):
                        r[pre + f"sum_c_s{si}"] = A[key][pre + f"sum_c_s{si}"]
            out.append(r)
    return out


def share(recs, w, s):
    d = np.array([r[f"{w}_{s}_acc_c"] - r[f"{w}_{s}_acc_p"] for r in recs if r[f"{w}_{s}_n"] > 0])
    return (int((d < -1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d > 1e-12).sum()), len(d))


# ---------------- report ----------------
def report(df, meta, s_meta, subs, n_boot, seed, name=""):
    rng = np.random.default_rng(seed)
    multi = len(subs) >= 3
    L = []
    say = L.append
    variants = [v for v in VARS if v in set(df.variant)]
    say("F1b: finetunes retrained on the H4-correct trials (cut3: 2000 samples from START in wall time, run's lag). Unity vote recomputed offline.")
    say(f"sessions: {', '.join(subs)}; variants with results: {', '.join(variants)}; bootstrap {n_boot} resamples, seed {seed}")
    nft = df[df.variant == "v0"].groupby("model").size().to_dict()
    say(f"finetunes with all 5 seeds of v0: {nft}; v1: {df[df.variant == 'v1'].groupby('model').size().to_dict()}; v2: {df[df.variant == 'v2'].groupby('model').size().to_dict()}; "
        f"saved children: {df[df.variant == 'saved'].groupby('model').size().to_dict()}")
    say("delta = child' - parent accuracy (trial-weighted, pooled over finetunes), 5 seeds averaged per finetune first; CI = bootstrap over finetunes"
        + (" [F] and sessions -> finetunes [S>F]" if multi else " [F] (two sessions: not resampled)"))
    say("'seed sd' = SD over the 5 seeds of the pooled delta (training noise of the pooled estimate). W2 windows overlap between finetunes: CIs too narrow. LATE = last run of the session.")
    say("saved = the checkpoint the online system made (H4 cut), asran = our retrain on the same data (step 0b), v0 = same procedure on corrected trials.")
    say("")
    groups = [("A", lambda d: d[d.model == "A"]), ("B", lambda d: d[d.model == "B"]), ("A+B", lambda d: d)]
    prow = []
    for w, title in (("w1", "W1 (primary)"), ("w2", "W2 (next 48 trials)"), ("late", "LATE (last run, finetunes of run 1)")):
        say(f"== Pooled delta vs parent, {title} ==")
        say("model variant                          subset  n_ft n_trials  delta    CI F                  " + ("CI S>F               " if multi else "") + "seed sd   worse/equal/better (finetunes)")
        for g, sel in groups:
            for v in variants:
                recs = sel(df[df.variant == v]).to_dict("records")
                for s in SUBSETS:
                    if w == "late" and s != "all":
                        continue
                    rr = [r for r in recs if r[f"{w}_{s}_n"] > 0]
                    if not rr:
                        continue
                    d = delta(rr, w, s)
                    ci = ci_boot(rr, w, s, rng, n_boot)
                    ci2 = ci_boot(rr, w, s, rng, n_boot, by_session=True) if multi else None
                    sd = seed_sd(rr, w, s) if v != "saved" else np.nan
                    wq, eq, be, tot = share(rr, w, s)
                    say(f"{g:4s}  {v:6s} {LABEL[v][:28]:28s} {s:7s} {len(rr):3d} {sum(r[f'{w}_{s}_n'] for r in rr):7d}  {fmt(d)}  [{fmt(ci[0])},{fmt(ci[1])}]  "
                        + (f"[{fmt(ci2[0])},{fmt(ci2[1])}]  " if multi else "") + f"{sd:.3f}     {wq}/{eq}/{be}")
                    prow.append(dict(model=g, variant=v, window=w, subset=s, n_ft=len(rr), n=sum(r[f"{w}_{s}_n"] for r in rr), delta=d, ci_lo=ci[0], ci_hi=ci[1],
                                     seed_sd=sd, worse=wq, equal=eq, better=be))
        say("")

    say("== Differences between variants (paired by finetune), W2 and W1, subset all ==")
    say("pair            model window  n_ft   delta   CI F")
    for va, vb in (("v1", "v0"), ("v2", "v0"), ("v1", "v2"), ("v0", "saved"), ("v0", "asran"), ("asran", "saved")):
        if va not in variants or vb not in variants:
            continue
        for g, sel in groups:
            for w in ("w1", "w2", "late"):
                pr = sel(pd.DataFrame(pair(df, va, vb))).to_dict("records") if len(pair(df, va, vb)) else []
                pr = [r for r in pr if r[f"{w}_all_n"] > 0]
                if not pr:
                    continue
                ci = ci_boot(pr, w, "all", rng, n_boot)
                say(f"{va} - {vb:6s}   {g:4s}  {w:5s}  {len(pr):3d}   {fmt(delta(pr, w, 'all'))}  [{fmt(ci[0])},{fmt(ci[1])}]"
                    + ("   <- H4 cost per update (corrected minus what was saved/retrained as ran)" if (va, vb) in (("v0", "saved"), ("v0", "asran")) else ""))
    say("")

    say("== Per session and model, W2 / W1 (all trials): delta vs parent per variant ==")
    say("session model window n_ft  " + "  ".join(f"{v:>7s}" for v in variants))
    for sub in subs:
        for m in "AB":
            for w in ("w2", "w1"):
                cells, n_ft = [], 0
                for v in variants:
                    rr = [r for r in df[(df.variant == v) & (df.session == sub) & (df.model == m)].to_dict("records") if r[f"{w}_all_n"] > 0]
                    n_ft = max(n_ft, len(rr))
                    cells.append(f"{fmt(delta(rr, w, 'all')):>7s}" if rr else "      -")
                if n_ft:
                    say(f"{sub}  {m}     {w}     {n_ft:3d}  " + "  ".join(cells))
    say("")

    # rates
    say("== V1 parent kept / V2 acceptance ==")
    if len(meta):
        v1 = meta[meta.variant == "v1"]
        v2 = meta[meta.variant == "v2"]
        for g, sel in (("A", "A"), ("B", "B"), ("A+B", None)):
            a = v1 if sel is None else v1[v1.model == sel]
            b = v2 if sel is None else v2[v2.model == sel]
            if len(a):
                ch = a.chosen_epoch.value_counts().sort_index().to_dict()
                say(f"V1 {g}: {len(a)} children ({a.groupby(['session', 'model', 'k']).ngroups} finetunes): parent kept {int(a.kept_parent.sum())} = {a.kept_parent.mean():.3f}; chosen epoch (0 = parent) counts {ch}; "
                    f"finetunes where the parent was kept in all 5 seeds: {int((a.groupby(['session', 'model', 'k']).kept_parent.sum() == R.N_SEEDS).sum())}; in no seed: {int((a.groupby(['session', 'model', 'k']).kept_parent.sum() == 0).sum())}")
            if len(b):
                real = b[b.v2_fallback_v0 == 0]
                fb = b[b.v2_fallback_v0 == 1]
                say(f"V2 {g}: {len(b)} children; no earlier block (V0 used) {len(fb)} children = {fb.groupby(['session', 'model', 'k']).ngroups} finetunes; with a decision {len(real)}: accepted {int(real.accepted.sum())} = "
                    f"{real.accepted.mean() if len(real) else float('nan'):.3f}; finetunes accepted in all 5 seeds {int((real.groupby(['session', 'model', 'k']).accepted.sum() == R.N_SEEDS).sum())}, in no seed {int((real.groupby(['session', 'model', 'k']).accepted.sum() == 0).sum())}")
        if len(v1):
            say(f"V1 hold-out fallbacks (block had no trial of the wanted class): max count seen {int(v1.holdout_fallbacks.max())}; rows without logged vote in the flags: {int(meta.n_rows_no_logged_vote.fillna(0).sum())}")
    say("")

    # early A finetunes
    say("== Early finetunes (trained on a block of run 1 = the first song(s)), accuracy on W2 / LATE: parent, saved child, as-ran retrain, V0, V1, V2 ==")
    early_rows = []
    for m in "AB":
        for sub in subs:
            base = df[(df.session == sub) & (df.model == m) & (df.early)]
            ks = sorted(base.k.unique())
            if not ks:
                continue
            say(f"-- {sub} model {m}: {len(ks)} early finetunes --")
            say("k  child   after-trial  W2 n  parent  saved  asran  v0     v1     v2   | LATE n parent saved asran v0     v1     v2  | V1 parent kept /5, V2 accepted /5")
            for k in ks:
                cells = {}
                for v in variants:
                    r = base[(base.k == k) & (base.variant == v)]
                    cells[v] = r.iloc[0].to_dict() if len(r) else None
                ref = next(c for c in cells.values() if c)
                f2 = lambda c, w: ("  -   " if c is None or c[f"{w}_all_n"] == 0 else f"{c[f'{w}_all_acc_c']:.3f}")  # noqa: E731
                pw = lambda w: f"{ref[f'{w}_all_acc_p']:.3f}" if ref[f"{w}_all_n"] else "  -  "  # noqa: E731
                mk = meta[(meta.session == sub) & (meta.model == m) & (meta.k == k)] if len(meta) else meta
                k1 = mk[mk.variant == "v1"].kept_parent.sum() if len(mk) and (mk.variant == "v1").any() else float("nan")
                a2 = mk[(mk.variant == "v2") & (mk.v2_fallback_v0 == 0)].accepted.sum() if len(mk) and ((mk.variant == "v2") & (mk.v2_fallback_v0 == 0)).any() else float("nan")
                say(f"{k:2d} {ref['child']:7s} {int(ref['swap_seq']):5d}        {int(ref['w2_all_n']):3d}   {pw('w2')}  " + " ".join(f"{f2(cells.get(v), 'w2'):6s}" for v in VARS)
                    + f" |  {int(ref['late_all_n']):3d}  {pw('late')}  " + " ".join(f"{f2(cells.get(v), 'late'):6s}" for v in VARS) + f" |  {k1:.0f}  {a2:.0f}")
                for v in variants:
                    if cells.get(v):
                        early_rows.append(dict(session=sub, model=m, k=k, variant=v, w2_acc=cells[v]["w2_all_acc_c"], w2_parent=cells[v]["w2_all_acc_p"], w2_n=cells[v]["w2_all_n"],
                                               late_acc=cells[v]["late_all_acc_c"], late_parent=cells[v]["late_all_acc_p"], late_n=cells[v]["late_all_n"]))
            say("")
    pd.DataFrame(early_rows).to_csv(os.path.join(OUT, f"f1b_early{name}.csv"), index=False)
    say("== Early finetunes pooled (finetunes of run 1), delta vs parent ==")
    say("model variant  window n_ft  n_trials   delta    CI F")
    for g, sel in groups[:2]:
        for v in variants:
            recs = [r for r in sel(df[(df.variant == v) & (df.early)]).to_dict("records")]
            for w in ("w2", "w1", "late"):
                rr = [r for r in recs if r[f"{w}_all_n"] > 0]
                if not rr:
                    continue
                ci = ci_boot(rr, w, "all", rng, n_boot)
                say(f"{g:4s}  {v:6s} {w:5s}  {len(rr):3d}  {sum(r[f'{w}_all_n'] for r in rr):7d}   {fmt(delta(rr, w, 'all'))}  [{fmt(ci[0])},{fmt(ci[1])}]")
    say("")
    say("== sub01 A, cA_004 - cA_006 (the finetunes where the as-ran chain fell on run 7): accuracy on run 7 ==")
    for k in (4, 5, 6):
        for sub in subs:
            sel = df[(df.session == sub) & (df.model == "A") & (df.k == k)]
            if sel.empty or sub != "sub01":
                continue
            row = {v: sel[sel.variant == v].iloc[0] for v in variants if len(sel[sel.variant == v])}
            ref = next(iter(row.values()))
            say(f"k={k} {ref['child']}: parent {ref['late_all_acc_p']:.3f} | " + " | ".join(f"{v} {r['late_all_acc_c']:.3f}" for v, r in row.items()) + f"   (n = {int(ref['late_all_n'])} trials)")
    pd.DataFrame(prow).to_csv(os.path.join(OUT, f"f1b_pooled{name}.csv"), index=False)
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub01,sub05")
    ap.add_argument("--name", default="")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    subs = args.subs.split(",")
    df, meta, s_meta = build_records(subs)
    df.to_csv(os.path.join(OUT, f"f1b_per_finetune{args.name}.csv"), index=False)
    L = report(df, meta, s_meta, subs, args.n_boot, args.seed, args.name)
    with open(os.path.join(OUT, f"summary{args.name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
