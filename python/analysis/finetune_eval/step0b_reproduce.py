# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/step0b_reproduce.py --prep --subs sub01,sub05
# time one retraining: ... step0b_reproduce.py --time_one ; run (resumable, several processes): ... step0b_reproduce.py --subs sub01,sub05 --worker 0 --nworkers 3 ; then ... --report --subs sub01,sub05
"""
Step 0b (gate for F1b): does offline retraining reproduce the saved finetunes?

For every saved finetune (a block with ft_<model> = 1) the model's replay buffer is rebuilt from the ORIGINAL data_*.pt (every block added whether
trained or not, FIFO / validation as ReplayBuffer, per-model failure flags from the logged votes), the child is retrained from the saved parent
with N_SEEDS seeds (seed 0 online-like: global seed 42, shuffle generator 42 (A) / 43 (B) replayed through the earlier trainings; seeds 1-4 fresh),
and compared with the saved child by Unity's offline vote on the finetune's W1 and W2 trials.

Gate: median over finetunes of (vote agreement retrained vs saved child) >= median of (seed-vs-seed agreement) - TOL on W2.
Outputs in results/finetune_eval/f1b/step0b/: step0b_votes_w*.csv, step0b_meta_w*.csv (raw), and with --report: step0b_per_finetune.csv, step0b_summary.txt
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_common as R
F, C = R.F, R.C

OUT = os.path.join(F.OUT_DIR, "f1b", "step0b")
TOL = 0.02


def finetune_models(sess):
    return [m for m in "AB" if len(sess.chain[m]) > 1]


def prep(subs):
    for sub in subs:
        sess = F.Session(sub, load_eeg=True)
        sess.load_lags()
        for m in "AB":
            t0 = time.time()
            R.EvalSet(sess, m, build=True)
            print(f"{sub} {m}: eval windows cached ({time.time() - t0:.0f} s)", flush=True)


def saved_votes_check(sess, es, m, seqs, ft):
    """Score the saved child with this pipeline (same windows as the table)."""
    model = C.load_model(F.local_ckpt(sess.sub, sess.chain[m][ft["child_index"]]["path"]))[0]
    return es.votes(model, seqs)


def run_worker(subs, worker, nworkers, only_models=None, limit=None):
    R.set_tmp_ckpt_dir("step0b")
    res = R.Results(OUT, "step0b", worker)
    done = res.done()
    t_all, n_run = time.time(), 0
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        for m in finetune_models(sess):
            if only_models and m not in only_models:
                continue
            es = R.EvalSet(sess, m)
            keys_of = lambda k: [(sub, m, k, "saved", -1)] + [(sub, m, k, "asran", s) for s in range(R.N_SEEDS)]  # noqa: E731
            st = R.Stream(sess, m, "asran", ("v0",), assigned=lambda k: k % nworkers == worker and any(x not in done for x in keys_of(k)))
            for ev in st.events():
                ft = ev["ft"]
                seqs = R.eval_seqs(sess, ft)
                models, keys, metas = [], [], []
                if (sub, m, ft["k"], "saved", -1) not in done:
                    models.append(C.load_model(F.local_ckpt(sub, sess.chain[m][ft["child_index"]]["path"]))[0])
                    keys.append((sub, m, ft["k"], "saved", -1))
                    metas.append({})
                for s in range(R.N_SEEDS):
                    key = (sub, m, ft["k"], "asran", s)
                    if key in done:
                        continue
                    t0 = time.time()
                    gs, gg = R.seed_values(m, s)
                    gen = R.copy_gen(ev["gen0"]) if s == 0 else R.make_gen(gg)
                    models.append(R.train_online(ev["parent_path"], ev["n_chans"], ev["v0"]["train"], ev["v0"]["val"], gs, gen, f"s0b{worker}"))
                    keys.append(key)
                    metas.append({"seconds": f"{time.time() - t0:.2f}", "n_train": len(ev["v0"]["train"]), "n_val": len(ev["v0"]["val"]),
                                  "n_failed_rows": int(ev["flags"].sum()), "n_rows_no_logged_vote": ev["n_nolog"], "n_prev_blocks": ev["n_prev_blocks"]})
                t0 = time.time()
                v, mg = es.votes_multi(models, seqs)
                for i, key in enumerate(keys):
                    res.write(key, metas[i], seqs, v[i], mg[i])
                    done.add(key)
                n_run += len(keys)
                print(f"[w{worker}] {sub} {m} k={ft['k']} {ft['child']} block {ev['blk'].number}: {len(keys)} models, n_train {len(ev['v0']['train'])} "
                      f"train {sum(float(x.get('seconds', 0)) for x in metas):.0f} s, score {len(seqs)} trials {time.time() - t0:.0f} s (elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
                del models
                if limit and n_run >= limit:
                    return
            del es


def time_one():
    R.set_tmp_ckpt_dir("step0b_time")
    sess = F.Session("sub01", load_eeg=False)
    es = R.EvalSet(sess, "A")
    for target in (5, 30):
        st = R.Stream(sess, "A", "asran", ("v0",), assigned=lambda k, t=target: k == t)
        for ev in st.events():
            t0 = time.time()
            gs, gg = R.seed_values("A", 0)
            model = R.train_online(ev["parent_path"], ev["n_chans"], ev["v0"]["train"], ev["v0"]["val"], gs, R.copy_gen(ev["gen0"]), "s0b_time")
            torch.cuda.synchronize()
            t1 = time.time()
            seqs = R.eval_seqs(sess, ev["ft"])
            es.votes(model, seqs)
            print(f"A k={target} block {ev['blk'].number}: train windows {len(ev['v0']['train'])}, val {len(ev['v0']['val'])}: train {t1 - t0:.1f} s, "
                  f"score {len(seqs)} trials {time.time() - t1:.1f} s")


# ---------------- report ----------------
def agree(a, b):
    """Share of equal votes over trials where both exist; nan if none."""
    v = ~np.isnan(a) & ~np.isnan(b)
    return float((a[v] == b[v]).mean()) if v.any() else np.nan


def per_finetune(subs):
    votes, meta = R.read_results(OUT, "step0b")
    tab = pd.read_csv(os.path.join(F.OUT_DIR, "trial_checkpoint.csv.gz"))
    rows, check = [], {"n": 0, "diff": 0, "n_ft": 0}
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        for ft in sess.finetunes:
            vv = votes[(votes.session == sub) & (votes.model == ft["model"]) & (votes.k == ft["k"])]
            if vv.empty:
                continue
            w1, w2, late = R.ft_windows(sess, ft)
            lab = sess.labels
            tm = tab[(tab.session == sub) & (tab.model == ft["model"])]
            vote_tab = {k: tm[tm.ckpt_index == k].sort_values("trial_seq").vote.to_numpy() for k in (ft["parent_index"], ft["child_index"])}
            piv = {}
            for (variant, seed), g in vv.groupby(["variant", "seed"]):
                a = np.full(len(sess.trials), np.nan)
                a[g.seq.to_numpy()] = g.vote.to_numpy()
                piv[(variant, seed)] = a
            if ("saved", -1) in piv:
                sv, tv = piv[("saved", -1)], vote_tab[ft["child_index"]]
                ok = ~np.isnan(sv)
                check["n"] += int(ok.sum())
                check["diff"] += int((sv[ok] != tv[ok]).sum() + np.isnan(tv[ok]).sum())
                check["n_ft"] += 1
            seeds = sorted(s for (v, s) in piv if v == "asran")
            if len(seeds) < R.N_SEEDS:
                continue
            saved, parent = vote_tab[ft["child_index"]], vote_tab[ft["parent_index"]]
            for wname, idx in (("w1", w1), ("w2", w2)):
                if len(idx) == 0:
                    continue
                R_ = [piv[("asran", s)][idx] for s in seeds]
                sv, pv, y = saved[idx], parent[idx], lab[idx].astype(float)
                acc = lambda v: float(np.nanmean(v == y)) if (~np.isnan(v)).any() else np.nan  # noqa: E731
                rows.append(dict(session=sub, model=ft["model"], k=ft["k"], child=ft["child"], window=wname, n=len(idx),
                                 agree_saved=np.nanmean([agree(r, sv) for r in R_]),
                                 agree_seeds=np.nanmean([agree(a, b) for a, b in itertools.combinations(R_, 2)]),
                                 agree_saved_parent=agree(sv, pv), agree_retrained_parent=np.nanmean([agree(r, pv) for r in R_]),
                                 acc_retrained=np.nanmean([acc(r) for r in R_]), acc_saved=acc(sv), acc_parent=acc(pv),
                                 run_at_swap=sess.trials[int(w2[0])]["run"] if len(w2) else np.nan))
    return pd.DataFrame(rows), meta, check


def report(subs, suffix=""):
    df, meta, check = per_finetune(subs)
    df.to_csv(os.path.join(OUT, f"step0b_per_finetune{suffix}.csv"), index=False)
    L = []
    say = L.append
    say("Step 0b: offline retraining vs the saved finetunes (original data_*.pt, as-ran cut, 5 seeds, offline Unity vote)")
    say(f"sessions: {', '.join(subs)}; finetunes with all {R.N_SEEDS} seeds: {df[df.window == 'w2'].shape[0]}; gate tolerance {TOL:.3f}")
    say(f"seeds: s0 = global 42, shuffle generator 42 (A) / 43 (B) replayed through the earlier trainings of that model; s1-s4 = global 42+1000 s, generator +1000 s")
    say(f"check, saved child scored with this pipeline vs the F1 table votes: {check['diff']} differing of {check['n']} votes over {check['n_ft']} finetunes")
    say("agree_saved = share of trials with the same vote, retrained seed vs saved child (mean over the 5 seeds); agree_seeds = mean over the 10 seed pairs;")
    say("agree_saved_parent = saved child vs parent (how much a finetune changes votes at all)")
    say("")
    res = {}
    for w in ("w2", "w1"):
        d = df[df.window == w]
        say(f"== {w.upper()} ==")
        say("group                n_ft  med agree_saved  med agree_seeds  diff     mean agree_saved  mean agree_seeds  med saved~parent  acc_retrained  acc_saved  acc_parent")
        groups = [("all", d)] + [(f"{s} {m}", d[(d.session == s) & (d.model == m)]) for s in subs for m in "AB"] + [("A", d[d.model == "A"]), ("B", d[d.model == "B"])]
        for name, g in groups:
            if g.empty:
                continue
            ds_ = g.agree_saved.median() - g.agree_seeds.median()
            say(f"{name:18s} {len(g):4d}   {g.agree_saved.median():.3f}           {g.agree_seeds.median():.3f}          {ds_:+.3f}   {g.agree_saved.mean():.3f}             {g.agree_seeds.mean():.3f}"
                f"             {g.agree_saved_parent.median():.3f}             {g.acc_retrained.mean():.3f}         {g.acc_saved.mean():.3f}      {g.acc_parent.mean():.3f}")
            if name == "all":
                res[w] = ds_
        say("")
    say(f"GATE (W2, median agree_saved >= median agree_seeds - {TOL:.2f}): {'PASS' if res.get('w2', -9) >= -TOL else 'FAIL'}  (difference {res.get('w2', float('nan')):+.3f})")
    say(f"W1 (information): difference {res.get('w1', float('nan')):+.3f} -> {'within' if res.get('w1', -9) >= -TOL else 'outside'} the tolerance")
    d2 = df[df.window == "w2"]
    from scipy import stats
    for w in ("w2", "w1"):
        dw = df[df.window == w]
        x, y = (dw.acc_saved - dw.acc_parent).to_numpy(), (dw.acc_retrained - dw.acc_parent).to_numpy()
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() > 3:
            say(f"{w.upper()} effect per finetune, retrained (5-seed mean) vs saved child, accuracy minus parent: Pearson r {stats.pearsonr(x[ok], y[ok])[0]:+.3f}, Spearman rho {stats.spearmanr(x[ok], y[ok])[0]:+.3f} "
                f"(n {int(ok.sum())}); mean effect saved {np.nanmean(x):+.3f}, retrained {np.nanmean(y):+.3f}")
    say(f"finetunes with agree_saved < agree_seeds - {TOL:.2f} (W2): {int((d2.agree_saved < d2.agree_seeds - TOL).sum())} of {len(d2)}")
    say("")
    say("== Training meta (seconds per retraining) ==")
    mm = meta[meta.variant == "asran"]
    say(f"retrainings {len(mm)}; seconds mean {mm.seconds.mean():.1f}, median {mm.seconds.median():.1f}, max {mm.seconds.max():.1f}; "
        f"rows without a logged vote (flag fallback): {int(mm.n_rows_no_logged_vote.sum())}")
    with open(os.path.join(OUT, f"step0b_summary{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return res.get("w2", -9) >= -TOL


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub01,sub05")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--time_one", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--name", default="", help="suffix of the report files (e.g. _all)")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many retrainings (testing)")
    args = ap.parse_args(argv)
    subs = args.subs.split(",")
    os.makedirs(OUT, exist_ok=True)
    if args.prep:
        prep(subs)
    elif args.time_one:
        time_one()
    elif args.report:
        report(subs, args.name)
    else:
        run_worker(subs, args.worker, args.nworkers, limit=args.limit)


if __name__ == "__main__":
    main()
