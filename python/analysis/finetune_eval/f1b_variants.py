# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f1b_variants.py --prep --subs sub01,sub05
# run (resumable, several processes): ... f1b_variants.py --subs sub01,sub05 --worker 0 --nworkers 3 [--variants v0,v1,v2]
"""
F1b: retrain every saved finetune on the H4-correct trials (re-cut from the EEG csv with the run's lag, cut3) under three variants, 5 seeds each.

v0  online buffer and validation unchanged (corrected trials only).
v1  whole held-out trials as validation (1 of 4 per block, 12 for the 48-trial block, rolling 6 per class), train buffer = per-class FIFO of the
    other trials (evicted windows dropped), the parent weights compete with the 4 epochs on validation (so the child can be the parent).
v2  candidate trained like v0 on the buffer up to block k-1, accepted only if its weighted loss on block k's windows beats the parent's.
    No earlier block (first block of the session): v0 is used (flagged v2_fallback_v0).

Each finetune is retrained from its real parent (the chain is not rebuilt). Children are scored with the offline Unity vote on W1 / W2 (and the
session's last run for finetunes in run 1 of a multi-run session). Never writes data_*.pt; corrected arrays are cached in %TEMP%/f1b_cache.
Outputs in results/finetune_eval/f1b/raw/: f1b_votes_w*.csv, f1b_meta_w*.csv; with --prep also f1b_prep.txt (lag-0 check).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_common as R
F, C = R.F, R.C

OUT = os.path.join(F.OUT_DIR, "f1b", "raw")


def prep(subs):
    lines = []
    for sub in subs:
        sess = F.Session(sub, load_eeg=True)
        lags = sess.load_lags()
        blocks = R.load_blocks(sess)
        worst, n_tr, n_miss, n_short = 0.0, 0, 0, 0
        for blk in blocks:
            d = R.cut_lag0_equals_data(blk)
            worst = max(worst, d)
            n_tr += blk.n
            n_miss += len(blk.missing_trials)
            n_short += int(blk.x.shape[2] != C.SEG)
            R.corrected_x(sess, blk, lags)
        msg = (f"{sub}: {len(blocks)} blocks, {n_tr} trials re-cut; corrected cut with lag 0 vs data_*.pt x_data: max abs difference {worst} "
               f"({'EXACT' if worst == 0 else 'NOT EXACT'}); lags {lags}; trials in ab_blocks blocks without a saved data row: {n_miss}; blocks whose x_data has fewer than 2000 samples: {n_short}")
        print(msg, flush=True)
        lines.append(msg)
        for m in "AB":
            R.EvalSet(sess, m, build=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "f1b_prep.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def v2_decide(parent, cand, block_ds):
    """Accept the candidate if its weighted loss on block k beats the parent's (strictly): (accepted, parent loss, candidate loss)."""
    lp, lc = R.val_loss(parent, block_ds), R.val_loss(cand, block_ds)
    return lc < lp, lp, lc


def parent_model(ev):
    state, n = R.load_state(ev["parent_path"])
    m = R.new_model(n)
    m.load_state_dict(state)
    return m.eval()


def run_worker(subs, worker, nworkers, variants, limit=None):
    R.set_tmp_ckpt_dir("f1b")
    res = R.Results(OUT, "f1b", worker)
    done = res.done()
    want = set(variants) | ({"v0"} if "v2" in variants else set())
    t_all, n_run = time.time(), 0
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        for m in "AB":
            if len(sess.chain[m]) < 2:
                continue
            es = R.EvalSet(sess, m)
            keys_of = lambda k: [(sub, m, k, v, s) for v in variants for s in range(R.N_SEEDS)]  # noqa: E731
            st = R.Stream(sess, m, "corr", tuple(want), assigned=lambda k: k % nworkers == worker and any(x not in done for x in keys_of(k)))
            for ev in st.events():
                ft = ev["ft"]
                seqs = R.eval_seqs(sess, ft)
                models, keys, metas = [], [], []
                for s in range(R.N_SEEDS):
                    gs, gg = R.seed_values(m, s)
                    base_meta = {"n_failed_rows": int(ev["flags"].sum()), "n_rows_no_logged_vote": ev["n_nolog"], "n_prev_blocks": ev["n_prev_blocks"]}
                    v0_model = None

                    def get_v0():
                        gen = R.copy_gen(ev["gen0"]) if s == 0 else R.make_gen(gg)
                        return R.train_online(ev["parent_path"], ev["n_chans"], ev["v0"]["train"], ev["v0"]["val"], gs, gen, f"f1b{worker}")

                    for variant in variants:
                        key = (sub, m, ft["k"], variant, s)
                        if key in done:
                            continue
                        t0 = time.time()
                        meta = dict(base_meta)
                        if variant == "v0":
                            model = get_v0()
                            meta.update(n_train=len(ev["v0"]["train"]), n_val=len(ev["v0"]["val"]))
                        elif variant == "v1":
                            R.seed_all(gs)
                            model = parent_model(ev)
                            model.train()
                            info = R.train_select(model, ev["v1"]["train"], ev["v1"]["val"], R.make_gen(gg), include_parent=True)
                            meta.update(n_train=len(ev["v1"]["train"]), n_val=len(ev["v1"]["val"]), chosen_epoch=info["chosen"],
                                        kept_parent=int(info["chosen"] == 0), val_parent=f"{info['val'][0]:.5f}", val_chosen=f"{info['val'][info['chosen']]:.5f}",
                                        held_trials=" ".join(map(str, ev["v1"]["held"])), holdout_fallbacks=ev["v1"]["fallbacks"])
                        else:
                            v2 = ev["v2"]
                            if v2 is None:
                                if v0_model is None:
                                    v0_model = get_v0()
                                model = v0_model
                                meta.update(v2_fallback_v0=1, n_train=len(ev["v0"]["train"]), n_val=len(ev["v0"]["val"]))
                            else:
                                cand = R.train_online(ev["parent_path"], ev["n_chans"], v2["train"], v2["val"], gs, R.make_gen(gg), f"f1b{worker}")
                                par = parent_model(ev)
                                acc, lp, lc = v2_decide(par, cand, v2["block"])
                                model = cand if acc else par
                                meta.update(v2_fallback_v0=0, accepted=int(acc), val_parent=f"{lp:.5f}", val_chosen=f"{lc:.5f}",
                                            n_train=len(v2["train"]), n_val=len(v2["val"]))
                        meta["seconds"] = f"{time.time() - t0:.2f}"
                        models.append(model)
                        keys.append(key)
                        metas.append(meta)
                t0 = time.time()
                v, mg = es.votes_multi(models, seqs)
                for i, key in enumerate(keys):
                    res.write(key, metas[i], seqs, v[i], mg[i])
                    done.add(key)
                n_run += len(keys)
                print(f"[w{worker}] {sub} {m} k={ft['k']} {ft['child']} block {ev['blk'].number}: {len(keys)} children, train {sum(float(x['seconds']) for x in metas):.0f} s, "
                      f"score {len(seqs)} trials {time.time() - t0:.0f} s (elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
                del models
                if limit and n_run >= limit:
                    return
            del es


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub01,sub05")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--variants", default="v0,v1,v2")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many retrainings (testing)")
    args = ap.parse_args(argv)
    subs = args.subs.split(",")
    os.makedirs(OUT, exist_ok=True)
    if args.prep:
        prep(subs)
    else:
        run_worker(subs, args.worker, args.nworkers, args.variants.split(","), args.limit)


if __name__ == "__main__":
    main()
