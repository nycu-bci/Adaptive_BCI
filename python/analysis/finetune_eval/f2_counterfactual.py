# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/finetune_eval/f2_counterfactual.py --decisions --subs sub01,sub05
# run (resumable, several processes): ... f2_counterfactual.py --subs sub01,sub05 --worker 0 --nworkers 3
"""
F2: how good is the current finetune gate? For every block of every model (fired or not), a counterfactual V0 finetune of the checkpoint the model
had at the end of the block, on its replay buffer after the block rebuilt from the H4-correct trials (F1b corrected copies), 5 seeds.

Parent and children are scored with the offline Unity vote on the TEST_N trials after the block's last trial; the parent also on the PREV_N trials
up to the block's end (its recent accuracy). Each decision is judged with the history that actually happened (the real chain, the real buffer).
Seed 0 replays the online generator as F1b, so a fired block's children equal F1b's V0 children.

Outputs in results/finetune_eval/f2/: decisions.csv (--decisions: one row per decision with the gate inputs; no training), raw/f2_votes_w*.csv,
raw/f2_meta_w*.csv (variant 'cf' = child, seeds 0-4; variant 'parent', seed 0).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrain_common as R
F, C = R.F, R.C

OUT = os.path.join(F.OUT_DIR, "f2")
RAW = os.path.join(OUT, "raw")
TEST_N = 48
PREV_N = 24
THRESHOLD = C.config.FINETUNE_ACC_THRESHOLD
DEC_COLS = ["session", "model", "d", "run", "block", "n_trials", "is_cal", "correct", "errors", "run_correct", "run_scored", "run_acc", "gate",
            "fired", "parent", "parent_index", "last_seq", "n_test", "n_prev"]


def gate(errors, run_correct, run_scored, threshold=THRESHOLD):
    """EEGFineTunePipeline._gates: >= 1 wrong trial in the block and run accuracy (this block included) below the threshold."""
    return errors >= 1 and run_scored > 0 and run_correct / run_scored < threshold


def test_seqs(last_seq, n_total, n=TEST_N):
    return np.arange(last_seq + 1, min(last_seq + 1 + n, n_total), dtype=int)


def prev_seqs(last_seq, n=PREV_N):
    return np.arange(max(0, last_seq - n + 1), last_seq + 1, dtype=int)


def decisions(sess, model):
    """One dict per block of the session for `model` (runs with ab_blocks.csv, in order): the gate inputs from ab_blocks (counters reset per run),
    the logged decision, the parent (chain checkpoint before any finetune of this block) and the block's last trial (log order)."""
    out, d, n_fired = [], 0, 0
    for run in sess.runs:
        if run.test_only:
            continue
        rc = rs = 0
        i0 = 0
        for row in run.blocks:
            n = int(row["n_trials"])
            correct = int(row[f"correct_{model}"])
            rc, rs = rc + correct, rs + n
            fired = str(row[f"ft_{model}"]) == "1"
            last_seq = run.trials[i0 + n - 1]["seq"]
            d += 1
            out.append({"session": sess.sub, "model": model, "d": d, "run": run.run, "block": int(row["block"]), "n_trials": n, "is_cal": int(n > 4),
                        "correct": correct, "errors": n - correct, "run_correct": rc, "run_scored": rs, "run_acc": rc / rs,
                        "gate": int(gate(n - correct, rc, rs)), "fired": int(fired), "parent_index": n_fired,
                        "parent": F.ckpt_name(sess.chain[model][n_fired]["path"]), "last_seq": last_seq,
                        "n_test": len(test_seqs(last_seq, len(sess.trials))), "n_prev": len(prev_seqs(last_seq))})
            n_fired += int(fired)
            i0 += n
    return out


def write_decisions(subs):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        for m in "AB":
            rows += decisions(sess, m)
    with open(os.path.join(OUT, "decisions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, DEC_COLS)
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["gate"] != r["fired"]]
    print(f"{len(rows)} decisions; gate recomputed from ab_blocks differs from the logged ft flag in {len(bad)}; "
          f"with test trials: {sum(r['n_test'] > 0 for r in rows)}", flush=True)
    return rows


def events(sess, model, assigned):
    """Walk the blocks with corrected trials, rebuilding the online buffer; yield (decision, train, val, gen0) for every assigned decision.
    The s0 generator advances only on blocks the online system trained (as online)."""
    decs = decisions(sess, model)
    blocks = R.load_blocks(sess)
    if len(blocks) != len(decs):
        raise ValueError(f"{sess.sub} {model}: {len(blocks)} blocks vs {len(decs)} decisions")
    st = R.Stream(sess, model, "corr")
    pos = st.positions()
    lags = sess.load_lags()
    buf = R.ReplayBuffer()
    gen = R.make_gen(R.GEN_SEED[model])
    for blk, dec in zip(blocks, decs):
        if blk.number != dec["block"] or blk.run.run != dec["run"]:
            raise ValueError(f"{sess.sub} {model}: block order differs at decision {dec['d']}")
        flags, _ = R.model_failures(blk, model)
        buf.add(R.window_ds(R.corrected_x(sess, blk, lags), blk.y, flags, pos))
        if dec["n_test"] > 0 and assigned(dec["d"]):
            yield dec, buf.train_set(), buf.val_set(), R.copy_gen(gen), len(pos), int(flags.sum())
        if dec["fired"]:
            R.advance_gen(gen, R.n_train(buf))


def run_worker(subs, worker, nworkers, limit=None):
    R.set_tmp_ckpt_dir("f2")
    res = R.Results(RAW, "f2", worker)
    done = res.done()
    t_all, n_run = time.time(), 0
    for sub in subs:
        sess = F.Session(sub, load_eeg=False)
        n_total = len(sess.trials)
        for m in "AB":
            es = R.EvalSet(sess, m)
            keys_of = lambda d: [(sub, m, d, "cf", s) for s in range(R.N_SEEDS)] + [(sub, m, d, "parent", 0)]  # noqa: E731
            for dec, train, val, gen0, n_chans, n_failed in events(sess, m, lambda d: d % nworkers == worker and any(k not in done for k in keys_of(d))):
                d = dec["d"]
                parent_path = F.local_ckpt(sub, sess.chain[m][dec["parent_index"]]["path"])
                seqs = test_seqs(dec["last_seq"], n_total)
                models, keys, metas = [], [], []
                for s in range(R.N_SEEDS):
                    key = (sub, m, d, "cf", s)
                    if key in done:
                        continue
                    gs, gg = R.seed_values(m, s)
                    t0 = time.time()
                    gen = R.copy_gen(gen0) if s == 0 else R.make_gen(gg)
                    models.append(R.train_online(parent_path, n_chans, train, val, gs, gen, f"f2{worker}"))
                    keys.append(key)
                    metas.append({"seconds": f"{time.time() - t0:.2f}", "n_train": len(train), "n_val": len(val), "n_failed_rows": n_failed})
                t0 = time.time()
                if models:
                    v, mg = es.votes_multi(models, seqs)
                    for i, key in enumerate(keys):
                        res.write(key, metas[i], seqs, v[i], mg[i])
                        done.add(key)
                pkey = (sub, m, d, "parent", 0)
                if pkey not in done:
                    pseqs = np.concatenate([prev_seqs(dec["last_seq"]), seqs])
                    v, mg = es.votes_multi([load_parent(parent_path)], pseqs)
                    res.write(pkey, {"seconds": "0", "n_train": len(train), "n_val": len(val), "n_failed_rows": n_failed}, pseqs, v[0], mg[0])
                    done.add(pkey)
                n_run += len(keys)
                print(f"[w{worker}] {sub} {m} d={d} run{dec['run']} block {dec['block']} fired={dec['fired']}: {len(keys)} children, "
                      f"train {sum(float(x['seconds']) for x in metas):.0f} s, score {time.time() - t0:.0f} s (elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
                del models
                if limit and n_run >= limit:
                    return
            del es


def load_parent(path):
    state, n = R.load_state(path)
    m = R.new_model(n)
    m.load_state_dict(state)
    return m.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="sub01,sub05")
    ap.add_argument("--decisions", action="store_true", help="write decisions.csv only (no training)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many retrainings (testing)")
    args = ap.parse_args(argv)
    subs = args.subs.split(",")
    if args.decisions:
        write_decisions(subs)
    else:
        os.makedirs(RAW, exist_ok=True)
        run_worker(subs, args.worker, args.nworkers, args.limit)


if __name__ == "__main__":
    main()
