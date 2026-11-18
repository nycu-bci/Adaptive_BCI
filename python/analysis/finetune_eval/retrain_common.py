# Shared helpers for step 0b and F1b (imported; no run command of its own)
"""
Offline retraining of the online finetunes (docs/plan_finetune_evaluation.md, "Later: F1b and F2").

Rebuilds each model's replay buffer block by block from the data_*.pt files (every block is added, trained or not), the per-model failure
flags from the logged votes, retrains a saved finetune from its saved parent with the online trainer (OnlineCalibrationTrainer, config
settings), and scores children with the offline Unity vote on the windows cached by trial_table.run_windows.

Variants (F1b): v0 = online buffer and validation unchanged; v1 = whole held-out trials as validation, the parent may be kept;
v2 = candidate trained on the buffer up to block k-1, accepted only if it beats the parent on block k.
The trainer writes per-epoch files to config.EEG_CHECKPOINT_TMP_BASE_FILE: set_tmp_ckpt_dir() redirects it to %TEMP%.
"""
import collections
import contextlib
import copy
import io
import os
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_common as F  # noqa: E402
import trial_table as T  # noqa: E402
C = F.C
from main.EEG.EEG_Train import prepare_calibration_dataset  # noqa: E402
from main.EEG.MI_train import OnlineCalibrationTrainer, ReplayBuffer  # noqa: E402

CACHE = os.path.join(tempfile.gettempdir(), "f1b_cache")
N_SEEDS = 5
GLOBAL_SEED = 42
GEN_SEED = {"A": 42, "B": 43}
SEED_STEP = 1000
FAIL_WEIGHT = 2.0
CAL_MIN_TRIALS = 9
HOLD_PER_CLASS = 6
VARIANTS = ("v0", "v1", "v2")


def set_tmp_ckpt_dir(tag):
    d = tempfile.mkdtemp(prefix=f"{tag}_ckpt_")
    C.config.EEG_CHECKPOINT_TMP_BASE_FILE = d + os.sep
    return d


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_values(model, s):
    """(global seed, generator seed) of retraining seed s: s=0 is online-like (42; 42 for A, 43 for B)."""
    return GLOBAL_SEED + SEED_STEP * s, GEN_SEED[model] + SEED_STEP * s


# ---------------- blocks and data ----------------
class Block:
    """One block of a run: ab_blocks row + its data_*.pt (original, as ran) + the matching log trials."""

    def __init__(self, run, bi, row, i0):
        self.run, self.bi, self.row, self.number = run, bi, row, int(row["block"])
        self.path = run.data_files[bi]
        self.n_expected = row["n_trials"]
        d = torch.load(self.path, weights_only=False)
        self.ids = np.asarray(d["trial_ids"]).astype(int)
        self.y = np.asarray(d["y_data"]).astype(int)
        self.stored = np.asarray(d["failures"]).astype(bool)
        self.n = len(self.ids)
        self._x = d["x_data"]
        block_trials = run.trials[i0:i0 + row["n_trials"]]
        by_id = {}
        for t in block_trials:
            if t["idx"] in by_id:
                raise ValueError(f"{run.name} block {self.number}: duplicate trial idx {t['idx']}")
            by_id[t["idx"]] = t
        missing = [i for i in self.ids if i not in by_id]
        if missing or len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{run.name} block {self.number}: data trial_ids {list(self.ids)} do not match the log trials {list(by_id)}")
        self.trials = [by_id[i] for i in self.ids]
        if [t["label"] for t in self.trials] != list(self.y):
            raise ValueError(f"{run.name} block {self.number}: data labels differ from the log")
        self.missing_trials = sorted(set(by_id) - set(self.ids))

    @property
    def x(self):
        return self._x

    def ft(self, model):
        return str(self.row[f"ft_{model}"]) == "1"


def load_blocks(sess):
    """Blocks of the session in order (runs with ab_blocks.csv), each matched to its data_*.pt by order. Raises on a count mismatch."""
    out = []
    for run in sess.runs:
        if run.test_only:
            continue
        if len(run.data_files) != len(run.blocks):
            raise ValueError(f"{sess.sub} {run.name}: {len(run.data_files)} data files vs {len(run.blocks)} ab_blocks rows")
        i0 = 0
        for bi, row in enumerate(run.blocks):
            out.append(Block(run, bi, row, i0))
            i0 += row["n_trials"]
    return out


def model_failures(blk, model):
    """Failure flag per row of the block as EEGFineTunePipeline._failures_per_model: the live model uses the stored flags (Unity's result),
    the other model's flag = its logged END-line vote is wrong (NA vote: not failed). Rows without a logged vote keep the stored flag."""
    out = blk.stored.copy()
    n_nolog = 0
    for r, t in enumerate(blk.trials):
        if t["logged"] is None or t["live"] is None:
            n_nolog += 1
            continue
        if t["live"] != model:
            v = t["logged"][model]
            out[r] = (v is not None and v != t["label"])
    return out, n_nolog


def window_ds(x, y, f, positions):
    """TensorDataset (X (N,1,C,500), onehot, failure) exactly as _ModelSlot.finetune builds it."""
    return prepare_calibration_dataset([x[:, positions, :]], [y], [f])


def weighted(ds, w=FAIL_WEIGHT):
    """(x, y, failure flag) -> (x, y, weight) as ReplayBuffer stores it."""
    X, Y, f = ds.tensors
    return TensorDataset(X, Y, torch.where(f > 0.5, torch.tensor(w), torch.tensor(1.0)))


def n_train(buf):
    return sum(len(v) for v in buf.train.values())


# ---------------- V1 buffer ----------------
def holdout_trials(y, ids, normal_counter):
    """Positions (in the block) of the trials held out for validation.
    Calibration block (>= CAL_MIN_TRIALS trials): HOLD_PER_CLASS per class, evenly spread over the class's trials (by trial id).
    Normal block: one trial, class alternating with normal_counter (0 -> class 0), the class's latest trial; if the block has no trial of that
    class the other class is used. Returns (positions, fallback flag)."""
    y, ids = np.asarray(y), np.asarray(ids)
    if len(y) >= CAL_MIN_TRIALS:
        held = []
        for c in (0, 1):
            pos = np.flatnonzero(y == c)
            pos = pos[np.argsort(ids[pos])]
            if len(pos) > HOLD_PER_CLASS:
                pos = pos[np.unique(np.round(np.linspace(0, len(pos) - 1, HOLD_PER_CLASS)).astype(int))]
            held += list(pos)
        return sorted(held), False
    want = int(normal_counter % 2)
    pos = np.flatnonzero(y == want)
    fallback = False
    if len(pos) == 0:
        pos, fallback = np.flatnonzero(y == 1 - want), True
    if len(pos) == 0:
        return [], True
    return [int(pos[np.argmax(ids[pos])])], fallback


class HoldoutBuffer:
    """V1: train = per-class FIFO of the windows of non-held-out trials (evicted windows dropped); val = the last HOLD_PER_CLASS held-out
    trials per class (all their windows). The held-out trials are never trained on."""

    def __init__(self, limit=None, fail_weight=FAIL_WEIGHT):
        self.train = ReplayBuffer(limit=limit, use_val=False, fail_weight=fail_weight)
        self.val = {0: collections.deque(maxlen=HOLD_PER_CLASS), 1: collections.deque(maxlen=HOLD_PER_CLASS)}
        self.fail_weight = fail_weight
        self.normal_counter = 0
        self.fallbacks = 0
        self.last_held = []

    def add_block(self, ds, y, ids):
        X, Y, f = ds.tensors
        n = len(y)
        per = len(X) // n
        held, fb = holdout_trials(y, ids, self.normal_counter)
        if n < CAL_MIN_TRIALS:
            self.normal_counter += 1
        self.fallbacks += int(fb)
        self.last_held = list(held)
        for j in held:
            rows = range(j * per, (j + 1) * per)
            items = [(X[r].cpu(), Y[r].cpu(), self.fail_weight if f[r].item() > 0.5 else 1.0) for r in rows]
            self.val[int(y[j])].append(items)
        held_set = set(held)
        keep = np.array([r for j in range(n) if j not in held_set for r in range(j * per, (j + 1) * per)], int)
        if len(keep):
            self.train.add(TensorDataset(X[keep], Y[keep], f[keep]))

    def train_set(self):
        return self.train.train_set()

    def val_set(self):
        parts = {c: [it for trial in self.val[c] for it in trial] for c in (0, 1)}
        return ReplayBuffer._to_dataset(parts)


# ---------------- training ----------------
CE_NONE = nn.CrossEntropyLoss(reduction="none")


def val_loss(model, ds, bs=None):
    """Weighted validation loss exactly as OnlineCalibrationTrainer.train: mean over batches (size 8, no shuffle) of the batch mean of weight * CE."""
    bs = bs or C.config.adaption_batch_size
    dev = next(model.parameters()).device
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, lab, w in DataLoader(ds, batch_size=bs, shuffle=False):
            out = model(x.to(dev).float().squeeze(1))
            tot += (CE_NONE(out, torch.argmax(lab.to(dev).float(), 1)) * w.to(dev).float()).mean().item()
            n += 1
    return tot / n


def new_model(n_chans, device=None):
    return C.config.USE_MODEL(**dict(C.config.LOAD_MODEL_PARAM, channels=n_chans)).to(device or C.DEVICE)


def load_state(path):
    ck = torch.load(path, map_location=C.DEVICE, weights_only=False)
    return ck["model_state_dict"], int(ck.get("n_chans", C.config.N_CHANNELS))


def make_gen(seed):
    return torch.Generator().manual_seed(seed)


def advance_gen(gen, n, epochs=None, bs=None):
    """Consume the generator as one training over n windows does (epochs x one shuffled pass), without training."""
    epochs = C.config.adaption_epochs if epochs is None else epochs
    dl = DataLoader(TensorDataset(torch.zeros(n, 1)), batch_size=bs or C.config.adaption_batch_size, shuffle=True, generator=gen)
    for _ in range(epochs):
        for _ in dl:
            pass


def copy_gen(gen):
    g = torch.Generator()
    g.set_state(gen.get_state())
    return g


def train_online(parent_path, n_chans, train_ds, val_ds, global_seed, gen, tag, device=None):
    """The online finetune (_ModelSlot.finetune -> OnlineCalibrationTrainer.train): returns the model with the best-validation epoch loaded."""
    seed_all(global_seed)
    tr = OnlineCalibrationTrainer(model_class=C.config.USE_MODEL, model_kwargs=dict(C.config.LOAD_MODEL_PARAM, channels=n_chans),
                                  batch_size=C.config.adaption_batch_size, num_epochs=C.config.adaption_epochs,
                                  lr=C.config.adaption_learning_rate, device=device, ft=True, tag=tag, seed=0)
    tr.gen = gen
    with contextlib.redirect_stdout(io.StringIO()):
        tr.load_checkpoint(parent_path)
        tr.train(train_ds, val_ds)
    return tr.model


def train_select(model, train_ds, val_ds, gen, include_parent, epochs=None, lr=None, bs=None):
    """Same loop as OnlineCalibrationTrainer.train (Adam default betas, weighted CE, shuffled batches from gen, best-validation epoch kept);
    with include_parent the parent weights (epoch 0) are a candidate too and win ties. `model` holds the parent weights and is left with the
    selected ones. Returns {'chosen': 0 = parent, 1..epochs = epoch, 'val': [losses]}."""
    epochs = C.config.adaption_epochs if epochs is None else epochs
    lr = C.config.adaption_learning_rate if lr is None else lr
    bs = bs or C.config.adaption_batch_size
    dev = next(model.parameters()).device
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=gen)
    opt = optim.Adam(model.parameters(), lr=lr)
    best, best_state, chosen, losses = float("inf"), None, None, []
    if include_parent:
        best = val_loss(model, val_ds, bs)
        best_state, chosen = copy.deepcopy(model.state_dict()), 0
        losses.append(best)
    model.train()
    for ep in range(epochs):
        for x, lab, w in loader:
            opt.zero_grad()
            out = model(x.to(dev).float().squeeze(1))
            loss = (CE_NONE(out, torch.argmax(lab.to(dev).float(), 1)) * w.to(dev).float()).mean()
            loss.backward()
            opt.step()
        v = val_loss(model, val_ds, bs)
        model.train()
        losses.append(v)
        if v < best:
            best, best_state, chosen = v, copy.deepcopy(model.state_dict()), ep + 1
    model.load_state_dict(best_state)
    return {"chosen": chosen, "val": losses}


# ---------------- evaluation windows and scoring ----------------
class EvalSet:
    """Preprocessed 100-window vote inputs of every trial of the session for one model (trial_table.run_windows), cached as float32 .npy
    in %TEMP% and memory-mapped, so several worker processes share them."""

    def __init__(self, sess, model, build=False):
        self.sess, self.model = sess, model
        lags = sess.load_lags()
        self.pos = C.config.channel_positions(torch.load(C.BASE_CKPT[model], map_location="cpu", weights_only=False).get("n_chans", C.config.N_CHANNELS))
        os.makedirs(CACHE, exist_ok=True)
        self.runs = {}
        for run in sess.runs:
            key = os.path.join(CACHE, f"win_{sess.sub}_{model}_{run.name}_{lags[run.name]:.3f}")
            if not os.path.exists(key + "_x.npy"):
                if not build:
                    raise FileNotFoundError(key + "_x.npy (run the prep step)")
                x, counts, bad = T.run_windows(sess, run, model, lags[run.name], self.pos)
                np.save(key + "_counts.npy", np.array(counts))
                np.save(key + "_bad.npy", np.array(bad))
                np.save(key + "_x.npy", x.cpu().numpy())
                del x
                torch.cuda.empty_cache()
            counts = np.load(key + "_counts.npy")
            self.runs[run.run] = {"x": np.load(key + "_x.npy", mmap_mode="r"), "counts": counts, "off": np.concatenate([[0], np.cumsum(counts)]),
                                  "bad": np.load(key + "_bad.npy")}
        self.where = {}
        for run in sess.runs:
            for i, t in enumerate(run.trials):
                self.where[t["seq"]] = (run.run, i)

    @torch.no_grad()
    def votes_multi(self, models, seqs, chunk=8192):
        """Votes of several models (eval mode) on the trials `seqs` (indices into sess.trials) with one pass over the cached windows:
        (votes, margins), each (n_models, n_trials) with nan = no vote (fewer than 2 predictions or windows outside the recording)."""
        for mdl in models:
            mdl.eval()
        nm = len(models)
        votes, margins = np.full((nm, len(seqs)), np.nan), np.full((nm, len(seqs)), np.nan)
        buf, owners, n_buf = [], [], 0

        def flush():
            nonlocal buf, owners, n_buf
            if not buf:
                return
            x = torch.from_numpy(np.concatenate(buf)).to(C.DEVICE)
            for mi, mdl in enumerate(models):
                pred = T.predict_all(mdl, x, chunk)
                k = 0
                for j, cnt in owners:
                    p = pred[k:k + cnt]
                    k += cnt
                    v = C.unity_vote(p)
                    votes[mi, j] = np.nan if v is None else v
                    margins[mi, j] = C.vote_margin(p, self.sess.trials[seqs[j]]["label"])
            buf, owners, n_buf = [], [], 0

        for j, s in enumerate(seqs):
            r, i = self.where[s]
            d = self.runs[r]
            if d["bad"][i]:
                continue
            buf.append(np.ascontiguousarray(d["x"][d["off"][i]:d["off"][i + 1]]))
            owners.append((j, int(d["counts"][i])))
            n_buf += int(d["counts"][i])
            if n_buf >= chunk:
                flush()
        flush()
        return votes, margins

    def votes(self, model, seqs, chunk=8192):
        v, m = self.votes_multi([model], seqs, chunk)
        return v[0], m[0]


def trained_in_first_run(sess, ft):
    """The finetune's training block (ab_blocks row with ft_<model> = 1 and this child) belongs to the session's first run."""
    run = sess.runs[0]
    return (not run.test_only) and any(str(b[f"ft_{ft['model']}"]) == "1" and F.ckpt_name(b[f"ckpt_{ft['model']}"]) == ft["child"] for b in run.blocks)


def ft_windows(sess, ft):
    """(W1, W2, LATE) trial indices for a finetune. LATE = the session's last run, for finetunes trained in run 1 of a multi-run session
    (the common late test set of the F1b run-7 comparison), else empty."""
    chain = sess.chain[ft["model"]]
    ci = ft["child_index"]
    nxt = chain[ci + 1]["time"] if ci + 1 < len(chain) else None
    w1, w2 = F.windows(sess.ats, ft["swap_time"], nxt)
    last = sess.runs[-1].run
    late = np.array([t["seq"] for t in sess.trials if t["run"] == last], int) if (trained_in_first_run(sess, ft) and last != sess.runs[0].run) else np.array([], int)
    return w1, w2, late


def eval_seqs(sess, ft):
    w1, w2, late = ft_windows(sess, ft)
    return np.unique(np.concatenate([w1, w2, late]).astype(int))


# ---------------- corrected trials ----------------
def corrected_x(sess, blk, lags, cache=True):
    """The block's trials (same ids, same order as the data file) re-cut from the EEG csv: 2000 samples from START in wall time, run's lag
    (cut3). (n, 32, 2000) float64. Cached in %TEMP%."""
    lag = lags[blk.run.name]
    path = os.path.join(CACHE, f"corr_{sess.sub}_{blk.run.name}_b{blk.number}_{lag:.3f}.npy")
    if cache and os.path.exists(path):
        return np.load(path)
    run = blk.run
    segs = [run.cut3(t, lag) for t in blk.trials]
    if not all(len(s) == C.SEG for s in segs):
        raise ValueError(f"{sess.sub} {run.name} block {blk.number}: short corrected segment")
    x = np.ascontiguousarray(np.transpose(np.stack(segs), (0, 2, 1)))
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.save(path, x)
    return x


def cut_lag0_equals_data(blk):
    """The corrected cut with lag 0 (= the online cut1) reproduces the data file (max abs difference over the samples the data file has;
    a few blocks of sub04 have x_data shorter than 2000 samples: their prefix is compared). inf if trials / channels differ."""
    segs = np.transpose(np.stack([blk.run.cut3(t, 0.0) for t in blk.trials]), (0, 2, 1))
    if segs.shape[:2] != blk.x.shape[:2] or segs.shape[2] < blk.x.shape[2]:
        return float("inf")
    return float(np.max(np.abs(segs[:, :, :blk.x.shape[2]] - blk.x)))


# ---------------- finetune event stream ----------------
class Stream:
    """Walks the blocks of one session for one model, rebuilding the buffers, and yields one event per saved finetune (fired block).

    data = 'asran' (original data_*.pt) or 'corr' (re-cut trials). want = set of variants whose snapshots are built ('v0', 'v1', 'v2'); only
    finetunes with assigned(k) true get snapshots (the others only advance the buffers and the s0 generator).
    Event fields: ft, blk, parent_path, n_chans, flags, v0 (train, val), v1 (train, val, held, fallbacks), v2 (cand_train, cand_val, block_ds) or
    None if there is no earlier block, gen0 (generator state replay for the online-like seed; used for v0 only).
    """

    def __init__(self, sess, model, data="asran", want=("v0",), assigned=lambda k: True):
        self.sess, self.model, self.data, self.want, self.assigned = sess, model, data, set(want), assigned

    def events(self):
        s, m = self.sess, self.model
        blocks = load_blocks(s)
        pos = self.positions()
        fts = {f["child"]: f for f in s.finetunes if f["model"] == m}
        lags = s.load_lags() if self.data == "corr" else None
        buf = ReplayBuffer()
        hold = HoldoutBuffer() if "v1" in self.want else None
        gen = make_gen(GEN_SEED[m])
        n_seen = 0
        for blk in blocks:
            x = blk.x if self.data == "asran" else corrected_x(s, blk, lags)
            flags, n_nolog = model_failures(blk, m)
            ds = window_ds(x, blk.y, flags, pos)
            fired = blk.ft(m)
            ft = fts.get(F.ckpt_name(blk.row[f"ckpt_{m}"])) if fired else None
            if fired and ft is None:
                raise ValueError(f"{s.sub} block {blk.number}: ft_{m}=1 but checkpoint {blk.row[f'ckpt_{m}']} is not in the chain")
            ev = None
            if fired and self.assigned(ft["k"]):
                ev = {"ft": ft, "blk": blk, "flags": flags, "n_nolog": n_nolog, "n_chans": len(pos),
                      "parent_path": F.local_ckpt(s.sub, s.chain[m][ft["parent_index"]]["path"]), "n_prev_blocks": n_seen}
                if "v2" in self.want:
                    ev["v2"] = {"train": buf.train_set(), "val": buf.val_set(), "block": weighted(ds)} if n_seen else None
            buf.add(ds)
            if hold is not None:
                hold.add_block(ds, blk.y, blk.ids)
            if fired:
                if ev is not None:
                    if "v0" in self.want or "v2" in self.want:
                        ev["v0"] = {"train": buf.train_set(), "val": buf.val_set()}
                        ev["gen0"] = copy_gen(gen)
                    if hold is not None:
                        ev["v1"] = {"train": hold.train_set(), "val": hold.val_set(), "held": hold.last_held, "fallbacks": hold.fallbacks}
                advance_gen(gen, n_train(buf))
            n_seen += 1
            if ev is not None:
                yield ev

    def positions(self):
        n = torch.load(C.BASE_CKPT[self.model], map_location="cpu", weights_only=False).get("n_chans", C.config.N_CHANNELS)
        return C.config.channel_positions(int(n))


# ---------------- results files (resumable, one file pair per worker) ----------------
import csv  # noqa: E402
import glob  # noqa: E402

VOTE_COLS = ["session", "model", "k", "variant", "seed", "seq", "vote", "margin"]
META_COLS = ["session", "model", "k", "variant", "seed", "seconds", "n_train", "n_val", "chosen_epoch", "kept_parent", "accepted", "val_parent",
             "val_chosen", "v2_fallback_v0", "n_failed_rows", "n_rows_no_logged_vote", "held_trials", "holdout_fallbacks", "n_prev_blocks"]


class Results:
    def __init__(self, out_dir, name, worker=0):
        self.dir, self.name = out_dir, name
        os.makedirs(out_dir, exist_ok=True)
        self.vpath = os.path.join(out_dir, f"{name}_votes_w{worker}.csv")
        self.mpath = os.path.join(out_dir, f"{name}_meta_w{worker}.csv")
        for p, cols in ((self.vpath, VOTE_COLS), (self.mpath, META_COLS)):
            if not os.path.exists(p):
                with open(p, "w", newline="") as f:
                    csv.writer(f).writerow(cols)

    def done(self):
        """Set of (session, model, k, variant, seed) with a meta row in any worker file (the meta row is written after the votes)."""
        out = set()
        for p in glob.glob(os.path.join(self.dir, f"{self.name}_meta_w*.csv")):
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    out.add((r["session"], r["model"], int(r["k"]), r["variant"], int(r["seed"])))
        return out

    def write(self, key, meta, seqs, votes, margins):
        sub, m, k, variant, seed = key
        with open(self.vpath, "a", newline="") as f:
            w = csv.writer(f)
            for s, v, mg in zip(seqs, votes, margins):
                w.writerow([sub, m, k, variant, seed, int(s), "" if np.isnan(v) else int(v), "" if np.isnan(mg) else f"{mg:.5f}"])
        row = dict(session=sub, model=m, k=k, variant=variant, seed=seed)
        row.update(meta)
        with open(self.mpath, "a", newline="") as f:
            csv.DictWriter(f, META_COLS, restval="").writerow(row)


def read_results(out_dir, name):
    """(votes DataFrame, meta DataFrame) merged over all worker files."""
    import pandas as pd
    v = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out_dir, f"{name}_votes_w*.csv")))]
    m = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out_dir, f"{name}_meta_w*.csv")))]
    votes = pd.concat(v, ignore_index=True).drop_duplicates(["session", "model", "k", "variant", "seed", "seq"], keep="last")
    meta = pd.concat(m, ignore_index=True).drop_duplicates(["session", "model", "k", "variant", "seed"], keep="last")
    return votes, meta
