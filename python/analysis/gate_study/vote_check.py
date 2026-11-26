# python python/analysis/gate_study/vote_check.py v0 --out <dir>
# python python/analysis/gate_study/vote_check.py v1 --out <dir> --start_ckpts <starts.json> --configs 70:0.3,50:0.3
# python python/analysis/gate_study/vote_check.py summarize --out <dir>
"""
Does the trial vote matter? (gate study V0 / V1; online_vote.py has the replica of the online vote)

V0  Is the replica of the online vote right? The only online model we still have is the start checkpoint c_000.pth
    (= 13c_s_g_all.pth, braindecode ShallowFBCSPNet, 13 channels); every later c_0xx was overwritten. Each session's
    run 1 trials voted while c_000 was loaded (session log.txt: before the first update_check_point to another file)
    are replayed with c_000 under the replica (grid over prediction rate, vote lead, with / without CAR) and under the
    simulator's 16-window majority, and compared with what was logged (5th CUT = success).
V1  Same start model, two votes: the gate study's start models (4-fold SCCNet, 22 channels) on every trial of the 48
    sessions, voted by the simulator's 16-window majority and by the replica; agreement, accuracy, and which trials
    the SPRT and block gates would fire on (no finetuning: the "never" condition).
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import simulate_gates as sg  # noqa: E402  (installs the braindecode stub)
import online_vote as ov  # noqa: E402
from main.Utils import config  # noqa: E402
from main.Utils.sprt_gate import SprtGate  # noqa: E402

C000 = os.environ.get("VC_CKPT", "/mnt/project/MIEXP/VR_BCI/python/main/EEG/checkpoint_main/c_000.pth")  # VC_CKPT: controls
RATES = [15, 20, 25, 30, 35, 40, 50, 60, 70, 100]
LEADS = [round(-0.5 + 0.1 * i, 2) for i in range(21)]  # -0.5 .. 1.5 s


class Shallow(nn.Module):
    """braindecode ShallowFBCSPNet forward (eval), from its state dict: input (b, ch, 500)"""

    def __init__(self, sd):
        super().__init__()
        n_ch = sd["conv_time_spat.conv_spat.weight"].shape[-1]
        self.conv_time = nn.Conv2d(1, 40, (25, 1))
        self.conv_spat = nn.Conv2d(40, 40, (1, n_ch), bias=False)
        self.bnorm = nn.BatchNorm2d(40)
        self.cls = nn.Conv2d(40, 2, tuple(sd["final_layer.conv_classifier.weight"].shape[2:]))
        m = {
            "conv_time_spat.conv_time.weight": "conv_time.weight",
            "conv_time_spat.conv_time.bias": "conv_time.bias",
            "conv_time_spat.conv_spat.weight": "conv_spat.weight",
            "final_layer.conv_classifier.weight": "cls.weight",
            "final_layer.conv_classifier.bias": "cls.bias",
        }
        self.load_state_dict({m.get(k, k): v for k, v in sd.items()}, strict=True)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)  # (b, 1, time, ch)
        x = self.bnorm(self.conv_spat(self.conv_time(x)))
        x = F.avg_pool2d(x * x, (75, 1), (15, 1))
        x = self.cls(torch.log(torch.clamp(x, min=1e-6)))
        return x[:, :, 0, 0]


@torch.no_grad()
def predict(model, Xw, device, bs=512):
    out = []
    for i in range(0, len(Xw), bs):
        out.append(model(torch.from_numpy(Xw[i : i + bs]).to(device)).argmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, int)


def majority16(model, epochs, labels, device):
    """the simulator's vote: 16 windows (sg.windows_of = CAR, then band-pass), majority, a tie is wrong"""
    W = sg.windows_of(epochs)  # (trial, 16, 1, ch, 500)
    p = predict(model, W.reshape(-1, *W.shape[3:]), device).reshape(W.shape[0], W.shape[1])
    hits = (p == labels[:, None]).sum(1)
    return hits * 2 > W.shape[1], hits / W.shape[1]


# ---------------- V0 ----------------
def c000_trials(ses_dir):
    """trial ids of run 1 whose END was logged while c_000 was the loaded checkpoint (session log.txt)"""
    p = os.path.join(ses_dir, "log.txt")
    if not os.path.exists(p):
        return None
    run, ck, ids, seen_ck = None, None, [], False
    for line in open(p, encoding="utf-8", errors="ignore"):
        m = re.search(r"目前 Run 編號:\s*(\d+)", line)
        if m:
            run = int(m[1])
            if run > 1:
                break
        m = re.search(r"update_check_point\s+(\S+)", line)
        if m:
            ck = re.split(r"[\\/]", m[1])[-1]
            seen_ck = True
            if ck != "c_000.pth" and ids:
                break
        m = re.search(r"Trial\s+(\d+)\s+END:", line)
        if m and run == 1 and ck == "c_000.pth":
            ids.append(int(m[1]))
    return ids if seen_ck else None


def v0_session(job):
    sub, ses, out = job
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdir = os.path.join(sg.DATA_DIR, sub, ses)
    rdir = os.path.join(sdir, "run1")
    ids = c000_trials(sdir)
    if (
        not ids
        or not glob.glob(os.path.join(rdir, "eeg_record_*.csv"))
        or not glob.glob(os.path.join(rdir, "log_*.txt"))
    ):
        return None
    ts, X, rec, trials = ov.read_stream(rdir)
    refs, kinds, _ = ov.vote_refs(trials)
    labels = np.array([t["label"] for t in trials])
    # check the 13-channel stream against mi.pt (recorded order)
    chk = np.nan
    if os.path.exists(os.path.join(rdir, "mi.pt")):
        d = torch.load(os.path.join(rdir, "mi.pt"), weights_only=False)
        x = np.asarray(d["x_data"])
        if len(x) == len(trials):
            xr = np.empty_like(x)
            xr[sg.arrange_idx(labels)] = x
            ep = ov.epoch_from_stream(ts, X, rec, trials, ov.C13, x.shape[2])
            chk = float(np.abs(ep - xr).max())
    sel = [i for i, t in enumerate(trials) if t["id"] in set(ids)]
    if not sel:
        return None
    ck = torch.load(C000, map_location="cpu", weights_only=False)
    model = Shallow(ck["model_state_dict"]).to(device).eval()
    rows = []
    ep = ov.epoch_from_stream(ts, X, rec, [trials[i] for i in sel], ov.C13)
    c16, m16 = majority16(model, ep, labels[sel], device)
    lo = ov.end_index(ts, rec, refs[sel] - max(LEADS) - (ov.HIST - 1) / min(RATES))
    hi = ov.end_index(ts, rec, refs[sel] - min(LEADS))
    dense = {}
    for car in (True, False):
        for k, i in enumerate(sel):
            ends = np.arange(lo[k], hi[k] + 1)
            dense[(car, i)] = (lo[k], predict(model, ov.prep_windows(X, ends, ov.C13, car=car), device))
    for k, i in enumerate(sel):
        t = trials[i]
        base = dict(
            subject=sub,
            session=ses,
            trial=t["id"],
            label=int(t["label"]),
            n_cut=len(t["cut"]),
            kind=int(kinds[i]),
            trial_len=round(t["end"] - t["start"], 3),
            logged_success=int(kinds[i] == 5),
            c16=int(c16[k]),
            m16=round(float(m16[k]), 4),
            mi_check=chk,
        )
        for car in (True, False):
            lo_i, pr = dense[(car, i)]
            for rate in RATES:
                for lead in LEADS:
                    e = ov.vote_ends(ts, rec, refs[i : i + 1], lead, rate)[0]
                    P = pr[e - lo_i]
                    rows.append(
                        dict(
                            base,
                            car=int(car),
                            rate=rate,
                            lead=lead,
                            vote=int(ov.unity_vote(P[None])[0]),
                            share1=round(float((P * ov.unity_weights()[1]).sum() / ov.unity_weights()[1].sum()), 4),
                        )
                    )
    print(f"V0 {sub}/{ses}: {len(sel)} c_000 trials, mi.pt check max err {chk}", flush=True)
    return rows


def v0(args):
    subs = sorted(d for d in os.listdir(sg.DATA_DIR) if d.isdigit())
    jobs = [
        (s, ses, args.out) for s in subs for ses in ("s1", "s2") if os.path.isdir(os.path.join(sg.DATA_DIR, s, ses))
    ]
    with get_context("spawn").Pool(args.workers) as pool:
        res = [r for r in pool.map(v0_session, jobs) if r]
    rows = [r for rs in res for r in rs]
    write(os.path.join(args.out, "v0_rows.csv"), rows)
    v0_report(args.out)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def v0_report(out):
    import pandas as pd

    df = pd.read_csv(os.path.join(out, "v0_rows.csv"))
    df["study"] = df["subject"].astype(str).str.zfill(2).isin(sg.SUBJECTS)
    lines = []
    for scope, d in (("all subjects", df), ("study subjects 35-70", df[df.study])):
        one = d.drop_duplicates(["subject", "session", "trial"])
        n = len(one)
        lines.append(
            f"== {scope}: {n} trials voted by c_000 ({one.groupby(['subject','session']).ngroups} sessions); "
            f"logged success {one.logged_success.mean():.3f}; mi.pt check max err {one.mi_check.max()}"
        )
        for mp, sign in (("label 1 = class 1", 1), ("label 1 = class 0", -1)):
            c16 = one.c16 if sign == 1 else ((one.m16 < 0.5) & (one.m16 * 16 != 8))
            po, k = kappa(c16, one.logged_success)
            lines.append(
                f"   16-window majority ({mp}): agreement {po:.3f}, kappa {k:.3f}, c_000 accuracy {c16.mean():.3f}"
            )
        g = []
        for (car, rate, lead), dd in d.groupby(["car", "rate", "lead"]):
            for mp, sign in (("same", 1), ("inverted", -1)):
                corr = (dd.vote == dd.label) if sign == 1 else (dd.vote != dd.label)
                po, k = kappa(corr, dd.logged_success)
                g.append((po, k, car, rate, lead, mp, corr.mean()))
        g.sort(key=lambda r: -r[0])
        lines.append("   replica, best 15 settings (agreement, kappa, CAR, rate Hz, lead s, label map, accuracy):")
        for r in g[:15]:
            lines.append(
                f"     {r[0]:.3f}  {r[1]:.3f}  car={r[2]}  rate={r[3]:>3}  lead={r[4]:.2f}  {r[5]:8s} acc {r[6]:.3f}"
            )
        lines.append(
            "   replica, worst 3: " + "; ".join(f"{r[0]:.3f} (car={r[2]},{r[3]}Hz,{r[4]:.2f}s,{r[5]})" for r in g[-3:])
        )
        # agreement by lead at the best rate / car / map
        b = g[0]
        dd = d[(d.car == b[2]) & (d.rate == b[3])]
        prof = []
        for lead, x in dd.groupby("lead"):
            corr = (x.vote == x.label) if b[5] == "same" else (x.vote != x.label)
            prof.append(f"{lead:.2f}:{(corr == x.logged_success).mean():.3f}")
        lines.append(f"   agreement by lead (car={b[2]}, {b[3]} Hz, {b[5]}): " + " ".join(prof))
        prof = []
        dd = d[(d.car == b[2]) & (d.lead == b[4])]
        for rate, x in dd.groupby("rate"):
            corr = (x.vote == x.label) if b[5] == "same" else (x.vote != x.label)
            prof.append(f"{rate}:{(corr == x.logged_success).mean():.3f}")
        lines.append(f"   agreement by rate (car={b[2]}, lead {b[4]:.2f}, {b[5]}): " + " ".join(prof))
        x = d[(d.car == b[2]) & (d.rate == b[3]) & (d.lead == b[4])]
        corr = (x.vote == x.label) if b[5] == "same" else (x.vote != x.label)
        for kind in (5, 4, 0):
            s = x.kind == kind
            if s.any():
                lines.append(
                    f"   best setting, reference kind {kind} (5 = brain note cut, 4 = 4th note + gap, 0 = START + offset): n {s.sum()}, "
                    f"agreement {(corr[s] == x.logged_success[s]).mean():.3f}"
                )
        for nm, s in (("short trials (<4.2 s)", x.trial_len < 4.2), ("long trials", x.trial_len >= 4.2)):
            if s.any():
                lines.append(
                    f"   best setting, {nm}: n {s.sum()}, agreement {(corr[s] == x.logged_success[s]).mean():.3f}, "
                    f"16-window {(x.c16[s] == x.logged_success[s]).mean():.3f}"
                )
        # an exact replica must agree on the trials where the model is sure (vote far from the 50 % line)
        for thr in (0.1, 0.2, 0.3):
            s = (x.share1 - 0.5).abs() > thr
            s16 = (x.m16 - 0.5).abs() > thr
            lines.append(
                f"   best setting, trials with |weighted share of class 1 - 0.5| > {thr}: n {s.sum()}, "
                f"agreement {(corr[s] == x.logged_success[s]).mean():.3f};  16-window with |share - 0.5| > {thr}: "
                f"n {s16.sum()}, agreement {(x.c16[s16] == x.logged_success[s16]).mean():.3f}"
            )
        dd = d[(d.rate == 70) & (d.lead == 0.3) & (d.car == 1)]
        if len(dd):
            corr = dd.vote == dd.label
            po, k = kappa(corr, dd.logged_success)
            lines.append(
                f"   a priori setting (70 Hz per the old config comment, lead 0.3 s, CAR): agreement {po:.3f}, kappa {k:.3f}"
            )
    txt = "\n".join(lines)
    open(os.path.join(out, "v0_summary.txt"), "w").write(txt + "\n")
    print(txt)


# ---------------- V1 ----------------
def parse_configs(s):
    return [(float(a), float(b)) for a, b in (c.split(":") for c in s.split(","))]


def cfg_name(rate, lead):
    return f"u{rate:g}_{lead:g}"


def v1_session(job):
    sub, ses, ckpt, configs = job
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = sg.SCCNet(**dict(config.LOAD_MODEL_PARAM, channels=int(ck.get("n_chans", 22)))).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    runs = sg.load_session(sub, ses, sg.DATA_DIR)
    c22 = config.CHANNEL_DEFINITIONS["22"]
    rows, fb = [], None
    for r in runs:
        rdir = os.path.join(sg.DATA_DIR, sub, ses, f"run{r['run']}")
        ts, X, rec, trials = ov.read_stream(rdir)
        labels = np.array([t["label"] for t in trials])
        if len(labels) != len(r["y"]) or not (labels == r["y"]).all():
            raise RuntimeError(f"{sub}/{ses}/run{r['run']}: log trials differ from load_session")
        ep = ov.epoch_from_stream(ts, X, rec, trials, c22, r["x"].shape[2])
        err = float(np.abs(ep - r["x"]).max())
        if err > 1e-3:
            raise RuntimeError(f"{sub}/{ses}/run{r['run']}: stream epoch != mi_22 (max err {err})")
        refs, kinds, fb_run = ov.vote_refs(trials, fb)
        fb = fb or fb_run
        W = sg.windows_of(r["x"])
        c16, m16 = sg.vote_all(model, W, labels, device)
        ends = {c: ov.vote_ends(ts, rec, refs, c[1], c[0]) for c in configs}
        uniq, inv = np.unique(np.concatenate([e.reshape(-1) for e in ends.values()]), return_inverse=True)
        pu = predict(model, ov.prep_windows(X, uniq, c22), device)
        res, off = {}, 0
        for c, e in ends.items():
            P = pu[inv[off : off + e.size]].reshape(e.shape)
            off += e.size
            res[c] = ov.unity_correct(P, labels)
        for i, t in enumerate(trials):
            row = dict(
                subject=sub,
                session=ses,
                run=r["run"],
                pos=i,
                label=int(labels[i]),
                kind=int(kinds[i]),
                trial_len=round(t["end"] - t["start"], 3),
                logged_success=int(kinds[i] == 5),
                c16=int(c16[i]),
                m16=round(float(m16[i]), 4),
            )
            for c in configs:
                row["c_" + cfg_name(*c)] = int(res[c][0][i])
                row["m_" + cfg_name(*c)] = round(float(res[c][1][i]), 4)
            rows.append(row)
    print(
        f"V1 {sub}/{ses}: {len(rows)} trials, acc16 {np.mean([x['c16'] for x in rows]):.3f}, "
        + ", ".join(f"{cfg_name(*c)} {np.mean([x['c_' + cfg_name(*c)] for x in rows]):.3f}" for c in configs),
        flush=True,
    )
    return rows


def v1(args):
    starts = json.load(open(args.start_ckpts))
    configs = parse_configs(args.configs)
    jobs = [(s, ses, starts[s], configs) for s in sg.SUBJECTS for ses in ("s1", "s2")]
    with get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(v1_session, jobs)
    write(os.path.join(args.out, "v1_trials.csv"), [r for rs in res for r in rs])
    v1_report(args.out)


def gate_fires(c, run_of, kind, params=None):
    """trial indices a gate would fire on with this correctness sequence and no finetune (resets at run start)"""
    fires = []
    for run in np.unique(run_of):
        idx = np.where(run_of == run)[0]
        cc = c[idx]
        if kind == "sprt":
            g = SprtGate(*params)
            for j, ok in enumerate(cc):
                if g.update(bool(ok)) == "finetune":
                    fires.append(idx[j])
        else:  # block: >= 1 wrong in the block and run accuracy < 0.75
            for k in range(4, len(cc) + 1, 4):
                if (~cc[k - 4 : k]).sum() >= 1 and cc[:k].mean() < 0.75:
                    fires.append(idx[k - 1])
    return set(fires)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def v1_report(out):
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.read_csv(os.path.join(out, "v1_trials.csv"))
    cfgs = [c[2:] for c in df.columns if c.startswith("c_u")]
    lines = [f"V1: {len(df)} trials, {df.groupby(['subject','session']).ngroups} sessions, start models, no finetune"]
    for cfg in cfgs:
        cu = df["c_" + cfg].astype(bool)
        c16 = df.c16.astype(bool)
        po, k = kappa(c16, cu)
        r = np.corrcoef(df.m16, df["m_" + cfg])[0, 1]
        ses = df.assign(cu=cu, c16=c16).groupby(["subject", "session"])[["c16", "cu"]].mean()
        sub = ses.groupby("subject").mean()
        d = (sub.cu - sub.c16).to_numpy()
        lo, hi = boot_ci(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        lines.append(f"\n== replica {cfg} vs 16-window majority")
        lines.append(f"   trial agreement {po:.3f}, kappa {k:.3f}, margin correlation {r:.3f}")
        lines.append(
            f"   accuracy (subject mean): 16-window {sub.c16.mean():.4f}, replica {sub.cu.mean():.4f}, "
            f"replica - 16 = {d.mean()*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}], Wilcoxon p {p:.3g}"
        )
        lines.append(f"   subject accuracy rank correlation (Spearman) {sub.c16.rank().corr(sub.cu.rank()):.3f}")
        for nm, s in (("short trials (<4.2 s)", df.trial_len < 4.2), ("long trials", df.trial_len >= 4.2)):
            if s.any():
                lines.append(
                    f"   {nm}: n {s.sum()}, agreement {(c16[s] == cu[s]).mean():.3f}, acc16 {c16[s].mean():.3f}, "
                    f"replica {cu[s].mean():.3f}"
                )
        off = c16 != cu
        lines.append(
            f"   disagreements: {off.sum()}; 16 right / replica wrong {(c16 & ~cu).sum()}, "
            f"16 wrong / replica right {(~c16 & cu).sum()}; share by position in run (quarters): "
            + " ".join(f"{x:.3f}" for x in off.groupby(df.pos // 10).mean())
        )
        for gname, kind, params in (
            ("SPRT 0.65/0.85/0.05/0.2", "sprt", (0.65, 0.85, 0.05, 0.2)),
            ("SPRT 0.75/0.95/0.05/0.2", "sprt", (0.75, 0.95, 0.05, 0.2)),
            ("block (0.75)", "block", None),
        ):
            n16, nu, inter, uni = [], [], 0, 0
            for _, g in df.assign(cu=cu, c16=c16).groupby(["subject", "session"]):
                f16 = gate_fires(g.c16.to_numpy(), g.run.to_numpy(), kind, params)
                fu = gate_fires(g.cu.to_numpy(), g.run.to_numpy(), kind, params)
                n16.append(len(f16)), nu.append(len(fu))
                inter += len(f16 & fu)
                uni += len(f16 | fu)
            lines.append(
                f"   {gname}: fires per session 16-window {np.mean(n16):.1f}, replica {np.mean(nu):.1f}; "
                f"same trial (Jaccard) {inter / max(uni, 1):.3f}"
            )
    txt = "\n".join(lines)
    open(os.path.join(out, "v1_summary.txt"), "w").write(txt + "\n")
    print(txt)


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["v0", "v1", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--start_ckpts")
    ap.add_argument("--configs", default="70:0.3")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what == "v0":
        v0(a)
    elif a.what == "v1":
        v1(a)
    else:
        if os.path.exists(os.path.join(a.out, "v0_rows.csv")):
            v0_report(a.out)
        if os.path.exists(os.path.join(a.out, "v1_trials.csv")):
            v1_report(a.out)


if __name__ == "__main__":
    main()
