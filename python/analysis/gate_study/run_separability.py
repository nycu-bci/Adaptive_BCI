# compute (server, CPU only):  python run_separability.py compute --out <dir> [--gate_dir <dir with simulate_gates.py>] [--jobs 8]
# summarize (local):          python run_separability.py summarize --out <dir> [--never <round 2 per_session.csv>]
"""
Are later runs easier to decode (the subject got better at MI), or not? Premise check for a "judge finetunes on the late
runs" oracle. Independent of the start model and of any gate: classic MI decoders fitted per session on the raw EEG.

Per session (subjects 35-70, s1/s2, trials in recorded order via simulate_gates.load_session):
  within-run separability: each run alone, repeated stratified 5-fold CV (10 repeats)
    csp = CAR, 8-30 Hz band-pass (4th-order Butterworth, zero phase), whole 4 s trial (2000 samples at 500 Hz),
          CSP 6 components (Ledoit-Wolf regularised, log-variance) + shrinkage LDA
    bp  = log band power (8-13 Hz and 13-30 Hz) of each of the 22 channels + shrinkage LDA
  drift: CSP + LDA fitted on runs 1-2 (80 trials), tested on each later run (no CV needed)
Unit = subject (sessions averaged). Per-run mean with bootstrap 95% CI; per-subject linear slope over runs (pp per run,
Wilcoxon); runs 1-2 vs 5-7 paired difference. Chance = 50%; one run's CV accuracy has an SD of about 8 pp.
Writes per_run.csv (compute) and summary.txt (summarize) into --out.
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np

CV_FOLDS, CV_REPEATS = 5, 10


def features_session(runs, fs=500):
    from scipy.signal import butter, sosfiltfilt

    def car(x):
        return x - x.mean(axis=1, keepdims=True)  # (trial, ch, sample)

    sos_mu_beta = butter(4, [8, 30], btype="band", fs=fs, output="sos")
    sos_mu = butter(4, [8, 13], btype="band", fs=fs, output="sos")
    sos_beta = butter(4, [13, 30], btype="band", fs=fs, output="sos")
    out = []
    for r in runs:
        x = car(np.asarray(r["x"][:, :, :2000], dtype=np.float64))
        xf = sosfiltfilt(sos_mu_beta, x, axis=-1)
        bp = np.concatenate(
            [np.log(np.var(sosfiltfilt(s, x, axis=-1), axis=-1)) for s in (sos_mu, sos_beta)], axis=1
        )  # (trial, 44)
        out.append(dict(run=r["run"], y=r["y"], xf=xf, bp=bp))
    return out


def one_session(args, sub, ses):
    sys.path.insert(0, args.gate_dir)
    import simulate_gates as sg  # load_session: trials put back in recorded order
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    import mne

    mne.set_log_level("ERROR")

    def csp_lda():
        return make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True), LDA(solver="lsqr", shrinkage="auto"))

    def bp_lda():
        return LDA(solver="lsqr", shrinkage="auto")

    feats = features_session(sg.load_session(sub, ses, args.data_dir))
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=0)
    by_run = {f["run"]: f for f in feats}
    train = [by_run[r] for r in (1, 2) if r in by_run]
    drift_model = None
    if len(train) == 2:
        drift_model = csp_lda().fit(np.concatenate([f["xf"] for f in train]), np.concatenate([f["y"] for f in train]))
    rows = []
    for f in feats:
        y = f["y"]
        rows.append(
            dict(
                subject=sub,
                session=ses,
                run=f["run"],
                n_trials=len(y),
                n_class1=int(y.sum()),
                acc_csp=cross_val_score(csp_lda(), f["xf"], y, cv=cv).mean(),
                acc_bp=cross_val_score(bp_lda(), f["bp"], y, cv=cv).mean(),
                acc_drift=(drift_model.score(f["xf"], y) if drift_model is not None and f["run"] >= 3 else np.nan),
            )
        )
    print(f"{sub}_{ses}: " + " ".join(f"r{r['run']} {r['acc_csp']:.2f}" for r in rows), flush=True)
    return rows


def compute(args):
    from joblib import Parallel, delayed

    sys.path.insert(0, args.gate_dir)
    import simulate_gates as sg

    if os.path.exists(os.path.join(args.out, "per_run.csv")):
        raise SystemExit(f"{args.out}/per_run.csv exists: use another --out")
    os.makedirs(args.out, exist_ok=True)
    jobs = [(s, ses) for s in sg.SUBJECTS for ses in ("s1", "s2")]
    res = Parallel(n_jobs=args.jobs)(delayed(one_session)(args, s, ses) for s, ses in jobs)
    rows = [r for rs in res for r in rs]
    with open(os.path.join(args.out, "per_run.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs", flush=True)


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])


def summarize(args):
    from scipy.stats import wilcoxon

    rows = list(csv.DictReader(open(os.path.join(args.out, "per_run.csv"))))
    metrics = ["acc_csp", "acc_bp", "acc_drift"]
    subj = collections.defaultdict(lambda: collections.defaultdict(list))  # (metric, sub) -> run -> values
    for r in rows:
        for m in metrics:
            v = float(r[m])
            if not np.isnan(v):
                subj[(m, r["subject"])][int(r["run"])].append(v)
    if args.never:  # 起始模型固定不變 (Never) 在各 run 的正確率
        for r in csv.DictReader(open(args.never)):
            if r["condition"] != "never":
                continue
            for k, v in r.items():
                if k.startswith("acc_run") and v not in ("", "nan"):
                    subj[("acc_never", r["subject"])][int(k[7:])].append(float(v))
        metrics.append("acc_never")
    subjects = sorted({s for (_, s) in subj})
    runs = range(1, 8)

    def mat(m):  # subject x run, sessions averaged
        return np.array([[np.mean(subj[(m, s)][k]) if subj[(m, s)].get(k) else np.nan for k in runs] for s in subjects])

    names = {
        "acc_csp": "within-run CSP+LDA (CV)",
        "acc_bp": "within-run band power+LDA (CV)",
        "acc_drift": "trained on runs 1-2, tested on run k",
        "acc_never": "start model, never finetuned (round 2)",
    }
    lines = [
        "Per-run separability (subjects 35-70; unit = subject, sessions averaged; chance 50%)",
        f"{len(subjects)} subjects, {len(rows)} runs; CV = stratified {CV_FOLDS}-fold x {CV_REPEATS} within each run",
        "",
        f"{'metric':40s} " + " ".join(f"{'run' + str(k):>6s}" for k in runs),
    ]
    for m in metrics:
        a = mat(m)
        lines.append(
            f"{names[m]:40s} "
            + " ".join(f"{np.nanmean(a[:, k]):6.3f}" if np.isfinite(a[:, k]).any() else f"{'-':>6s}" for k in range(7))
        )
    lines += ["", "95% CI of each run's mean (within-run CSP+LDA)"]
    a = mat("acc_csp")
    lines.append(
        " ".join(f"run{k + 1} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in enumerate(boot_ci(a[:, k]) for k in range(7)))
    )
    lines += ["", "Trend per subject (pp per run, mean [95% CI], Wilcoxon p) and runs 5-7 minus runs 1-2 (pp)"]
    for m in metrics:
        a = mat(m)
        slopes, diffs = [], []
        for row in a:
            ok = np.isfinite(row)
            if ok.sum() >= 3:
                slopes.append(np.polyfit(np.arange(1, 8)[ok], row[ok], 1)[0])
            early, late = np.nanmean(row[0:2]) if np.isfinite(row[0:2]).any() else np.nan, np.nanmean(row[4:7])
            diffs.append(late - early)
        slopes, diffs = np.array(slopes), np.array(diffs)
        lo, hi = boot_ci(slopes)
        line = f"  {names[m]:40s} slope {100 * slopes.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] p={wilcoxon(slopes).pvalue:.3g}"
        d = diffs[np.isfinite(diffs)]
        if len(d) >= 5:
            lo, hi = boot_ci(d)
            line += f" | late-early {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] p={wilcoxon(d).pvalue:.3g}"
        lines.append(line)
    a = mat("acc_csp")
    best = np.nanargmax(a, axis=1) + 1
    lines += [
        "",
        "Run with the highest within-run CSP accuracy, per subject: "
        + ", ".join(f"run{k} {int((best == k).sum())}" for k in runs),
        f"Subjects whose late runs (5-7) beat early runs (1-2) on CSP: "
        f"{int(np.sum(np.nanmean(a[:, 4:7], 1) > np.nanmean(a[:, 0:2], 1)))} / {len(subjects)}",
    ]
    text = "\n".join(lines)
    open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8").write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["compute", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data_dir", default="/mnt/project/MIEXP/DATA_Cygnus")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--never", default=None, help="round 2 per_session.csv (for the never-finetuned start model)")
    args = ap.parse_args()
    compute(args) if args.step == "compute" else summarize(args)


if __name__ == "__main__":
    main()
