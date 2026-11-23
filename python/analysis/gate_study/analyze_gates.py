# python python/analysis/gate_study/analyze_gates.py --sim <simulate_gates.py --out dir> [--ref block]
"""
Summary of a simulate_gates.py run.
Per session x condition x seed: online accuracy (each trial voted by the model in use), accuracy per run and in runs 5-7,
number of finetunes, finetune seconds. Seeds are averaged per session, sessions per subject (n = subjects).
Per finetune: child - parent vote accuracy on the next 48 trials (from matrix.npz), as F1 in plan_finetune_evaluation.md.
Paired comparison of every condition with --ref: mean difference over subjects, bootstrap 95% CI, Wilcoxon signed-rank.
Writes per_session.csv, per_finetune.csv and summary.txt into --sim.
"""

import argparse
import collections
import csv
import glob
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # 沒有 scipy 就不算 p 值
    wilcoxon = None

NEXT = 48


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--ref", default="block")
    args = ap.parse_args()

    per_session, per_ft = [], []
    for d in sorted(glob.glob(os.path.join(args.sim, "*_s[12]", "*_seed*"))):
        if not os.path.exists(os.path.join(d, "trials.csv")):
            continue
        sub, ses = os.path.basename(os.path.dirname(d)).split("_")
        cond, seed = os.path.basename(d).rsplit("_seed", 1)
        tr = read_csv(os.path.join(d, "trials.csv"))
        up = read_csv(os.path.join(d, "updates.csv"))
        correct = np.array([int(r["correct"]) for r in tr])
        runs = np.array([int(r["run"]) for r in tr])
        row = dict(
            subject=sub,
            session=ses,
            condition=cond,
            seed=int(seed),
            acc=correct.mean(),
            n_ft=len(up),
            ft_seconds=sum(float(u["seconds"]) for u in up),
            acc_late=correct[runs >= 5].mean() if (runs >= 5).any() else np.nan,
        )
        for r in sorted(set(runs)):
            row[f"acc_run{r}"] = correct[runs == r].mean()
        per_session.append(row)
        if up:
            m = np.load(os.path.join(d, "matrix.npz"))["correct"]
            for u in up:
                v, t = int(u["version"]), int(u["after_trial"])
                t_load = int(u.get("load_trial") or t + 1)  # --delay: 新權重從 load_trial 開始用
                w = slice(t_load, min(t_load + NEXT, m.shape[1]))
                if w.stop - w.start < 1:
                    continue
                parent = int(tr[t]["version"])  # 微調時在用的模型
                per_ft.append(
                    dict(
                        subject=sub,
                        session=ses,
                        condition=cond,
                        seed=int(seed),
                        version=v,
                        after_trial=t,
                        n_next=w.stop - w.start,
                        parent_acc=m[parent, w].mean(),
                        child_acc=m[v, w].mean(),
                        effect=m[v, w].mean() - m[parent, w].mean(),
                    )
                )
    if not per_session:
        raise SystemExit(f"no results in {args.sim}")

    keys = sorted(
        {k for r in per_session for k in r}, key=lambda k: (not k in ("subject", "session", "condition", "seed"), k)
    )
    with open(os.path.join(args.sim, "per_session.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_session)
    if per_ft:
        with open(os.path.join(args.sim, "per_finetune.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_ft[0]))
            w.writeheader()
            w.writerows(per_ft)

    # seeds -> session -> subject
    sess = collections.defaultdict(list)
    for r in per_session:
        sess[(r["condition"], r["subject"], r["session"])].append(r)
    subj = collections.defaultdict(lambda: collections.defaultdict(list))
    for (cond, sub, ses), rs in sess.items():
        for k in ("acc", "acc_late", "n_ft", "ft_seconds"):
            subj[cond][(sub, k)].append(np.nanmean([r[k] for r in rs]))
    conds = sorted(subj)
    subjects = sorted({s for c in conds for (s, _) in subj[c]})

    def subj_vals(cond, k):
        return np.array([np.mean(subj[cond][(s, k)]) if (s, k) in subj[cond] else np.nan for s in subjects])

    lines = [
        f"Gate simulation summary: {args.sim}",
        f"{len(subjects)} subjects, sessions per condition: "
        + ", ".join(f"{c} {len({(s, ses) for (cc, s, ses) in sess if cc == c})}" for c in conds),
        "",
    ]
    lines.append(f"{'condition':28s} {'acc':>7s} {'acc run5-7':>10s} {'finetunes':>10s} {'ft sec/session':>15s}")
    for c in conds:
        lines.append(
            f"{c:28s} {np.nanmean(subj_vals(c, 'acc')):7.3f} {np.nanmean(subj_vals(c, 'acc_late')):10.3f} "
            f"{np.nanmean(subj_vals(c, 'n_ft')):10.1f} {np.nanmean(subj_vals(c, 'ft_seconds')):15.1f}"
        )
    if args.ref in conds:
        lines += ["", f"Paired difference vs {args.ref} (per subject; mean, 95% bootstrap CI, Wilcoxon p)"]
        for k in ("acc", "acc_late"):
            for c in conds:
                if c == args.ref:
                    continue
                dlt = subj_vals(c, k) - subj_vals(args.ref, k)
                dlt = dlt[~np.isnan(dlt)]
                lo, hi = boot_ci(dlt)
                p = wilcoxon(dlt).pvalue if wilcoxon is not None and np.any(dlt != 0) else float("nan")
                lines.append(
                    f"  {k:9s} {c:28s} {100 * dlt.mean():+6.2f} pp  [{100 * lo:+.2f}, {100 * hi:+.2f}]  p={p:.3g}"
                )
    if per_ft:
        lines += [
            "",
            f"Per finetune: child - parent on the next {NEXT} trials (mean over finetunes; share that helped / hurt)",
        ]
        by = collections.defaultdict(list)
        for r in per_ft:
            by[r["condition"]].append(r["effect"])
        for c in sorted(by):
            e = np.array(by[c])
            lines.append(
                f"  {c:28s} n={len(e):6d}  {100 * e.mean():+6.2f} pp  helped {np.mean(e > 0):.2f}  hurt {np.mean(e < 0):.2f}"
            )
    # accuracy per run
    run_keys = sorted({k for r in per_session for k in r if k.startswith("acc_run")}, key=lambda k: int(k[7:]))
    lines += [
        "",
        "Accuracy per run (mean over sessions and seeds)",
        f"{'condition':28s} " + " ".join(f"{k[4:]:>6s}" for k in run_keys),
    ]
    for c in conds:
        rs = [r for r in per_session if r["condition"] == c]
        lines.append(f"{c:28s} " + " ".join(f"{np.nanmean([r.get(k, np.nan) for r in rs]):6.3f}" for k in run_keys))
    text = "\n".join(lines)
    open(os.path.join(args.sim, "summary.txt"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
