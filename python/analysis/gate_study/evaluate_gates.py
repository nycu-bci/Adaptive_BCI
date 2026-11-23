# python python/analysis/gate_study/evaluate_gates.py --sim <dir with per_session.csv from analyze_gates.py> \
#        [--design <sprt_design_space.py out>] [--out <dir>]
"""
Is the SPRT gate worth it, and with which parameters? Pre-specified evaluation of a simulate_gates.py round.
Unit = subject (seeds averaged per session, sessions per subject; n = 24). Accuracy = online trial accuracy over all runs.

Families: periodic (never, every:20/12/8/6, always), random (random:q), block (block, block:thr), sprt (all sprt_*),
oracle (cheating ceiling, not a usable gate).

1. Table: accuracy, accuracy runs 5-7, finetunes (and trainings) per session, per condition.
2. Information value: for each subject, the periodic family gives accuracy as a function of finetunes per session
   (piecewise linear between its points). A gate's "residual" = its accuracy - the periodic curve at the gate's own
   number of finetunes. Residual > 0 = the gate picks better moments than a clock with the same budget.
   Also against the random family when present.
3. Paired comparisons with the block gate (mean over subjects, 95% bootstrap CI, Wilcoxon signed-rank).
4. Tuned families by nested cross-validation over the 4 subject folds of the start models: for each fold, pick the
   configuration of a family on the other 18 subjects (objective: accuracy, optionally with a cap on finetunes per
   session), score it on the 6 held-out subjects. Removes the optimism of picking the best of 21 SPRT settings.
5. Decision rules (Holm over H1-H3):
   H1 superiority     tuned SPRT accuracy > block accuracy
   H2 efficiency      tuned SPRT (cap: block's finetunes x 0.7) not worse than block by more than 1 pp
                      (lower 95% CI bound > -1 pp) while using >= 30% fewer finetunes
   H3 information     residual of SPRT default > 0 (beats a clock with the same budget)
   Worth it = H1 or (H2 and H3).
Writes evaluation.txt, conditions.csv, subject_conditions.csv into --out.
"""

import argparse
import collections
import csv
import os
import re

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

FOLDS = [
    ["35", "43", "44", "50", "54", "58"],
    ["38", "41", "48", "65", "68", "69"],
    ["40", "42", "47", "63", "64", "70"],
    ["37", "45", "51", "52", "55", "57"],
]
DEFAULT_SPRT = "sprt_0.65_0.85_0.05_0.2"


def family(c):
    if c in ("never", "always") or c.startswith("every_"):
        return "periodic"
    if c.startswith("random_"):
        return "random"
    if c == "block" or c.startswith("block_"):
        return "block"
    if c.startswith("sprt_"):
        return "sprt"
    if c.startswith("oracle"):
        return "oracle"
    return "other"


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])


def wil(x):
    x = np.asarray(x, float)
    return wilcoxon(x).pvalue if wilcoxon is not None and np.any(x != 0) else float("nan")


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--design", default=None, help="sprt_design_space.py output dir (grid.csv) for SPRT properties")
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=0.01, help="non-inferiority margin (accuracy, 0.01 = 1 pp)")
    args = ap.parse_args()
    out = args.out or args.sim
    os.makedirs(out, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(args.sim, "per_session.csv"))))
    trainings = {}
    for r in rows:
        if r["condition"].startswith("oracle"):
            p = os.path.join(
                args.sim, f"{r['subject']}_{r['session']}", f"{r['condition']}_seed{r['seed']}", "oracle.csv"
            )
            if os.path.exists(p):
                trainings[(r["subject"], r["session"], r["condition"], r["seed"])] = sum(1 for _ in open(p)) - 1
    # seeds -> session -> subject
    ses = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = (r["condition"], r["subject"], r["session"])
        for k in ("acc", "acc_late", "n_ft"):
            ses[key][k].append(float(r[k]))
        ses[key]["n_train"].append(
            trainings.get((r["subject"], r["session"], r["condition"], r["seed"]), float(r["n_ft"]))
        )
        ses[key]["seeds"].append(int(r["seed"]))
    subj = collections.defaultdict(dict)  # cond -> subject -> metrics
    for (c, s, _), d in ses.items():
        m = subj[c].setdefault(s, collections.defaultdict(list))
        for k in ("acc", "acc_late", "n_ft", "n_train"):
            m[k].append(np.mean(d[k]))
        m["n_seeds"].append(len(d["seeds"]))
    subjects = sorted({s for c in subj for s in subj[c]})
    conds = sorted(subj, key=lambda c: (family(c), c))
    complete = [
        c for c in conds if all(s in subj[c] and len(subj[c][s]["acc"]) == 2 for s in subjects if s in subj["never"])
    ]

    def val(c, k):
        return np.array([np.mean(subj[c][s][k]) if s in subj[c] else np.nan for s in subjects])

    props = {}
    if args.design:
        for g in csv.DictReader(open(os.path.join(args.design, "grid.csv"))):
            name = "sprt_" + "_".join(f"{float(g[k]):g}" for k in ("p0", "p1", "alpha", "beta"))
            props[name] = g

    L = [
        f"Gate evaluation: {args.sim}",
        f"{len(subjects)} subjects; conditions complete for all: {len(complete)} of {len(conds)}",
        "",
    ]
    # ---- 1. table
    L.append(
        f"{'condition':26s} {'family':8s} {'acc':>6s} {'acc5-7':>6s} {'ft/ses':>6s} {'train/ses':>9s} {'seeds':>5s}  sprt properties"
    )
    table = []
    for c in conds:
        a, al, nf, nt = (np.nanmean(val(c, k)) for k in ("acc", "acc_late", "n_ft", "n_train"))
        nseed = int(np.median([np.mean(subj[c][s]["n_seeds"]) for s in subj[c]]))
        pr = props.get(c)
        ptxt = (
            f"p*={float(pr['p_star']):.3f} wrongs={pr['wrongs_to_fire']} c/w={float(pr['corrects_per_wrong']):.2f} good={pr['corrects_to_good']}"
            if pr
            else ""
        )
        L.append(f"{c:26s} {family(c):8s} {a:6.3f} {al:6.3f} {nf:6.1f} {nt:9.1f} {nseed:5d}  {ptxt}")
        table.append(
            dict(
                condition=c,
                family=family(c),
                acc=a,
                acc_late=al,
                ft_per_session=nf,
                trainings_per_session=nt,
                seeds=nseed,
                **(
                    {
                        k: pr[k]
                        for k in (
                            "p0",
                            "p1",
                            "alpha",
                            "beta",
                            "p_star",
                            "wrongs_to_fire",
                            "corrects_per_wrong",
                            "corrects_to_good",
                        )
                    }
                    if pr
                    else {}
                ),
            )
        )

    # ---- 2. residual vs the periodic (clock) curve, per subject
    def curve_residual(c, fam):
        pts = [x for x in conds if family(x) == fam or (fam == "random" and x == "never")]
        if fam == "periodic":
            pts = [x for x in conds if family(x) == "periodic"]
        res = []
        for i, s in enumerate(subjects):
            xs = [(np.mean(subj[p][s]["n_ft"]), np.mean(subj[p][s]["acc"])) for p in pts if s in subj[p]]
            if len(xs) < 2 or s not in subj[c]:
                res.append(np.nan)
                continue
            xs.sort()
            x, y = np.array(xs).T
            res.append(np.mean(subj[c][s]["acc"]) - np.interp(np.mean(subj[c][s]["n_ft"]), x, y))
        return np.array(res)

    have_periodic = sum(family(c) == "periodic" for c in conds) >= 3
    have_random = sum(family(c) == "random" for c in conds) >= 2
    resid = {}
    if have_periodic:
        L += [
            "",
            "Information value: accuracy - clock curve at the same finetunes per subject (pp; 95% CI; Wilcoxon p)",
        ]
        for c in conds:
            if family(c) in ("periodic",):
                continue
            r = curve_residual(c, "periodic")
            r = r[~np.isnan(r)]
            if len(r) < 5:
                continue
            resid[c] = r
            lo, hi = boot_ci(r)
            extra = ""
            if have_random and family(c) != "random":
                rr = curve_residual(c, "random")
                rr = rr[~np.isnan(rr)]
                extra = f"   vs random curve {100 * rr.mean():+5.2f} pp"
            L.append(f"  {c:26s} {100 * r.mean():+5.2f} pp [{100 * lo:+.2f}, {100 * hi:+.2f}] p={wil(r):.3g}{extra}")

    # ---- 3. paired vs block
    if "block" in subj:
        L += ["", "Paired vs block (per subject): accuracy pp [95% CI] p | runs 5-7 pp | finetunes per session"]
        b_acc, b_late, b_ft = val("block", "acc"), val("block", "acc_late"), val("block", "n_ft")
        for c in conds:
            if c == "block":
                continue
            d, dl, df = val(c, "acc") - b_acc, val(c, "acc_late") - b_late, val(c, "n_ft") - b_ft
            ok = ~np.isnan(d)
            if ok.sum() < 5:
                continue
            lo, hi = boot_ci(d[ok])
            L.append(
                f"  {c:26s} {100 * d[ok].mean():+5.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] p={wil(d[ok]):.3g} | "
                f"{100 * np.nanmean(dl):+5.2f} | {np.nanmean(df):+6.1f} ({100 * np.nanmean(val(c, 'n_ft')) / np.nanmean(b_ft):.0f}% of block)"
            )

    # ---- 4. nested CV over subject folds
    def nested(fam_conds, cap=None):
        """per held-out subject: accuracy and finetunes of the config picked on the other folds"""
        acc, nft, picks = np.full(len(subjects), np.nan), np.full(len(subjects), np.nan), []
        for f in FOLDS:
            train = [i for i, s in enumerate(subjects) if s not in f]
            test = [i for i, s in enumerate(subjects) if s in f]
            best, best_v = None, -np.inf
            for c in fam_conds:
                a, n = val(c, "acc"), val(c, "n_ft")
                if np.isnan(a[train]).any():
                    continue
                if cap is not None and np.mean(n[train]) > cap:
                    continue
                if a[train].mean() > best_v:
                    best, best_v = c, a[train].mean()
            picks.append(best)
            if best is not None:
                acc[test], nft[test] = val(best, "acc")[test], val(best, "n_ft")[test]
        return acc, nft, picks

    fams = collections.defaultdict(list)
    for c in complete:
        fams[family(c)].append(c)
    cv_rows = []
    if "block" in subj:
        b_acc, b_ft = val("block", "acc"), val("block", "n_ft")
        L += [
            "",
            "Tuned by nested CV over the 4 subject folds (pick on 18 subjects, score on the 6 others)",
            "  family    cap(ft/ses)  acc    ft/ses   vs block pp [95% CI] p     picks per fold",
        ]
        caps = [None, 45, 35, 25, 15]
        for cap in caps:
            for fam in ("sprt", "block", "periodic", "random"):
                if not fams.get(fam):
                    continue
                a, n, picks = nested(fams[fam], cap)
                ok = ~np.isnan(a)
                if ok.sum() < len(subjects):
                    L.append(f"  {fam:9s} {str(cap):11s} (no config under the cap in some fold)")
                    continue
                d = a - b_acc
                lo, hi = boot_ci(d)
                L.append(
                    f"  {fam:9s} {str(cap):11s} {a.mean():.3f} {n.mean():7.1f}   {100 * d.mean():+5.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] p={wil(d):.3g}   {', '.join(picks)}"
                )
                cv_rows.append(dict(family=fam, cap=cap, acc=a.mean(), ft=n.mean(), d=d, picks=picks, a=a, n=n))

    # ---- 5. decision rules
    if "block" in subj and fams.get("sprt"):
        L += ["", f"Decision rules (non-inferiority margin {100 * args.margin:.1f} pp; Holm over H1-H3)"]
        b_acc, b_ft = val("block", "acc"), val("block", "n_ft")
        a1, n1, p1 = nested(fams["sprt"], None)
        d1 = a1 - b_acc
        cap = 0.7 * np.mean(b_ft)
        a2, n2, p2 = nested(fams["sprt"], cap)
        if np.isnan(a2).any():
            d2, ratio2 = np.full(len(subjects), np.nan), np.nan
        else:
            d2, ratio2 = a2 - b_acc, np.mean(n2) / np.mean(b_ft)
        r3 = resid.get(DEFAULT_SPRT, np.array([np.nan]))
        p_h1 = wilcoxon(d1, alternative="greater").pvalue if wilcoxon is not None else np.nan
        lo2 = boot_ci(d2)[0] if not np.isnan(d2).any() else np.nan
        # H2 as a one-sided test of d2 + margin > 0
        p_h2 = (
            wilcoxon(d2 + args.margin, alternative="greater").pvalue
            if wilcoxon is not None and not np.isnan(d2).any()
            else np.nan
        )
        p_h3 = wilcoxon(r3, alternative="greater").pvalue if wilcoxon is not None and len(r3) > 5 else np.nan
        adj = holm(np.array([p_h1, p_h2, p_h3]))
        h1 = adj[0] < 0.05
        h2 = (adj[1] < 0.05) and (lo2 > -args.margin) and (ratio2 <= 0.7)
        h3 = adj[2] < 0.05
        L += [
            f"  H1 superiority: tuned SPRT - block {100 * d1.mean():+.2f} pp, one-sided p={p_h1:.3g}, Holm {adj[0]:.3g} -> {'YES' if h1 else 'no'}  (picks {p1})",
            f"  H2 efficiency : SPRT capped at {cap:.1f} ft/ses: {100 * np.nanmean(d2):+.2f} pp (lower CI {100 * lo2:+.2f}), "
            f"finetunes {100 * ratio2:.0f}% of block, p={p_h2:.3g}, Holm {adj[1]:.3g} -> {'YES' if h2 else 'no'}  (picks {p2})",
            f"  H3 information: SPRT default residual vs clock {100 * np.nanmean(r3):+.2f} pp, one-sided p={p_h3:.3g}, Holm {adj[2]:.3g} -> {'YES' if h3 else 'no'}",
            f"  => SPRT worth adopting (H1 or (H2 and H3)): {'YES' if (h1 or (h2 and h3)) else 'NO'}",
        ]

    # ---- 6. heterogeneity: SPRT default - block vs start accuracy
    if DEFAULT_SPRT in subj and "block" in subj and "never" in subj:
        st = val("never", "acc")
        d = val(DEFAULT_SPRT, "acc") - val("block", "acc")
        L += ["", "SPRT default - block by subject start accuracy (never)"]
        for lo, hi in [(0, 0.55), (0.55, 0.65), (0.65, 1.01)]:
            sel = (st >= lo) & (st < hi) & ~np.isnan(d)
            if sel.sum():
                L.append(
                    f"  start acc {lo:.2f}-{min(hi, 1):.2f}: {100 * d[sel].mean():+5.2f} pp (n={sel.sum()}), "
                    f"finetunes SPRT {np.nanmean(val(DEFAULT_SPRT, 'n_ft')[sel]):.1f} vs block {np.nanmean(val('block', 'n_ft')[sel]):.1f}"
                )

    text = "\n".join(L)
    open(os.path.join(out, "evaluation.txt"), "w", encoding="utf-8").write(text + "\n")
    with open(os.path.join(out, "conditions.csv"), "w", newline="") as f:
        keys = sorted({k for t in table for k in t}, key=lambda k: list(table[0]).index(k) if k in table[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    with open(os.path.join(out, "subject_conditions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "condition", "family", "acc", "acc_late", "ft_per_session", "trainings_per_session"])
        for c in conds:
            for s in subjects:
                if s in subj[c]:
                    w.writerow(
                        [s, c, family(c)]
                        + [f"{np.mean(subj[c][s][k]):.5f}" for k in ("acc", "acc_late", "n_ft", "n_train")]
                    )
    print(text)


if __name__ == "__main__":
    main()
