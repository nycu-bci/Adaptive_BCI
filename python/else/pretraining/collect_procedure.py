"""
彙整嚴格流程 (run_procedure) 的結果。

輸出 (procedure/ 底下)：
  results.csv   每個 run × 受試者 × 角色 (val / test) 一列：視窗準確率、測試受試者另有 trial 多數決準確率
  summary.txt   每個 stage：各設定的驗證平均 (決策用) 與測試平均 (只報告)，規則比較結果、贏家，
                以及測試受試者依組別 (HIGH / LOW / LH / HL，22 通道受試者內準確率中位數分組) 的平均
流程沒跑完也能執行，只彙整已完成的設定。

用法：python python/else/pretraining/collect_procedure.py
"""
import os
import csv
import glob
import json

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
EXP = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "procedure")
GROUPS = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "group_pool", "groups.json")
TEST_GROUPS = ["HIGH", "LOW", "LH", "HL"]


def load_runs(proto):
    rows = []
    for d in sorted(glob.glob(os.path.join(EXP, "runs", "*"))):
        setting = os.path.basename(d)
        for f in proto["folds"]:
            path = os.path.join(d, f"fold{f['fold']}.pth")
            if not os.path.exists(path):
                continue
            m = torch.load(path, map_location="cpu", weights_only=False)["pretrain_meta"]
            common = dict(setting=setting, fold=f["fold"], best_epoch=m["best_epoch"] + 1,
                          epochs_run=m["epochs_run"], val_loss=m["best_val_loss"])
            for s in f["val"]:
                rows.append(dict(common, subject=s, role="val", acc=m["heldout_per_subject_acc"][s], trial_acc=""))
            for s in f["test"]:
                rows.append(dict(common, subject=s, role="test", acc=m["test_per_subject"][s],
                                 trial_acc=m["test_trial_per_subject"][s]))
    return rows


def main():
    proto = json.load(open(os.path.join(EXP, "protocol.json")))
    groups = json.load(open(GROUPS))["groups"]
    group_of = {s: g for g, ss in groups.items() for s in ss}
    rows = load_runs(proto)
    for r in rows:
        r["group"] = group_of[r["subject"]]
    rows.sort(key=lambda r: (r["setting"], r["fold"], r["role"], int(r["subject"])))
    with open(os.path.join(EXP, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "fold", "subject", "group", "role", "acc", "trial_acc",
                                          "best_epoch", "epochs_run", "val_loss"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    by_setting = {}
    for r in rows:
        by_setting.setdefault(r["setting"], []).append(r)
    complete = {s: rs for s, rs in by_setting.items() if len({r["fold"] for r in rs}) == len(proto["folds"])}

    def test_line(setting):
        rs = [r for r in complete[setting] if r["role"] == "test"]
        cells = []
        for g in TEST_GROUPS:
            v = [r["acc"] for r in rs if r["group"] == g]
            cells.append(f"{g} {np.mean(v) * 100:5.2f}")
        return "  ".join(cells)

    out = [f"Strict Phase 1 procedure: {len(complete)} settings complete. "
           f"Decisions use VALIDATION only (>= +0.5 points and >= 10 of 15 validation slots); "
           f"test results are reported, never used to choose."]
    dec_path = os.path.join(EXP, "decisions.json")
    decisions = json.load(open(dec_path)) if os.path.exists(dec_path) else {}
    for stage, d in decisions.items():
        out.append(f"\n=== {stage} ===  winner: {d['winner']['tag']}")
        if "seed_sd_val" in d:
            out.append(f"seed-to-seed sd (3 seeds): validation {d['seed_sd_val'] * 100:.2f} points, "
                       f"test {d['seed_sd_test'] * 100:.2f} points")
        listed = d.get("all") or d.get("seeds") or [d["winner"]]
        out.append(f"{'setting':48s} {'val':>6s} {'test':>6s} {'trial':>6s}  chosen epochs | test by group")
        for x in sorted(listed, key=lambda x: -x["val_mean"]):
            mark = " <- winner" if x["tag"] == d["winner"]["tag"] else ""
            out.append(f"{x['tag']:48s} {x['val_mean'] * 100:6.2f} {x['test_mean'] * 100:6.2f} "
                       f"{x['test_trial_mean'] * 100:6.2f}  {x['epochs']} | {test_line(x['tag'])}{mark}")
        for c in d.get("comparisons", []):
            out.append(f"   vs incumbent: {c['tag']:48s} {c['diff'] * 100:+.2f} points, "
                       f"better in {c['wins']}/15 slots -> {'PASS' if c['passed'] else 'no'}")
        if "best_regularization_per_base" in d:
            out.append(f"   best regularization per base (interaction check): {d['best_regularization_per_base']}")
        if "model_file" in d:
            out.append(f"   final model: {d['model_file']} ({d['epochs']} epochs, all 22 subjects)")
    text = "\n".join(out)
    open(os.path.join(EXP, "summary.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
