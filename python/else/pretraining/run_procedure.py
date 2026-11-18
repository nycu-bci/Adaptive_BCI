"""
Phase 1 嚴格流程的總控：依 protocol.json 的固定切分，依序跑 stage 1-6，每個 stage 用事先定好的規則自動決定贏家。

規則 (只看驗證，不看測試)：候選設定要贏過目前最佳，必須在 15 個驗證位置 (5 fold × 3 位) 上
  做單尾配對 t 檢定 (候選 > 目前最佳) 得到 p < 0.05。
多個候選通過時取驗證平均最高者；沒有通過就保留目前最佳。
(2026-10-07 之前的規則是「平均 >= +0.5 個百分點且 15 個中至少 10 個比較好」，舊結果存在 decisions_rule_0.5_10of15*.json。)
測試受試者的結果只記錄、報告，不參與任何選擇。

stage 1 baseline (seed 42/43/44) → 2 架構 × lr → 3 lr × batch → 4 正則化 (前兩名) → 5 stride → 6 全 22 人最終模型
每個設定的 5 個 fold 存在 procedure/runs/<設定>/fold<k>.pth；已存在就跳過，所以中斷後可直接重跑續跑。

用法：python python/else/pretraining/run_procedure.py            (跑完全部)
      python python/else/pretraining/run_procedure.py --dry_run  (只跑 stage 1 的 fold 1，檢查用)
"""
import os
import sys
import json
import time
import argparse
import subprocess
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy import stats

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
PY = sys.executable
TRAIN = os.path.join(_script_dir, "pretrain_cross_subject.py")
EXP = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "procedure")
WEIGHTS = os.path.join(_python_dir, "Aden", "pretrained_result", "model_weights")

ALPHA = 0.05              # 單尾配對 t 檢定的門檻
PATIENCE = 15
MAX_EPOCHS = 100

BASE = dict(model="SCCNet", lr="1e-3", bs=64, dropout=None, wd="0", stride=100, seed=42)
MODELS = ["SCCNet", "ShallowFBCSPNet", "EEGNetv4"]
ARCH_LRS = ["1e-3", "3e-4", "1e-4"]
GRID_LRS = ["3e-3", "1e-3", "3e-4", "1e-4", "3e-5"]
GRID_BS = [64, 128, 256]
REG_DROPOUT = {"SCCNet": 0.7, "ShallowFBCSPNet": 0.7, "EEGNetv4": 0.5}   # 比各模型預設 (0.5 / 0.5 / 0.25) 高一階


def regs_for(model):
    return [dict(dropout=REG_DROPOUT[model]), dict(wd="1e-3"), dict(wd="1e-2")]
STRIDES = [50, 250]


def tag(s):
    do = "def" if s["dropout"] is None else s["dropout"]
    return f"{s['model']}_lr{s['lr']}_bs{s['bs']}_do{do}_wd{s['wd']}_st{s['stride']}_s{s['seed']}"


def setting_from_tag(t):
    """tag() 的反函數。"""
    model, lr, bs, do, wd, st, seed = t.split("_")
    return dict(model=model, lr=lr[2:], bs=int(bs[2:]), dropout=None if do[2:] == "def" else float(do[2:]),
                wd=wd[2:], stride=int(st[2:]), seed=int(seed[1:]))


def variant(s, **kw):
    return {**s, **kw}


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(EXP, "driver.log"), "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 訓練
def fold_cmd(s, fold, all_ids, out_dir):
    k = fold["fold"]
    cmd = [PY, TRAIN, "--ids", ",".join(all_ids), "--val_subjects", ",".join(fold["val"]),
           "--test_subjects", ",".join(fold["test"]), "--model", s["model"], "--stride", str(s["stride"]),
           "--epochs", str(MAX_EPOCHS), "--patience", str(PATIENCE), "--seed", str(s["seed"]),
           "--lr", s["lr"], "--batch_size", str(s["bs"]), "--weight_decay", s["wd"],
           "--select_by", "val_loss", "--no_plot",
           "--work_dir", os.path.join(out_dir, f".work_fold{k}"),
           "--out", os.path.join(out_dir, f"fold{k}.pth"), "--curve_dir", out_dir]
    if s["dropout"] is not None:
        cmd += ["--dropout", str(s["dropout"])]
    return cmd


def run_job(job):
    s, fold, all_ids = job
    out_dir = os.path.join(EXP, "runs", tag(s))
    os.makedirs(out_dir, exist_ok=True)
    k = fold["fold"]
    out = os.path.join(out_dir, f"fold{k}.pth")
    if os.path.exists(out):
        return True
    t0 = time.time()
    with open(os.path.join(out_dir, f"fold{k}.log"), "w") as f:
        ok = subprocess.run(fold_cmd(s, fold, all_ids, out_dir), stdout=f, stderr=subprocess.STDOUT,
                            cwd=_python_dir).returncode == 0
    subprocess.run(["rm", "-rf", os.path.join(out_dir, f".work_fold{k}")])
    log(f"{'DONE ' if ok else 'FAILED'} {tag(s)} fold{k} ({(time.time() - t0) / 60:.1f} min)")
    return ok


def ensure(settings, proto, workers):
    """跑完這些設定的全部 fold (已存在的跳過)；有任何失敗就停止整個流程。"""
    jobs = [(s, f, proto["subjects"]) for s in settings for f in proto["folds"]]
    todo = [j for j in jobs if not os.path.exists(os.path.join(EXP, "runs", tag(j[0]), f"fold{j[1]['fold']}.pth"))]
    if todo:
        log(f"running {len(todo)} jobs ({len(settings)} settings), {workers} in parallel")
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(run_job, todo))
    if not all(results):
        raise SystemExit("❌ 有 run 失敗，流程停止 (看 driver.log 與該 run 的 log)；修好後重跑會從斷點繼續")


# ---------------------------------------------------------------- 評分與規則
def score(s, proto):
    """一個設定的 15 個驗證位置與 22 位測試受試者的準確率。"""
    val, test, test_trial, epochs = {}, {}, {}, []
    for f in proto["folds"]:
        m = torch.load(os.path.join(EXP, "runs", tag(s), f"fold{f['fold']}.pth"),
                       map_location="cpu", weights_only=False)["pretrain_meta"]
        for subj in f["val"]:
            val[(f["fold"], subj)] = m["heldout_per_subject_acc"][subj]
        test.update(m["test_per_subject"])
        test_trial.update(m["test_trial_per_subject"])
        epochs.append(m["best_epoch"] + 1)
    return dict(tag=tag(s), setting=s, val=val, val_mean=float(np.mean(list(val.values()))),
                test=test, test_mean=float(np.mean(list(test.values()))),
                test_trial_mean=float(np.mean(list(test_trial.values()))), epochs=epochs)


def compare(cand, inc):
    """候選 vs 目前最佳，在驗證位置上做單尾配對 t 檢定。
    回傳 dict：diff (平均差)、wins (贏的位置數)、p、ci90 (平均差的 90% 信賴區間；下界 > 0 等同 p < 0.05)、passed。"""
    keys = sorted(inc["val"])
    assert keys == sorted(cand["val"]), "驗證位置不一致"
    d = np.array([cand["val"][k] - inc["val"][k] for k in keys])
    diff, wins = float(d.mean()), int((d > 0).sum())
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:                                    # 所有位置差距相同，t 無定義
        p, half = (0.0 if diff > 0 else 1.0), 0.0
    else:
        p = float(stats.t.sf(diff / se, len(d) - 1))
        half = float(stats.t.ppf(1 - ALPHA, len(d) - 1) * se)
    return dict(diff=diff, wins=wins, p=p, ci90=[diff - half, diff + half], passed=bool(p < ALPHA))


def decide(inc, cands):
    """回傳 (贏家, 比較紀錄)。通過規則的候選中取驗證平均最高者；都沒通過就保留目前最佳。"""
    rows, passed = [], []
    for c in cands:
        if c["tag"] == inc["tag"]:
            continue
        r = compare(c, inc)
        rows.append(dict(tag=c["tag"], val_mean=c["val_mean"], **r,
                         test_mean=c["test_mean"], test_trial_mean=c["test_trial_mean"]))
        if r["passed"]:
            passed.append(c)
    winner = max(passed, key=lambda c: c["val_mean"]) if passed else inc
    return winner, rows


def brief(sc):
    return dict(tag=sc["tag"], val_mean=sc["val_mean"], test_mean=sc["test_mean"],
                test_trial_mean=sc["test_trial_mean"], epochs=sc["epochs"])


# ---------------------------------------------------------------- 主流程
def tune_stages(inc, proto, workers, save, prefix=""):
    """stage 3-5 (lr × batch → 正則化前兩名 → stride)，從 inc 開始；回傳最後的贏家。"""
    # stage 3: lr x batch on the winning model
    grid = [variant(inc["setting"], lr=lr, bs=bs) for lr in GRID_LRS for bs in GRID_BS]
    ensure(grid, proto, workers)
    sc_grid = [score(s, proto) for s in grid]
    winner, rows = decide(inc, sc_grid)
    pool = {c["tag"]: c for c in sc_grid + [inc]}
    runner_up = max((c for t, c in pool.items() if t != winner["tag"]), key=lambda c: c["val_mean"])
    save(prefix + "stage3_lr_batch", dict(incumbent=brief(inc), all=[brief(x) for x in sc_grid], comparisons=rows,
                                          winner=brief(winner), runner_up=brief(runner_up)))
    inc, second = winner, runner_up

    # stage 4: regularization on the top two
    reg_list = regs_for(inc["setting"]["model"])
    regs = [variant(b["setting"], **r) for b in (inc, second) for r in reg_list]
    ensure(regs, proto, workers)
    sc_reg = [score(s, proto) for s in regs]
    winner, rows = decide(inc, sc_reg)
    best_on = {b["tag"]: max(sc_reg[i * 3:(i + 1) * 3], key=lambda c: c["val_mean"])["tag"]
               for i, b in enumerate((inc, second))}
    save(prefix + "stage4_regularization", dict(incumbent=brief(inc), second=brief(second),
                                                all=[brief(x) for x in sc_reg], comparisons=rows,
                                                best_regularization_per_base=best_on, winner=brief(winner)))
    inc = winner

    # stage 5: stride (stride 50 doubles memory, so at most 2 in parallel)
    strides = [variant(inc["setting"], stride=st) for st in STRIDES]
    ensure(strides, proto, min(workers, 2))
    sc_st = [score(s, proto) for s in strides]
    winner, rows = decide(inc, sc_st)
    save(prefix + "stage5_stride", dict(incumbent=brief(inc), all=[brief(x) for x in sc_st], comparisons=rows,
                                        winner=brief(winner)))
    return winner


def train_final(inc, proto, save, logname, prefix=""):
    """stage 6: 用全部 22 人訓練最終模型，epoch 數 = 5 個 fold 選到的 epoch 中位數。"""
    s = inc["setting"]
    n_ep = int(statistics.median(inc["epochs"]))
    out = os.path.join(WEIGHTS, f"final_procedure_{tag(s)}_e{n_ep}.pth")
    cmd = [PY, TRAIN, "--ids", ",".join(proto["subjects"]), "--no_val", "--model", s["model"],
           "--stride", str(s["stride"]), "--epochs", str(n_ep), "--seed", str(s["seed"]), "--lr", s["lr"],
           "--batch_size", str(s["bs"]), "--weight_decay", s["wd"], "--no_plot",
           "--work_dir", os.path.join(EXP, ".work_final"), "--out", out, "--curve_dir", EXP]
    if s["dropout"] is not None:
        cmd += ["--dropout", str(s["dropout"])]
    if not os.path.exists(out):
        with open(os.path.join(EXP, logname), "w") as f:
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=_python_dir).returncode:
                raise SystemExit(f"❌ 最終模型訓練失敗 (看 {logname})")
    save(prefix + "stage6_final", dict(winner=brief(inc), epochs=n_ep, model_file=out))


def tune_other_model(model, proto, workers):
    """對另一個架構跑同樣的 stage 3-5；起點 = 它在 stage 2 依驗證選出的最佳 lr。
    調完後用同一條規則和 SCCNet 的最終設定比，通過才訓練它的最終模型。結果存 decisions_<model>.json。"""
    main_dec = json.load(open(os.path.join(EXP, "decisions.json")))
    dec_path = os.path.join(EXP, f"decisions_{model}.json")
    decisions = {}

    def save(stage, d):
        decisions[stage] = d
        json.dump(decisions, open(dec_path, "w"), indent=2, default=str)
        log(f"[{model}] {stage}: winner {d['winner']['tag']} (val {d['winner']['val_mean'] * 100:.2f}%)")

    log(f"=== tuning {model} start ===")
    start = [variant(BASE, model=model, lr=lr) for lr in ARCH_LRS]
    ensure(start, proto, workers)
    inc = max((score(s, proto) for s in start), key=lambda c: c["val_mean"])
    save("stage2_start", dict(winner=brief(inc), note="best of the stage-2 learning rates for this model"))
    inc = tune_stages(inc, proto, workers, save)

    sccnet = score(setting_from_tag(main_dec["stage6_final"]["winner"]["tag"]), proto)
    r = compare(inc, sccnet)
    save("vs_sccnet", dict(sccnet=brief(sccnet), winner=brief(inc if r["passed"] else sccnet),
                           comparison=dict(tag=inc["tag"], **r)))
    if r["passed"]:
        train_final(inc, proto, save, f"final_model_{model}.log")
    log(f"=== tuning {model} complete ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--tune_model", default=None, choices=["EEGNetv4", "ShallowFBCSPNet"],
                    help="對這個架構跑 stage 3-5，並和 SCCNet 最終設定比較 (結果存 decisions_<model>.json)")
    args = ap.parse_args()
    proto = json.load(open(os.path.join(EXP, "protocol.json")))
    if args.tune_model:
        tune_other_model(args.tune_model, proto, args.workers)
        return
    dec_path = os.path.join(EXP, "decisions.json")
    decisions = {}

    def save(stage, d):
        decisions[stage] = d
        json.dump(decisions, open(dec_path, "w"), indent=2, default=str)
        log(f"{stage}: winner {d['winner']['tag']} (val {d['winner']['val_mean'] * 100:.2f}%)")

    if args.dry_run:
        ok = run_job((BASE, proto["folds"][0], proto["subjects"]))
        log(f"dry run {'ok' if ok else 'FAILED'}")
        return

    log("=== procedure start ===")
    # stage 1: baseline + seed noise
    seeds = [variant(BASE, seed=s) for s in (42, 43, 44)]
    ensure(seeds, proto, args.workers)
    sc = [score(s, proto) for s in seeds]
    inc = sc[0]
    save("stage1_baseline", dict(
        winner=brief(inc), seeds=[brief(x) for x in sc],
        seed_sd_val=float(np.std([x["val_mean"] for x in sc], ddof=1)),
        seed_sd_test=float(np.std([x["test_mean"] for x in sc], ddof=1))))

    # stage 2: architecture, each at its own best lr (by validation)
    arch = [variant(BASE, model=m, lr=lr) for m in MODELS for lr in ARCH_LRS]
    ensure(arch, proto, args.workers)
    sc_arch = {tag(s): score(s, proto) for s in arch}
    reps = [max((sc_arch[tag(variant(BASE, model=m, lr=lr))] for lr in ARCH_LRS), key=lambda c: c["val_mean"])
            for m in MODELS]
    winner, rows = decide(inc, reps)
    save("stage2_architecture", dict(incumbent=brief(inc), all=[brief(x) for x in sc_arch.values()],
                                     representatives=[r["tag"] for r in reps], comparisons=rows,
                                     winner=brief(winner)))
    inc = winner

    inc = tune_stages(inc, proto, args.workers, save)
    train_final(inc, proto, save, "final_model.log")
    log("=== procedure complete ===")


if __name__ == "__main__":
    main()
