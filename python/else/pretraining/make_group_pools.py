"""
依 22 通道受試者內準確率 (每個 session 的中位數) 把受試者分組，並產生 leave-one-subject-out 的訓練池工作清單。

組別：HIGH (兩個 session 都 >= 中位數)、LOW (都 < 中位數)、LH (s1 低 s2 高)、HL (s1 高 s2 低)。
每一輪留一位測試受試者；用其餘受試者建 ALL / HIGH / LOW / RAND0-2 四種訓練池，
每個池固定抽 2 位驗證受試者 (early stopping 用)，其餘訓練；每個池 × 每個學習率一行工作。
LH / HL 只當測試受試者。

用法：python python/else/pretraining/make_group_pools.py
輸出：Aden/pretrained_result/experiments/group_pool/{groups.json, jobs.txt}
jobs.txt 每行：<pool> <lr> <test> <ids (訓練+驗證+測試)> <驗證>
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, os.path.join(_python_dir, "else", "analysis"))
from analyze_metrics_and_plot import parse_offline_session_log

SUBJECTS = ["35", "38", "40", "41", "42", "43", "44", "47", "48", "50", "51",
            "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"]   # 不含 37、45
LRS = ["3e-3", "1e-3", "3e-4", "1e-4", "3e-5"]
N_VAL = 2
N_RAND = 3


def sorted_ids(ids):
    return sorted(ids, key=int)


def make_groups(acc):
    m1 = float(np.median([acc[s]["s1"] for s in SUBJECTS]))
    m2 = float(np.median([acc[s]["s2"] for s in SUBJECTS]))
    groups = {"HIGH": [], "LOW": [], "LH": [], "HL": []}
    for s in SUBJECTS:
        hi1, hi2 = acc[s]["s1"] >= m1, acc[s]["s2"] >= m2
        key = "HIGH" if hi1 and hi2 else "LOW" if not hi1 and not hi2 else "LH" if hi2 else "HL"
        groups[key].append(s)
    return m1, m2, groups


def make_jobs(groups):
    jobs = []
    for test in SUBJECTS:
        others = [s for s in SUBJECTS if s != test]
        high = [s for s in groups["HIGH"] if s != test]
        pools = {"ALL": others, "HIGH": high, "LOW": [s for s in groups["LOW"] if s != test]}
        for d in range(N_RAND):
            pools[f"RAND{d}"] = random.Random(1000 * d + int(test)).sample(others, len(high))
        for pi, (name, pool) in enumerate(pools.items()):
            val = random.Random(100 * int(test) + pi).sample(sorted_ids(pool), N_VAL)
            ids = sorted_ids(pool + [test])
            for lr in LRS:
                jobs.append(dict(pool=name, lr=lr, test=test, ids=ids, val=sorted_ids(val)))
    return jobs


def check(jobs, groups):
    assert len(jobs) == len(SUBJECTS) * (3 + N_RAND) * len(LRS)
    for j in jobs:
        train = [s for s in j["ids"] if s not in j["val"] and s != j["test"]]
        assert j["test"] in j["ids"] and j["test"] not in j["val"] and j["test"] not in train
        assert set(j["val"]) <= set(j["ids"]) and len(j["val"]) == N_VAL and train
        if j["pool"] in ("HIGH", "LOW"):
            assert set(j["ids"]) - {j["test"]} == set(groups[j["pool"]]) - {j["test"]}
        if j["pool"].startswith("RAND"):
            assert len(j["ids"]) - 1 == len([s for s in groups["HIGH"] if s != j["test"]])
    # 同一個池在不同學習率下，驗證受試者相同
    by = {}
    for j in jobs:
        by.setdefault((j["pool"], j["test"]), set()).add((tuple(j["ids"]), tuple(j["val"])))
    assert all(len(v) == 1 for v in by.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(_python_dir, "training_log_20260416_22_all.txt"))
    ap.add_argument("--out_dir", default=os.path.join(_python_dir, "Aden", "pretrained_result",
                                                      "experiments", "group_pool"))
    args = ap.parse_args()

    acc = parse_offline_session_log(args.log)
    missing = [s for s in SUBJECTS if not {"s1", "s2"} <= set(acc.get(s, {}))]
    if missing:
        raise SystemExit(f"❌ {args.log} 缺少這些受試者的 s1/s2 準確率: {missing}")
    m1, m2, groups = make_groups(acc)
    jobs = make_jobs(groups)
    check(jobs, groups)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "groups.json"), "w") as f:
        json.dump(dict(created=datetime.now().isoformat(timespec="seconds"), source=args.log,
                       median_s1=m1, median_s2=m2, groups=groups,
                       accuracy={s: acc[s] for s in SUBJECTS}, lrs=LRS, n_val=N_VAL, n_rand=N_RAND),
                  f, indent=2)
    with open(os.path.join(args.out_dir, "jobs.txt"), "w") as f:
        for j in jobs:
            f.write(f"{j['pool']} {j['lr']} {j['test']} {','.join(j['ids'])} {','.join(j['val'])}\n")

    print(f"medians s1 {m1:.3f}  s2 {m2:.3f}")
    for k, v in groups.items():
        print(f"  {k:4s} ({len(v)}): {v}")
    print(f"{len(jobs)} jobs -> {os.path.join(args.out_dir, 'jobs.txt')}")


if __name__ == "__main__":
    main()
