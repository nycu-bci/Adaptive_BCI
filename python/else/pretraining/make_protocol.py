"""
寫出 Phase 1 嚴格流程的固定切分 protocol.json：5 個測試 fold，每個 fold 隨機抽 3 位驗證受試者 (固定 seed，同一人最多當一次驗證)。
驗證受試者只從該 fold 的「訓練端」受試者抽 (不含測試 fold)，所有 stage 共用同一份切分。

用法：python python/else/pretraining/make_protocol.py
輸出：Aden/pretrained_result/experiments/procedure/protocol.json (已存在且內容不同時拒絕覆蓋)
"""
import os
import json
import random
import argparse
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(os.path.dirname(_script_dir))
DEFAULT_OUT = os.path.join(_python_dir, "Aden", "pretrained_result", "experiments", "procedure", "protocol.json")

SUBJECTS = ["35", "38", "40", "41", "42", "43", "44", "47", "48", "50", "51",
            "52", "54", "55", "57", "58", "63", "64", "65", "68", "69", "70"]   # 不含 37、45
TEST_FOLDS = [["40", "43", "51", "52", "69"], ["38", "41", "42", "44", "55"], ["35", "50", "54", "65"],
              ["47", "57", "58", "64"], ["48", "63", "68", "70"]]                 # 與 fullcov / phase1_redo 相同
N_VAL = 3
SEED = 2026


def build(seed=SEED):
    """隨機抽驗證受試者，但同一人最多當一次驗證 (15 個位置 = 15 位不同受試者)；不使用任何表現資訊。"""
    rng = random.Random(seed)
    used, folds = set(), []
    for k, test in enumerate(TEST_FOLDS, 1):
        rest = [s for s in SUBJECTS if s not in test]
        val = sorted(rng.sample([s for s in rest if s not in used], N_VAL), key=int)
        used.update(val)
        train = [s for s in rest if s not in val]
        folds.append(dict(fold=k, test=test, val=val, train=train))
    return folds


def check(folds):
    assert sorted(sum((f["test"] for f in folds), []), key=int) == SUBJECTS, "每位受試者恰好測試一次"
    for f in folds:
        assert not set(f["val"]) & set(f["test"]) and not set(f["train"]) & set(f["test"])
        assert not set(f["val"]) & set(f["train"]) and len(f["val"]) == N_VAL
        assert sorted(f["test"] + f["val"] + f["train"], key=int) == SUBJECTS
    all_val = sum((f["val"] for f in folds), [])
    assert len(all_val) == len(set(all_val)), "每位受試者最多當一次驗證"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    folds = build()
    check(folds)
    proto = dict(seed=SEED, n_val=N_VAL, subjects=SUBJECTS, folds=folds)
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        if {k: old.get(k) for k in proto} != proto:
            raise SystemExit(f"❌ {args.out} 已存在且內容不同，拒絕覆蓋 (切分必須固定)")
        print(f"protocol 未變：{args.out}")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    proto["created"] = datetime.now().isoformat(timespec="seconds")
    json.dump(proto, open(args.out, "w"), indent=2)
    for f in folds:
        print(f"fold {f['fold']}: test {f['test']}  val {f['val']}  train {len(f['train'])}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
