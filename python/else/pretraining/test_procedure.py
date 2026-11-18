"""
run_procedure 的規則與切分測試 (不需要 GPU、不跑訓練)。

用法：python python/else/pretraining/test_procedure.py   (或 pytest 這個檔案)
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_procedure as rp
import make_protocol as mp

KEYS = [(k, f"s{k}{i}") for k in range(1, 6) for i in range(3)]          # 15 個驗證位置


def fake(name, accs):
    val = dict(zip(KEYS, accs))
    return dict(tag=name, setting={}, val=val, val_mean=sum(accs) / len(accs),
                test={}, test_mean=0.5, test_trial_mean=0.5, epochs=[1] * 5)


def test_rule_passes():
    inc = fake("inc", [0.60] * 15)
    cand = fake("c", [0.61 + 0.002 * (i % 3) for i in range(15)])     # +1.2 點，每個位置都贏
    r = rp.compare(cand, inc)
    assert r["passed"] and r["p"] < 0.001 and r["wins"] == 15


def test_rule_fails_when_noisy():
    inc = fake("inc", [0.60] * 15)
    cand = fake("c", [0.66] * 8 + [0.55] * 7)          # 平均 +0.87 點但差距忽大忽小
    r = rp.compare(cand, inc)
    assert r["diff"] > 0.005 and not r["passed"] and r["p"] > 0.05


def test_rule_fails_when_worse():
    inc = fake("inc", [0.60] * 15)
    r = rp.compare(fake("c", [0.59] * 15), inc)
    assert not r["passed"] and r["p"] == 1.0


def test_rule_matches_ci():
    import random
    rnd = random.Random(0)
    for _ in range(200):
        inc = fake("inc", [0.6 + rnd.gauss(0, 0.02) for _ in range(15)])
        cand = fake("c", [v + rnd.gauss(0.004, 0.01) for v in inc["val"].values()])
        r = rp.compare(cand, inc)
        assert r["passed"] == (r["ci90"][0] > 0)           # 90% 區間下界 > 0 ⇔ 單尾 p < 0.05


def test_rule_matches_scipy():
    from scipy import stats
    inc = fake("inc", [0.60, 0.55, 0.62, 0.58, 0.61, 0.57, 0.63, 0.59, 0.60, 0.56, 0.64, 0.58, 0.61, 0.57, 0.60])
    cand = fake("c", [0.61, 0.57, 0.62, 0.60, 0.60, 0.58, 0.65, 0.60, 0.61, 0.57, 0.64, 0.60, 0.62, 0.56, 0.62])
    p = stats.ttest_rel(list(cand["val"].values()), list(inc["val"].values()), alternative="greater").pvalue
    assert abs(rp.compare(cand, inc)["p"] - p) < 1e-12


def test_decide_keeps_incumbent_when_none_pass():
    inc = fake("inc", [0.60] * 15)
    winner, rows = rp.decide(inc, [fake("a", [0.62] * 8 + [0.58] * 7), fake("b", [0.59] * 15)])
    assert winner["tag"] == "inc" and not any(r["passed"] for r in rows)


def test_decide_picks_best_passing():
    inc = fake("inc", [0.60] * 15)
    winner, _ = rp.decide(inc, [fake("a", [0.62] * 15), fake("b", [0.63] * 15), fake("big_but_noisy", [0.80] * 5 + [0.55] * 10)])
    assert winner["tag"] == "b"


def test_protocol_file_matches_builder():
    proto = json.load(open(os.path.join(rp.EXP, "protocol.json")))
    assert proto["folds"] == mp.build() and proto["seed"] == mp.SEED
    mp.check(proto["folds"])


def test_fold_command():
    proto = json.load(open(os.path.join(rp.EXP, "protocol.json")))
    f = proto["folds"][0]
    cmd = rp.fold_cmd(rp.variant(rp.BASE, dropout=0.7), f, proto["subjects"], "/x")
    s = " ".join(cmd)
    assert f"--val_subjects {','.join(f['val'])}" in s and f"--test_subjects {','.join(f['test'])}" in s
    assert "--patience 15" in s and "--dropout 0.7" in s and "--lr 1e-3" in s and "--batch_size 64" in s
    assert "--dropout" not in " ".join(rp.fold_cmd(rp.BASE, f, proto["subjects"], "/x"))


def test_setting_from_tag_roundtrip():
    for s in [rp.BASE, rp.variant(rp.BASE, model="EEGNetv4", lr="3e-4", bs=128, dropout=0.5, wd="1e-3", stride=50, seed=43)]:
        assert rp.setting_from_tag(rp.tag(s)) == s


def test_regs_per_model():
    assert rp.regs_for("SCCNet")[0] == dict(dropout=0.7) and rp.regs_for("EEGNetv4")[0] == dict(dropout=0.5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
