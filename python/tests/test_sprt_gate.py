# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_sprt_gate.py
"""
Unit tests for main.Utils.sprt_gate.SprtGate (no Unity / EEG / torch needed).
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from main.Utils.sprt_gate import SprtGate


def run(gate, seq):
    return [gate.update(c) for c in seq]


def test_constants_set_b():
    g = SprtGate(0.5, 0.75, 0.05, 0.2)
    assert abs(g.win - math.log(1.5)) < 1e-12 and abs(g.win - 0.405465) < 1e-6
    assert abs(g.loss - math.log(0.5)) < 1e-12 and abs(g.loss + 0.693147) < 1e-6
    assert abs(g.upper - math.log(16)) < 1e-12 and abs(g.upper - 2.772589) < 1e-6
    assert abs(g.lower - math.log(0.2 / 0.95)) < 1e-12 and abs(g.lower + 1.558145) < 1e-6
    assert g.score == 0.0


def test_three_wrong_in_a_row_finetunes():
    g = SprtGate(0.5, 0.75, 0.05, 0.2)
    assert run(g, [False, False]) == [None, None] and abs(g.score + 1.386294) < 1e-6   # -1.386 > -1.558
    assert g.update(False) == "finetune" and g.score == 0.0                             # -2.079 <= -1.558
    assert abs(g.last_score + 2.079442) < 1e-6


def test_seven_correct_in_a_row_is_good():
    g = SprtGate(0.5, 0.75, 0.05, 0.2)
    assert run(g, [True] * 6) == [None] * 6                                              # 6 x 0.405 = 2.43 < 2.77
    assert g.update(True) == "good" and g.score == 0.0                                   # 7 x 0.405 = 2.84
    assert abs(g.last_score - 7 * math.log(1.5)) < 1e-9


def test_mixed_sequence_and_set_a():
    g = SprtGate(0.5, 0.75, 0.05, 0.2)
    out = run(g, [True] * 8 + [False] * 2)        # stops at the 7th correct
    assert out.index("good") == 6 and out.count("good") == 1
    g = SprtGate(0.5, 0.75, 0.05, 0.2)
    run(g, [True, True, False, True, True, True, False, True, True, True])   # worked example in the explanation
    assert abs(g.score - (8 * math.log(1.5) + 2 * math.log(0.5))) < 1e-9 and abs(g.score - 1.857426) < 1e-6
    a = SprtGate(0.5, 0.75, 0.05, 0.05)           # set A: +-2.944, 8 correct / 5 wrong
    assert run(a, [True] * 8)[-1] == "good" and run(a, [True] * 7)[-1] is None
    a.reset()
    assert run(a, [False] * 5)[-1] == "finetune" and run(a, [False] * 4)[-1] is None


def test_none_does_not_move_score():
    g = SprtGate()
    g.update(False)
    s = g.score
    assert g.update(None) is None and g.score == s and g.step(None) == 0.0
    assert g.last_score == s


def test_reset_after_decision_and_manual_reset():
    g = SprtGate()
    run(g, [False] * 3)
    assert g.score == 0.0
    g.update(True)
    g.reset()
    assert g.score == 0.0


def test_invalid_parameters_raise():
    for kwargs in (dict(p0=0.75, p1=0.5), dict(p0=0.0), dict(p1=1.0), dict(alpha=0.0), dict(beta=1.0),
                   dict(alpha=0.6, beta=0.5)):
        try:
            SprtGate(**kwargs)
            raise AssertionError(f"should fail: {kwargs}")
        except ValueError:
            pass


def test_error_rates_within_wald_bounds():
    """Monte Carlo: P(good | acc = p0) <= alpha / (1 - beta), P(finetune | acc = p1) <= beta / (1 - alpha)."""
    rng = np.random.default_rng(0)

    def first_decision(p, n=20000):
        out = {"good": 0, "finetune": 0}
        for _ in range(n):
            g = SprtGate(0.5, 0.75, 0.05, 0.2)
            while True:
                d = g.update(bool(rng.random() < p))
                if d:
                    out[d] += 1
                    break
        return {k: v / n for k, v in out.items()}

    at_p0, at_p1 = first_decision(0.5), first_decision(0.75)
    assert at_p0["good"] <= 0.05 / 0.8 + 0.005, at_p0
    assert at_p1["finetune"] <= 0.2 / 0.95 + 0.01, at_p1
    assert at_p0["finetune"] > 0.9 and at_p1["good"] > 0.75


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t:.1f}s)")
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
