# python python/analysis/gate_study/make_queue.py round2 > <out>/queue.txt
"""Work queues for simulate_gates.py --queue (one 'sub:ses:condition[@seeds]' per line, session-major per phase)."""

import sys

from simulate_gates import SUBJECTS

SESSIONS = [(s, ses) for s in SUBJECTS for ses in ("s1", "s2")]
SPRT_DEFAULT = "sprt:0.65,0.85,0.05,0.2"
PHASES = {
    "round2": [
        # 1. 基準、預算對照 (固定間隔、隨機)、調過門檻的 block gate、作弊上限；5 seeds
        (
            [
                "never",
                "block",
                "always",
                SPRT_DEFAULT,
                "sprt:0.5,0.75,0.05,0.2",
                "every:6",
                "every:8",
                "every:12",
                "every:20",
                "random:0.05",
                "random:0.1",
                "random:0.15",
                "block:0.65",
                "block:0.85",
                "oracle",
            ],
            None,
        ),
        # 2. SPRT 參數網格；3 seeds
        (
            [
                # 位置 (p0, p1 一起移，alpha .05 beta .2)
                "sprt:0.5,0.7,0.05,0.2",
                "sprt:0.55,0.75,0.05,0.2",
                "sprt:0.6,0.8,0.05,0.2",
                "sprt:0.7,0.9,0.05,0.2",
                "sprt:0.75,0.95,0.05,0.2",
                # 寬度 (中心約 0.75)
                "sprt:0.7,0.8,0.05,0.2",
                "sprt:0.6,0.9,0.05,0.2",
                "sprt:0.55,0.95,0.05,0.2",
                # alpha
                "sprt:0.65,0.85,0.01,0.2",
                "sprt:0.65,0.85,0.1,0.2",
                "sprt:0.65,0.85,0.2,0.2",
                # beta
                "sprt:0.65,0.85,0.05,0.05",
                "sprt:0.65,0.85,0.05,0.1",
                "sprt:0.65,0.85,0.05,0.3",
                "sprt:0.65,0.85,0.05,0.4",
                # 同預算下選擇性最高 (sprt_design_space.py)
                "sprt:0.7,0.8,0.01,0.3",
                "sprt:0.7,0.85,0.01,0.4",
                "sprt:0.6,0.85,0.01,0.4",
                "sprt:0.65,0.9,0.01,0.3",
            ],
            "0,1,2",
        ),
    ],
}

if __name__ == "__main__":
    for conds, seeds in PHASES[sys.argv[1]]:
        for sub, ses in SESSIONS:
            for c in conds:
                print(f"{sub}:{ses}:{c}" + (f"@{seeds}" if seeds else ""))
