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
    # 穩健性：新權重延遲 2 個 trial 上線 (--delay 2)；含時鐘曲線的點，才能在同樣延遲下算殘差；3 seeds
    "round3": [
        (
            [
                "never",
                "block",
                "always",
                "every:8",
                "every:12",
                "every:20",
                SPRT_DEFAULT,
                "sprt:0.5,0.75,0.05,0.2",
                "sprt:0.6,0.85,0.01,0.4",
            ],
            "0,1,2",
        ),
    ],
    # 探索：背景驗證 (shadow) 才換上新模型；SPRT 延到 block 結束 (診斷)。延遲 0，session 綁定沿用第二輪的主機；5 seeds
    "round4": [
        (["shadow:4,0", "shadow:8,0", "shadow:16,0", "shadow:16,1", "sprtb:0.65,0.85,0.05,0.2"], None),
    ],
    # 作弊上限的敏感度：oracle 偷看的 trial 數 n (3 seeds)；誠實上限 oraclesplit：只用奇數 trial 決定、偶數 trial 評分
    # (5 seeds；_48 = 跟 oracle 同樣的時間範圍，_96 = 跟 oracle 同樣用 48 個 trial 決定)。延遲 0
    "round5": [
        (["oraclesplit:48", "oraclesplit:96"], None),  # 主要結果先跑
        (["oracle:8", "oracle:16", "oracle:24", "oracle", "oracle:96"], "0,1,2"),
    ],
    # SPRT 加最少間隔：觸發時距上次微調不足 k 個 trial 就先記著，滿 k 再做 (預設 SPRT 參數)。延遲 0；session 綁定沿用第二輪；
    # 5 seeds。判定規則寫在 round6_sprtk_prereg.txt (跑之前)
    "round6": [
        ([f"sprtk:{k},{SPRT_DEFAULT.split(':')[1]}" for k in (4, 8, 12)], None),
    ],
    # 同 round 6，但 SPRT 參數改成 0.75 / 0.95 / 0.05 / 0.2 (觸發較多)；session 綁定沿用第二輪；5 seeds。
    # 判定規則寫在 round7_sprtk_hi_prereg.txt (跑之前)
    "round7": [
        ([f"sprtk:{k},0.75,0.95,0.05,0.2" for k in (4, 8, 12)], None),
    ],
    # 時鐘曲線往左延伸：every:1/2/3 (every:4 = always，已在第二輪)。延遲 0；session 綁定沿用第二輪；5 seeds。
    # 計畫寫在 round8_every123_plan.txt (跑之前)
    "round8": [
        (["every:1", "every:2", "every:3"], None),
    ],
    # 後期判斷的上限 (oraclelate)：用 runs 4-6 每類一半 (A) 判斷新模型，在 A 以外的 trial 評分；嚴格 (tol 0) 與寬鬆
    # (差超過 5 pp 才不採用)。Block / Never 在同一台主機上重跑，主要比較不混進硬體差異。延遲 0；5 seeds；只在一台主機。
    # 計畫與判定規則寫在 round9_oraclelate_plan.txt (跑之前)
    "round9": [
        (["oraclelate", "oraclelate:0.05", "block", "never"], None),
    ],
}

if __name__ == "__main__":
    for conds, seeds in PHASES[sys.argv[1]]:
        for sub, ses in SESSIONS:
            for c in conds:
                print(f"{sub}:{ses}:{c}" + (f"@{seeds}" if seeds else ""))
