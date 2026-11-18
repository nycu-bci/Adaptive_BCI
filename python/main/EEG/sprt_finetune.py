"""
Per-trial SPRT finetune gate (online_adaptation.finetune_gate = "sprt").
Called by MIState / CalibrationState after every trial END (UnityMarkerReader.should_train_after):
  1. read the trials that ended since the last tick (own log pointer; the .pt save pointer is not touched)
  2. score them for A and B (trial_scoring.score_block) and update each model's SprtGate -> run{N}/sprt_trials.csv
  3. a model crossed the lower line, or a 4-trial block ended -> save the trials since the last save,
     run_calibration with the SPRT gates (both replay buffers always get the data), swap in new weights
Trials voted before a model's latest finetune was swapped in are logged as "stale" and not counted for the new model.
"""
import csv
import os

import main.Utils.config as config
import main.Utils.global_value as global_value
from main.Utils.file_pointer_reader import readLogFromPointer
from main.Utils.sprt_gate import SprtGate
from main.Utils.trial_scoring import score_block

MODELS = ("A", "B")


class SprtTicker:
    HEADER = ["trial", "model", "live", "correct", "step", "score", "decision", "p0", "p1", "alpha", "beta"]

    def __init__(self):
        self.gates = {name: self.new_gate() for name in MODELS}
        self.counter_run = None
        self.score_pointer = 0
        self.block_scores = {name: [0, 0] for name in MODELS}  # 本 4-trial block 的 [正確數, 計分數]，給 live 模型切換用
        self.valid_from = {name: 0.0 for name in MODELS}  # 這個時間之前投票的 trial 屬於微調前的舊模型

    @staticmethod
    def new_gate():
        return SprtGate(config.SPRT_P0, config.SPRT_P1, config.SPRT_ALPHA, config.SPRT_BETA)

    def new_run(self, game):
        """新 run：SPRT 分數歸零，計分 pointer 從目前的存檔 pointer 開始 (已跳過 Lobby 期間的 trial)"""
        self.counter_run = global_value.runCount
        self.gates = {name: self.new_gate() for name in MODELS}
        self.block_scores = {name: [0, 0] for name in MODELS}
        self.score_pointer = game.logLinePointer

    def tick(self, game):
        if self.counter_run != global_value.runCount:
            self.new_run(game)
        game.unity_reader.flushLog()
        trials, self.score_pointer, _ = readLogFromPointer(game.unity_reader.filename, self.score_pointer)
        ended = {idx: t for idx, t in trials.items() if 'end' in t}
        _, per_trial = score_block(ended)

        finetune, block_end, rows = set(), False, []
        for idx in sorted(per_trial):
            res = per_trial[idx]
            for name in MODELS:
                correct = res[name]
                stale = correct is not None and res['vote_ts'] is not None and res['vote_ts'] < self.valid_from[name]
                gate = self.gates[name]
                if stale:
                    step, decision = 0.0, "stale"
                else:
                    step = gate.step(correct)
                    decision = gate.update(correct) or ""
                if decision == "finetune":
                    finetune.add(name)
                if correct is not None:
                    self.block_scores[name][0] += int(correct)
                    self.block_scores[name][1] += 1
                rows.append([idx, name, res['live'], "" if correct is None else int(correct), f"{step:.3f}",
                             f"{gate.score if stale else gate.last_score:.3f}", decision,
                             gate.p0, gate.p1, gate.alpha, gate.beta])
            if (idx + 1) % config.trial_train_interval == 0:
                block_end = True
        self._log(rows)
        if rows:
            print(f"{config.TAGS.INFO.value} SPRT trials {sorted(per_trial)}: "
                  + " ".join(f"{n}={self.gates[n].score:+.2f}" for n in MODELS)
                  + (f" -> finetune {''.join(sorted(finetune))}" if finetune else ""))

        if not (finetune or block_end):
            return
        game.saveDataAsPtFromPointer()
        switch_scores = {n: tuple(self.block_scores[n]) for n in MODELS} if block_end else None
        if block_end:
            self.block_scores = {name: [0, 0] for name in MODELS}
        game.ft_pipeline.run_calibration(sprt={
            "gates": {n: n in finetune for n in MODELS},
            "trigger": "sprt" if finetune else "block",
            "switch_scores": switch_scores,
            "scores": {n: self.gates[n].score for n in MODELS},
        })
        game.dump_predictions()
        game.swap_in_new_models()
        for name, trained in game.ft_pipeline.last_trained.items():
            if trained and global_value.last_swap_time is not None:
                self.valid_from[name] = global_value.last_swap_time
                self.gates[name].reset()

    @classmethod
    def _log(cls, rows):
        if not rows:
            return
        path = os.path.join(config.getRunDataDir(), "sprt_trials.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(cls.HEADER)
            w.writerows(rows)
