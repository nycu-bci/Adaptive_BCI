# Online finetune gate: Wald SPRT per model

**Status (2026-11-19):**
- Implemented in `e3e86df`. The parameters were changed to 0.65/0.85 in `377b3d7`.
- It is the live default: `online_adaptation.finetune_gate = "sprt"`.
- No session has been recorded with it yet.

## Context

The old (block) gate is in [EEG_Train.py](../python/main/EEG/EEG_Train.py), `_gates`.
- **Rule:** after each 4-trial block, a model is finetuned if it had at least 1 wrong trial in that block and its run accuracy is below 0.75.
- **Problem:** in simulation this gives about 11 finetunes per 48-trial run at chance accuracy, and about 4 at 0.75. It reacts mostly to noise.

The SPRT gate replaces this rule with a sequential test for each model, updated after every trial. A model is finetuned only once there is enough evidence that it is performing below the break-even accuracy.

## Decisions

| Item | Choice |
|---|---|
| What is decided | Finetune or not, for each model separately |
| Observation | One correct/wrong result per trial: the model's trial vote, scored as `score_block` already does. The live model is scored by Unity's `CUT` / `WRONG_CUT`; the other model by the replica vote. |
| Parameters | **p0 = 0.65, p1 = 0.85, α = 0.05, β = 0.2**, all in `config.json`. The first version used p0 = 0.5, p1 = 0.75. |
| Lower line crossed | Finetune that model **immediately**, even mid-block, on its replay buffer. The trials since the last save are saved first. |
| Upper line crossed | Log "good", reset the score to 0, keep testing |
| Other resets | After each finetune of that model; at the start of each run |
| Models | A and B independently |
| Calibration song | Uses the same gate |
| A/B switching | Unchanged: decided on fixed 4-trial blocks |

### What the parameters give

| | Value |
|---|---|
| Step for a correct trial, ln(p1/p0) | +0.268 |
| Step for a wrong trial, ln((1−p1)/(1−p0)) | −0.847 |
| Upper line ("good"), ln((1−β)/α) | +2.773 |
| Lower line (finetune), ln(β/(1−α)) | −1.558 |
| Break-even accuracy (score drifts neither up nor down) | 0.760 |
| Starting from 0 | 2 wrong trials → finetune; 11 correct trials → good |

- **p0:** the accuracy of a model that needs finetuning.
- **p1:** the accuracy of a model that is fine.
- **α:** the probability of calling a bad model good, which means a missed finetune.
- **β:** the probability of finetuning a good model, which means a wasted finetune.

**Why 0.65/0.85.** The F2 counterfactual analysis covered 5 sessions and 328 normal blocks. Finetuning helped most when run accuracy was 0.6–0.75 (+2.7 to +4.1 pp). It helped little above 0.8 (+0.1 to +0.8 pp). A break-even near 76% matches that pattern.

**Replay on recorded sessions.** These are block-gate sessions; the models are taken as they were recorded.
- The SPRT gate would finetune 80 times: A 39, B 41.
- On the same 328 blocks:

| | Old gate | SPRT 0.65/0.85 |
|---|---|---|
| Blocks fired | 161 | 58 |
| Effect of the fired blocks | +2.5 pp | +2.6 pp |
| Fires that help | 62% | 60% |
| Helpful blocks caught | 53% | 19% |

The SPRT fires far less often with about the same benefit per fire, but it catches fewer of the helpful blocks.

## Design

1. **[sprt_gate.py](../python/main/Utils/sprt_gate.py)** (new, no I/O)
   - `SprtGate(p0, p1, alpha, beta)`.
   - `update(correct)` returns `None`, `"finetune"` or `"good"`. The score resets after a decision; `None` (a trial that wasn't scored) is skipped.
   - `reset()`.
2. **Config.** In [config.json](../python/config.json), `online_adaptation` gets:
   - `finetune_gate`: `"sprt"` or `"block"`;
   - `sprt_p0`, `sprt_p1`, `sprt_alpha`, `sprt_beta`.

   [config.py](../python/main/Utils/config.py) reads them.
3. **Trigger every trial.** In `"sprt"` mode, [UnityMarkerReader.py](../python/main/Utils/UnityMarkerReader.py) fires after every `END` in Adaptive runs and the calibration song. The legacy path (no `MODE:`) is unchanged.
4. **Per-trial step: [sprt_finetune.py](../python/main/EEG/sprt_finetune.py)** (new), called from `MIState` / `CalibrationState` in [game_state.py](../python/main/game_state.py).
   - **Read:** a separate scoring pointer reads the new log lines, so the save pointer is untouched. If a finetune blocked the loop, several trials are scored in one go.
   - **Update and decide:**
     - each trial updates A's and B's gates;
     - if a model crossed the lower line, the trials are saved and only that model is finetuned;
     - at a block end with no crossing, the trials are saved without a finetune, so both replay buffers keep growing;
     - on any other trial, the step only scores and logs.
   - **Resets:** a model's score resets after its finetune; both scores reset when `runCount` changes.
5. **[EEG_Train.py](../python/main/EEG/EEG_Train.py)**
   - `run_calibration` takes the list of models to finetune from the step. `_gates` is used only in `"block"` mode.
   - `_decide_live` (A/B switching) runs only at block ends.
   - The save check no longer requires at least 7 new log lines; at least one complete trial is enough.
6. **Logs**
   - New `run{N}/sprt_trials.csv`: `trial, model, live, correct, step, score, decision, p0, p1, alpha, beta`. This is the data for reviewing the parameters.
   - `ab_blocks.csv` gets the columns `trigger` (`block` / `sprt`), `sprt_A` and `sprt_B`.
7. **[sprt_gate_replay.py](../python/analysis/sprt_gate_replay.py)** (new) replays both gates on recorded sessions. Its defaults come from `config.json`.

## Revert

- **Switch back to the old gate, no code change:** set `"finetune_gate": "block"` in `config.json`. This runs the old block-of-4 path, and the old tests pass in this mode.
- **Remove the feature completely:** `git revert 377b3d7 e3e86df`. Revert `106bc6f` too only if the Unity materials and base model should also come out.

## Tests

[test_sprt_gate.py](../python/tests/test_sprt_gate.py) covers the gate itself.

[test_parallel_finetune.py](../python/tests/test_parallel_finetune.py) covers the online behaviour:
- a mid-block crossing finetunes only the model that crossed;
- a block end with no crossing still saves;
- score resets after a finetune and on a new run;
- the calibration song uses the SPRT;
- `"block"` mode keeps the old behaviour;
- several trials can be scored in one call;
- switching is still decided on 4-trial blocks;
- the config keys are read.

```
C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_sprt_gate.py
C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_parallel_finetune.py
```

## Open

1. **First live session.** Confirm that `sprt_trials.csv`, the `trigger` column and the finetuned checkpoints appear as expected. No dry run with Unity is on record.
2. **Compare the old and new gates.** Run both gates on the same subjects (a within-subject design); accuracy differs by about 16 pp between subjects (sub01–05). Two designs have been considered, and neither is chosen yet:
   - same session, with a model reset between gates: about 12 subjects;
   - two sessions per subject: about 23 subjects, for a 5 pp difference.

   As a secondary measure, compare the paired F2 decision quality (the analysis behind the table above) in every session.
3. **Review the parameters** after that data collection. Use `sprt_trials.csv`, the `ab_blocks.csv` SPRT columns and `sprt_gate_replay.py`.
4. **Known risk.** The plan noted that `Calibration2.cs` waits for "calibration done"; check that a broadcast from a mid-block finetune can't release that wait early. `TimeLogger.cs` still sends `calibration_start`, which Python ignores when `MODE:` is known.
