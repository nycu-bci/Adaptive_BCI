# A/B online system: Unity-based scoring, per-model finetune, WRONG_CUT, per-subject checkpoints

## Context

The first real-cap test (`real_time_data/sub01/ses1/run1`, `run2`) raised two design problems:

1. Scoring uses a whole-trial majority vote, which is not what Unity uses to cut.
2. Both models finetune whenever the live model does badly. A good background model is finetuned anyway, and when the live model does well the background model is never finetuned or scored.

Decisions taken in discussion:
- Scores come from Unity's real result (live scoring).
- Unity logs a new `WRONG_CUT` line at the vote time of a wrong trial.
- Python decides when each model finetunes; Unity must not pause.
- New weights are never loaded while Unity is about to vote.
- Finetuned checkpoints are stored per subject/session.
- Kept as they are: the 1 s main-loop polling, and the streak switching rule.
- Dropped: event wait, rolling-window switch, replay-buffer rebuild after a restart.

## Evidence from the test

| Measurement | run1 | run2 |
|---|---|---|
| Trials / trials with a CUT | 48 / 23 | 48 / 16 |
| Live model's `MODEL_PREDICTION` agrees with CUT | 48 / 48 | 47 / 48 |
| Re-computed vote (`MODEL_PREDICTION` = label), A / B | 23 / 25 | 17 / 16 |
| Whole-trial vote correct, A / B | 25 / 25 | 19 / 20 |
| Trials where A and B vote the same | 36 / 48 | 35 / 48 |
| CUT → END, median | 0.698 s | 0.698 s |

- Only the CUT count is Unity's real result, and only for the live model. The "re-computed vote" row is Python's replica of Unity's vote (the `A=` / `B=` values on END lines) compared with the label.
- START → CUT is 4.232–4.256 s over 66 trials in three runs (medians 4.246, 4.246, 4.254). Other songs are untested.
- Finetune time grows from about 2 s (block 1) to 8–9 s (blocks 8–12); a trial cycle is 6 s. Unity does not wait, so new weights load 1–2 trials late, sometimes mid-trial.
- The one mismatch (run2 trial 15) is most likely the 0.23 s late `AT`; `WRONG_CUT` removes that cause.

## Design

**Roles.** Unity reports events (START, CUT, WRONG_CUT, END, MODE) and keeps deciding each swing from the last 100 weighted predictions, unchanged. Python makes every model decision.

**WRONG_CUT.** Unity writes `Trial N WRONG_CUT: <ts>` at the moment of a wrong vote, between START and END:
```
Trial 8 START: 1791884622.154 LABEL: 0 MODE: Adaptive
Trial 8 WRONG_CUT: 1791884626.400
Trial 8 END: 1791884627.097 LABEL: 0 MODEL_PREDICTION: A=1 B=0 LIVE=B AT: 1791884626.400
```

**Scoring (Unity-based).** Per trial, vote time = the CUT or WRONG_CUT timestamp (fallback `END − 0.70 s` for old logs).
- Model that was broadcasting: CUT = correct, WRONG_CUT = wrong (exact, from Unity).
- Other model: replica of Unity's last-100 weighted vote at that same timestamp (`unity_vote_at`, already in `trial_scoring.py`).
- The whole-trial vote is no longer computed online. It can be recomputed later from `predictions.csv`.
- All predictions are saved to `run{N}/predictions.csv`.

**Trigger (Python).** On the END line of every `trial_train_interval`-th trial (trial number from the line), Python raises the same flag Unity's message raises today, at the same moment. The main loop's 1 s polling is unchanged.
- Runs only when `MODE: Adaptive` or the calibration song is active.
- If a START line has no `MODE:` (old Unity build), Python keeps the current behaviour and follows Unity's trigger message.
- Each model finetunes only if it had a wrong trial in the block and its run accuracy is below 0.75 (same rule as `TimeLogger.cs`, threshold in `config.json`).
- Both models are scored and both replay buffers receive the block's windows at every check.

**Replay buffer separated from training.** Today `OnlineCalibrationTrainer.online_train` (`MI_train.py`) does two jobs in one call: it stores the new block's windows in a replay buffer, then trains 4 epochs on the whole buffer. A's buffer is `global_value.replay_buffer` (a global that `else/pretraining/phase2_calibration_sim.py` overwrites), B's is a private dict. Per-model gating needs "store" without "train".

New structure (the offline script no longer constrains it):
- A small `ReplayBuffer` class in `MI_train.py`, one per model, owned by `_ModelSlot`:
  - `add(dataset)`: the existing FIFO rules (384 windows per class, overflow and new windows go to the validation part, validation capped at 40%).
  - `train_set()` / `val_set()`: the windows to train and validate on.
- `OnlineCalibrationTrainer` only trains: `train(train_set, val_set)` with the existing epochs, weighting and best-validation-epoch selection. It no longer knows about buffers or `global_value`.
- Per block, for each model: `slot.buffer.add(block)` always; `slot.trainer.train(...)` only if that model's gate is open.
- A and B are now symmetric, and `global_value.replay_buffer` / `replay_buffer_val` are removed.
- Without this, a model that did well would either be trained anyway or miss that block's data, and its buffer would only ever hold data from its bad blocks.
- `else/pretraining/phase2_calibration_sim.py` used the old `online_train` and the global buffer; it is deleted (kept in git history).

Full picture of the replay buffer (behaviour kept from today unless marked new):

| Stage | What happens |
|---|---|
| Block saved | `data_*.pt` holds the block's trials: 32 channels × 2000 samples (first 4 s of each trial), label, failure flag. |
| Per-model view | Each slot takes its own channels: A 22, B 32. |
| Windows | `prepare_calibration_dataset` cuts each trial into 16 windows (500 samples, stride 100), applies channel-mean removal and band-pass, one-hot label. A 4-trial block gives 64 windows. |
| Weight (new rule) | 2.0 if the trial failed for this model, else 1.0. See "Failure weighting" below. |
| `add` | Each window goes into its class list. While the class list has at most 384 windows, a copy also goes into the validation list. Above 384, the oldest window moves from the training list to the validation list. Validation is capped at 153 per class (40% of 384); the oldest is dropped. |
| Capacity | 384 per class ≈ 24 trials per class ≈ 48 trials (one run). Per-class lists keep the classes balanced when the labels are. |
| `train` (only if the gate is open, new) | All training windows, shuffled, batch 8, Adam at 2 × 0.001, 4 epochs, loss weighted by window weight. After each epoch, weighted loss on the validation list (training loss if empty). The best epoch's weights become `cA_NNN` / `cB_NNN`. |
| Lifetime | Created empty when Python starts, one per model, kept in memory across runs, never written to disk, lost on restart. |
| Threads | Each model's buffer is used only by that model's finetune thread, so no lock is needed. |
| Memory | About 47 MB (A) and 69 MB (B) when full. |

New code shape:
```
class ReplayBuffer:                      # MI_train.py
    add(dataset)                         # FIFO rules above
    train_set() -> TensorDataset | None
    val_set()   -> TensorDataset | None
    sizes()     -> (train per class, val per class)

class OnlineCalibrationTrainer:          # training only
    load_checkpoint(path)
    train(train_set, val_set) -> loss    # 4 epochs, best-validation epoch kept

_ModelSlot (EEG_Train.py), per block:
    windows = prepare_calibration_dataset(own channels of the block)
    self.buffer.add(windows)                         # always
    if gate_open:
        self.trainer.load_checkpoint(latest)
        self.trainer.train(self.buffer.train_set(), self.buffer.val_set())
        save cX_NNN
```

**Failure weighting (per model).** Failed trials count double in the training and validation loss. Today every trial counts as failed (`data_process_np.py:89` needs 5 CUTs), so the weighting does nothing, and the flag reflects only the live model.
- New config key `success_cut_count` (1 for new logs, 5 for old data): `is_fail = ('wrong_cut' in trial) or len(cuts) < success_cut_count`. A trial with a CUT is a success; with WRONG_CUT or no CUT, a failure. This also works before Unity is recompiled.
- `data_process_np.py` (`_parse_log_file`, `_extract_segments`) and `else/preprocessing/create_MInp.py` parse WRONG_CUT and use the same rule, so online and offline agree.
- `saveDataAsPtFromPointer` also stores each trial's number in `data_*.pt` (`trial_ids`), reordered together with the trials by `arrange_by_label`.
- In `_calibrate`, the model that was broadcasting keeps that flag; the other model's flag becomes "its replica vote was wrong", using the per-trial results already computed for scoring.
- No effect on timing or runtime; models will learn differently from the two test runs.

**Model swap (vote flag).** A flag marks "Unity is about to vote":
- On: from START + (vote offset − `swap_guard_s`, default 1.5 s).
- Off: when the CUT or WRONG_CUT line arrives.
- New weights load only while the flag is off.
- Vote offset: `vote_offset_s` (default 4.25) for the first trial of a run, then the START-to-vote time measured in that run, so songs with other timing adapt.

**Live-model switching.** The existing streak rule (`_decide_live`) stays, now fed with the Unity-based scores.

**Checkpoints.** `checkpoint_main/` stays and keeps the starting models (`cA_000`, `cB_000`, `model.pth`); it is still what the Unity lobby lists. Finetuned `cA_NNN` / `cB_NNN` go to `real_time_data/sub{S}/ses{N}/checkpoints/`, first file `001`.
- Saving always uses the highest existing number + 1, so the highest number is always the newest.
- `--start_from session | base | subject` (default `session`): newest of this session, else the base models; `base` always starts from `cA_000` / `cB_000`; `subject` takes the highest session that has checkpoints.
- A and B are resolved separately (for example `cA_009` and `cB_012`); `--live_model` and `--start_model` do not affect this.

## Files to change

Unity (`unity_noVR` only):
- `Assets/Script/FileProcess/TimeLogger.cs`: add `WrongCut` to `LogType`; write the `WRONG_CUT` line.
- `Assets/Script/BeatSaber/Effect/AutoSaber.cs`: invoke the log callback with `LogType.WrongCut` at the two `wrong_hit += 1` sites.

Python:
- `main/Utils/UnityMarkerReader.py`: accept `WRONG_CUT`; record each trial's vote time, result and broadcasting model; maintain the vote flag and the measured vote offset; raise the block flag on every Nth END; use the vote time for `AT`.
- `main/Utils/file_pointer_reader.py`: parse `WRONG_CUT` into the trial dict.
- `main/EEG/data_process_np.py`, `else/preprocessing/create_MInp.py`: WRONG_CUT parsing and the `success_cut_count` failure rule.
- `main/game_state.py` (`saveDataAsPtFromPointer`): store `trial_ids` in `data_*.pt`.
- `main/Utils/trial_scoring.py`: Unity-based `score_block` (returns per-trial results too); `dump_predictions`.
- `main/EEG/MI_train.py`: new `ReplayBuffer`; `OnlineCalibrationTrainer` reduced to training only.
- `main/Utils/global_value.py`: remove `replay_buffer` / `replay_buffer_val`.
- `else/pretraining/phase2_calibration_sim.py`: delete.
- `main/EEG/EEG_Train.py`: per-model gate and per-run counters in `_calibrate`; `_ModelSlot.finetune(train_data, train)`; save path in the session checkpoint folder; `ab_blocks.csv` columns `ft_A, ft_B, ft_seconds, swap_time` appended.
  - `ft_A` / `ft_B` (1 or 0): whether each model finetuned in that block, which is no longer the same for both.
  - `ft_seconds`: how long the finetune took (today only visible from file timestamps).
  - `swap_time`: when the new weights were loaded, so each trial can be tied to its weights.
- `main/game_state.py`: wait for the vote flag before `reload_predictors`; record the broadcasting model; dump predictions at run end.
- `main/Utils/config.py`, `config.json` (new keys: `finetune_acc_threshold`, `vote_offset_s`, `swap_guard_s`, `success_cut_count`; `prediction_log_offset_s` 0.47 → 0.70), `main/Utils/global_value.py`, `main/Utils/some_functions.py` (highest + 1 numbering), `main_start.py` (`--start_from` in `setup_args`).
- `README.md` section 4, and `tests/test_parallel_finetune.py`.

Documentation (in `docs/`):
- `docs/plan_ab_online_scoring_gating.md`: this plan, including the evidence table (known issues now in `docs/known_issues.md`), updated at the end with what was actually done and any deviation.
- `docs/CHANGELOG.md`: dated change log. First entries cover the changes already made in this work (the `weights_only=False` loads, `MODE:` on START lines and the `mode` column, `--subject` / `--session` / `--start_model`, `setup_args()`, README buffer numbers), then one entry per item of this round with the files touched and the test result.
- `docs/handoff_two_model_online.md`: left as it is; the plan document notes which of its open points this round closes.

Before starting: commit the Python changes made so far on a new branch, leaving the Unity asset changes and the `picture/` deletion unstaged.

## Verification

1. Unit tests (`C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_parallel_finetune.py`):
   - scoring from CUT / WRONG_CUT / fallback; `AT` equals the vote time;
   - gate matrix (each model good or bad → which checkpoints appear, both buffers always grow);
   - streak switching still works with the Unity-based scores;
   - vote flag on/off and the measured offset; swap refused while the flag is on;
   - `--start_from` choices with unequal A and B numbers; highest + 1 numbering after a deleted file;
   - `predictions.csv`; legacy path when `MODE:` is missing;
   - failure rule (CUT / WRONG_CUT / no CUT, and 5 for old logs); `trial_ids` stay aligned after `arrange_by_label`; per-model weights in each buffer.
2. Replay check: feed the run1 and run2 logs through the new parser and confirm the live results equal the CUT counts (23 and 16).
3. One Adaptive run with the cap after Unity recompiles:
   - every wrong trial has a `WRONG_CUT` at about START + 4.25 s, and CUT + WRONG_CUT = number of trials;
   - `ab_blocks.csv` has a row every 4 trials with `mode` filled, and `ft_A` / `ft_B` differ when the models differ;
   - no `swap_time` falls inside a vote-flag window;
   - prediction rate from `predictions.csv` confirms 100 predictions fit inside `swap_guard_s`.
4. Recompute scores from `predictions.csv` plus the log and compare with `ab_blocks.csv`.
5. Later: repeat step 3 with a different song to check the measured vote offset.

## Known issues left out of this round

Moved to [`docs/known_issues.md`](known_issues.md).

## Recording findings and fixes (2026-10-15 recording, `sub01/ses1`)

Scoring, gate and swap checks all passed, but the recording showed:

- **Missed song.** Unity's lobby connection was refused (`Player.log:150`, `game.exe` opened while Python was still starting). Unity dropped its stage messages (`send_string_to_python` drops when `stream == null`) and never retried; the song scene's own stage message crashed on `Config.Stage[2]` (`GameManager.Start`). Python stayed in Lobby for the whole first song and the block flag stayed set, so the first save of the next song read the log from line 0 and trained all 48 old trials as one block (19.2 s).
- **Log-pointer trial loss.** `readLogFromPointer` moved the pointer to the end of what it read, so a late save (after the next `Trial N START`) skipped that START and dropped the trial (run1 trial 4). Predates this branch.
- **AutoSaber vote race.** The receive thread changes `valueHistory` while `AutoSaber` iterates it. About 1 in 10,000 votes; 0 seen.

Fixes:

1. `readLogFromPointer`: if the last trial with a START in the read has no END, the pointer stays on that START line (earlier START-without-END trials are abandoned and not held). `newLineCount` and `trials` keep their meaning.
2. `LobbyState.update` clears the block flag with a WARNING; `MIState.start` / `CalibrationState.start` call `Game.skipLogToEnd()` (flush, move the LOG pointer to the end, WARNING with skipped trial lines).
3. `TCPServer.start_watchdog()` (started by `LSL.setup_lsl_inlet` through `TCPServer.on_lsl_connected`): "Waiting for Unity connection" / "Connection to unity lost" every 5 s after 3 s with no client.
4. `unity_noVR` `TCP_Client.cs`: retry every 1 s, pending messages (cap 50), disconnect detection and reconnect, close on `OnDestroy`.
5. Status label (top-right, icon, green/red) created in code by `TCP_Client`.
6. `valueHistory` lock in `update_history`; `AutoSaber` copies it under the lock and counts outside.
7. `GameManager.Start` sends the stage string only if `(int)now_stage < Config.Stage.Length`.

Not verified here: Unity compile and run (start order, one Adaptive song, quit `game.exe`).

## What was done / deviations (2026-10-14)

Branch `ab-online-scoring-gating`. Everything in "Files to change" was implemented; `python/tests/test_parallel_finetune.py` has 27 tests (24 at implementation, 3 added in the review fixes), all passing. The Unity edits (`TimeLogger.cs`, `AutoSaber.cs` in `unity_noVR`) were not compiled here, but they ran in the `sub01/ses1` tests (see "Log verification" below). Verification step 3 is covered by those runs; step 4 is done; step 5 (second song) is still open.

Replay check (Verification step 2, run1 / run2 logs fed through the new reader): 48 trials each, CUT lines 23 and 16, live-model results counted by `score_block` 23 and 16. With `MODE: Adaptive` added to the START lines the reader raised the block flag 12 times per run (every 4th END); without MODE it raised none (legacy path). Measured START to CUT offset 4.245 s (run1) and 4.246 s (run2).

Closes from `docs/handoff_two_model_online.md` (left unchanged): section 2 (every noVR trial counted as a failure because of the 5-CUT rule; `AT` now uses the trial's vote time with the 0.70 s fallback).

Deviations and choices the plan did not fix:
- `OnlineCalibrationTrainer.train(train_set, val_set)` overrides `BraindecodeTrainer.train(...)` with a different signature, as the plan's code shape says. Nothing calls the parent's `train` on this class.
- `ft_seconds` is the wall time of the parallel finetune step of the block (buffer add included, both models), not a per-model time.
- `ab_blocks.csv` is written before the new weights are loaded, so `swap_time` is filled afterwards: `game.swap_in_new_models()` calls `EEGFineTunePipeline.record_swap`, which rewrites that block's row. It is the time right after the vote-flag wait, before `reload_predictors`.
- The wait for the vote flag has a fixed 3 s timeout (`trial_scoring.VOTE_WAIT_TIMEOUT_S`), not a config key. After the timeout the swap proceeds.
- `prediction_log_offset_s` (0.70) is also written into `python/config.json`, not only the code default.
- Broadcasting model per trial (`global_value.trial_live`) is recorded when the CUT / WRONG_CUT line arrives (or at END if there is none), from `global_value.broadcast_model`, which only changes in `Game.reload_predictors`.
- Trial numbers in the Unity log start at 0, so the Python trigger fires on trials 3, 7, 11, ... (`(idx + 1) % trial_train_interval == 0`).
- A trial that the other model could not score (fewer than 2 predictions before the vote) is not counted for that model and is not marked as a failure for it.
- `trial_ids` is stored by `saveDataAsPtFromPointer` only; `EEGSelfDataLoader.save_as_pt_with_name` (offline path) does not store it, and `_calibrate` then keeps the stored failure flags for both models.
- `else/preprocessing/create_MInp.py` does not import `config`; it has a module constant `SUCCESS_CUT_COUNT = 1` with a comment.
- `else/pretraining/plot_phase2_overwrite.py` still mentions the deleted `phase2_calibration_sim.py` (it only reads that script's output); not touched.
- `run_braindecode_training` (non-A/B path) still calls `get_next_version_path(NOW_TRAINED_CHECKPOINT)`; with the highest + 1 rule it now starts at `c_001` in `checkpoint_main/` when no `c_NNN` exists there.
- `unity_noVR/Assets/StreamingAssets/config.json` and `unity/.../config.json` were not changed (the Python side reads `python/config.json` first).

## Review fixes

A code review of the implementation led to these changes (details in `docs/CHANGELOG.md`, 2026-10-14 review fixes):

1. The Python-side trigger uses a per-run flag `is_calibration_run` (set on the `Calibration` stage message) instead of the stage string, because Unity switches to `MI` right after `Calibration`.
2. `swap_in_new_models` is skipped when no model trained and the live model did not change (`EEGFineTunePipeline.models_changed`).
3. `predictions.csv` is also written after every processed block and in `go_to_training`.
4. `--start_from subject` still resolves A and B independently; the start folders are printed and a warning is shown when they differ.
5. `create_MInp.py` takes `--success_cut_count` instead of a code edit.

## Log verification (2026-10-14, `real_time_data/sub01/ses1`, runs 1 to 6)

Runs 1 (calibration song, `MODE: Static`), 2, 4, 5 (Adaptive) and 3 (Static) were checked against this plan. The signal was not a real brain signal, so accuracy says nothing about the models; only the mechanics were checked.

| Check | Result |
|---|---|
| `WRONG_CUT` | CUT + WRONG_CUT = 48 in every run; START to vote 4.235 to 4.263 s; `AT` equals the vote time on every END line |
| Unity-based scoring | All 96 block scores (`correct_A`, `correct_B`; runs 1, 2, 4, 5) recomputed exactly from `predictions.csv` plus the log |
| Replica vote | The live model's Unity-style replica equals Unity's real CUT / WRONG_CUT on 192 of 192 trials; END-line `A=` / `B=` match the recomputed votes |
| Gate | A finetuned in 46 of 48 blocks, B in 44; every skip follows the rule (no wrong trial, or run accuracy 0.75 or more); 46 `cA` and 44 `cB` files, numbered from 001 |
| Vote flag | No `swap_time` inside a vote-flag window; the swap is skipped when no model trained and the live model did not change |
| Static run | Run 3 has no `ab_blocks.csv` and no scoring |
| `data_*.pt` | `trial_ids` stored; labels and failure flags match the log in all trials of runs 1, 2, 4, 5 |
| Streak switching | Run 4 replayed by hand: the live model switches after 2 blocks in a row where the other scores higher |
| Tests | 27 of 27 pass |

Notes:
- The `failed` column of `ab_blocks.csv` lists models whose finetune raised an exception. It is empty in all 48 blocks (no exception); it is not a trial-failure flag.
- Run 1 is the calibration song (`MODE: Static`); it was scored and finetuned because the calibration song is allowed by the trigger rule.
- Run 6 is empty (aborted).
- Not verified: the other model's failure flag inside `_calibrate` (code read only), per-model buffer weights in a live run (unit tests only), and verification step 5 (a second song for the measured vote offset). The next recording will cover these.
- Unity asset changes in `unity/` and `unity_noVR/` remain uncommitted on purpose for now.
