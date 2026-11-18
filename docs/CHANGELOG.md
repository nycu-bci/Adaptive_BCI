# Changelog

Dated change log for the A/B online system. Newest first. Links point to the changed file or function; line numbers are as of 2026-10-18 and may drift with later edits.

## 2026-11-17 (online: SPRT parameters 0.65 / 0.85)

Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_parallel_finetune.py` 39/39, `test_sprt_gate.py` 8/8.

- **`sprt_p0` / `sprt_p1`** 0.5 / 0.75 → 0.65 / 0.85 in `python/config.json` (α, β unchanged).
  - Steps: +0.268 / −0.847 (were +0.405 / −0.693).
  - Break-even accuracy: 76% (was 63%).
  - From a score of 0: 2 wrong trials in a row finetune (was 3); 11 correct in a row call "good" (was 7).
- **Why:** F2 counterfactuals, sub01–05 pooled, 328 normal blocks. A finetune improves the next 48 trials by +2.6 pp on average when the run accuracy is below 0.75, and by +0.3 pp at 0.80 or above. Set B's break-even (63%) left most models in the 0.65–0.75 range alone.
- **Replay** ([`sprt_gate_replay.py`](../python/analysis/sprt_gate_replay.py), sub01–05 ses1): 80 finetunes (A 39, B 41), up from 40 (A 20, B 20); the block gate fired 166.
- **Simulation** (48-trial runs at fixed accuracy): 4.4 finetunes per run at 0.6 accuracy and 0.11 at 0.9; the block gate fires 9.3 and 0.14.
- **Defaults:** the `config.py` fallbacks now match these values. The replay's `--p0` / `--p1` / `--alpha` / `--beta` defaults are read from `config.json`.

## 2026-11-16 (online: Wald SPRT finetune gate)

Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_parallel_finetune.py` 39/39, `test_sprt_gate.py` 8/8, `test_f2.py` 10/10, `test_session_log.py` 2/2.

- **New gate** `online_adaptation.finetune_gate = "sprt"` (default in `python/config.json`; `"block"` restores the old gate with no code change). Each model (A, B) runs a Wald SPRT on its per-trial correctness; a model is finetuned as soon as its score crosses the lower line, mid-block included. Parameters `sprt_p0` / `sprt_p1` / `sprt_alpha` / `sprt_beta` = 0.5 / 0.75 / 0.05 / 0.2, to be reviewed after the data collection (M5 in `known_issues.md`).
- **New files**: [`sprt_gate.py`](../python/main/Utils/sprt_gate.py) (the test), [`sprt_finetune.py`](../python/main/EEG/sprt_finetune.py) (per-trial tick: own log pointer, scoring, `run{N}/sprt_trials.csv`, save + finetune decisions; trials voted before a model's new weights were loaded are `stale`).
- **Hooks in existing files** (all behind `finetune_gate == "sprt"` and a known `MODE:`; old Unity / the VR project keeps the block path, since its `Calibration2.cs` waits for "calibration done"): `should_train_after` fires every trial; `MIState` / `CalibrationState` call the tick; `saveDataAsPtFromPointer` accepts a single complete trial instead of ≥ 7 log lines; `run_calibration(sprt=...)` takes the gates from the tick and decides the live model only at 4-trial block ends.
- **`ab_blocks.csv`**: new columns `trigger` (`block` / `sprt`), `sprt_A`, `sprt_B`; `record_swap` finds `swap_time` by name.
- **Offline replay** [`sprt_gate_replay.py`](../python/analysis/sprt_gate_replay.py) on sub01–05 ses1: the replayed block gate matches every logged finetune (A 88, B 78); the SPRT gate would fire A 20, B 20.

## 2026-11-10 (offline analysis: F2 on sub02–05, online line, in-use windows)

Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_f2.py` 10/10. No online code changed.

- **Other sessions**: `--name` suffix for the F2 analysis and figures; run on sub02–05 (162 decisions, gate matches the logged decision on all 338 blocks of the 5 sessions). Next-48 result, A+B: fired +2.7 pp (+1.1 to +4.2), skipped −0.3 pp (−2.8 to +2.5).
- **Figures**: paired figure (accuracy if skipped vs if finetuned per block) and accuracy timeline with an online line (the checkpoint actually in use at each trial, from `trial_checkpoint_all.csv.gz`).
- **In-use windows**: [`f2_inuse.py`](../python/analysis/finetune_eval/f2_inuse.py) judges each block only on the trials its choice would be in use: from the next block through the next block the gate fired on (or the session end), capped at 48. Fired blocks → wrong fires (keeping the old model clearly better); skipped blocks → missed fires (a finetune clearly better). Clear = ≥ 1 trial and ≥ 4 of 5 seeds agree. Figures `f2_inuse_paired.png`, `f2_inuse_gained.png`. [`f2_analyze.py`](../python/analysis/finetune_eval/f2_analyze.py): vote loading split into `decision_votes()` (outputs byte-identical).
- **Selection caveat**: a window that ends at the next fire ends on a block chosen for the live model's errors, which favours the finetune at skipped blocks (last block +19.3 pp vs rest +1.8 pp, sub01+05). Every number is also given without that block, which is biased the other way (the blocks kept are ones where the live model did well). The two versions bracket the truth; trust only what both show.
- **Result** (finetuned − kept, A+B; full window / without the last block):

  | | sub01+05 | sub02–05 |
  |---|---|---|
  | Fired | +1.8 / −0.4 pp; wrong fires 11% / 24% | +3.0 / −2.5 pp; wrong fires 11% / 17% |
  | Skipped | −0.1 / −1.8 pp; missed fires 39% / 24% | −4.6 / −6.1 pp; missed fires 12% / 6% |

  Both versions agree: sub05 A skipped correctly (−8.1 pp, all 12 skips); missed fires are rare in sub02–05 (mostly sub05 A); sub01 A misses some useful fires (+3.2 / +1.3 pp). Not settled: whether the gate's fires help beyond the block right after them (most fired windows are one block, so only 21–23 fired blocks remain without it).

## 2026-11-03 (offline analysis: F2, how good is the current finetune gate)

Plan: [`plan_finetune_evaluation.md`](plan_finetune_evaluation.md) (F2). Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_f2.py` 7/7. No online code changed.

- **Counterfactuals**: [`f2_counterfactual.py`](../python/analysis/finetune_eval/f2_counterfactual.py) retrains a V0 finetune (H4-correct trials, 5 seeds) at every block of every model, fired or not, from the checkpoint the model had then; 192 decisions on sub01 + sub05. The gate recomputed from `ab_blocks.csv` matches the logged decision on all 194 blocks; fired-block children equal F1b V0 (1 differing vote of 14,095).
- **Analysis and figures**: [`f2_analyze.py`](../python/analysis/finetune_eval/f2_analyze.py), [`f2_figures.py`](../python/analysis/finetune_eval/f2_figures.py); output `python/analysis/results/finetune_eval/f2/`.
- **Result** (child − parent on the next 48 trials, A+B): fired blocks +2.8 pp (95% CI +1.4 to +4.2), skipped blocks +0.9 pp (−0.3 to +2.0), difference +1.9 pp (−0.0 to +3.8). The gate's direction is right but it separates weakly: it fires on 42 of the 112 blocks where a finetune would have helped. The parent's accuracy on the previous 24 trials predicts the effect better than the gate's inputs (Spearman −0.24, p 0.001).

## 2026-10-28 (offline analysis: step 0b and F1b; docs)

Plan: [`plan_finetune_evaluation.md`](plan_finetune_evaluation.md) ("Later: F1b and F2"). Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_f1b.py` 15/15. No online code changed. Main set sub01 + sub05 (61 finetunes, 5 seeds each); the sub02–sub04 extension is partial and not reported.

- **Step 0b, retraining check**: [`step0b_reproduce.py`](../python/analysis/finetune_eval/step0b_reproduce.py) rebuilds each model's replay buffer block by block from the original `data_*.pt` and retrains every saved finetune from its parent with the online settings ([`retrain_common.py`](../python/analysis/finetune_eval/retrain_common.py)). Pass: median vote agreement with the saved child 0.892 vs 0.879 between seeds (next 48 trials). Output `python/analysis/results/finetune_eval/f1b/step0b/`.
- **F1b, finetunes on the H4-correct cut**: [`f1b_variants.py`](../python/analysis/finetune_eval/f1b_variants.py) retrains each finetune on trials re-cut with the run's measured lag, in three variants: V0 (online training as is), V1 (whole held-out trials, parent kept if no epoch beats it), V2 (validate on the next block, accept only if not worse). [`f1b_analyze.py`](../python/analysis/finetune_eval/f1b_analyze.py) and [`f1b_figures.py`](../python/analysis/finetune_eval/f1b_figures.py) write `python/analysis/results/finetune_eval/f1b/` (`summary.txt`, figures).
- **Results** (child − parent on the next 48 trials, A+B): saved finetunes +1.4 pp (95% CI −0.1 to +3.1), V0 +2.9 pp (+1.6 to +4.3), V1 +1.0 pp, V2 +1.9 pp. H4 cost per update (V0 − saved) +1.5 pp (+0.2 to +2.8). V1 −1.9 pp and V2 −1.0 pp vs V0. Decision: keep the current training with the H4 fix; no validation change.
- **Docs**: [`known_issues.md`](known_issues.md) gets a "Fixed issues" section; H4 moved there with the measured cost; H3 gets the F1b test result (stays open).

## 2026-10-27 (later: learning rate, ab_blocks names, offline analyses F1 and cut comparison)

Tests: `test_cut_compare.py` 9/9, `test_finetune_eval.py` 11/11 (VRBCIenv interpreter).

- **Learning rate without the hidden ×2**: `online_adaptation.learning_rate` in [`config.json`](../python/config.json) 0.001 → 0.002, and [`OnlineCalibrationTrainer.train`](../python/main/EEG/MI_train.py) uses it as is (was `online_lr * 2.0`). Same effective rate as before. Fallback in [`config.py`](../python/main/Utils/config.py) 1e-3 → 2e-3.
- **ab_blocks.csv checkpoint names**: [`EEGFineTunePipeline._log_block`](../python/main/EEG/EEG_Train.py) writes the checkpoint file name instead of the full path; [`cut_compare/common.py`](../python/analysis/cut_compare/common.py) `Run` resolves names in the session's `checkpoints/` folder.
- **Three-cut comparison** ([`cut_compare/`](../python/analysis/cut_compare/)): online cut as recorded (~1–5 s, H4), intended 0–4 s, and 4 s ending at the vote, leave-one-run-out on sub01 ses1. Ending at the vote is −1.2 pp vs 0–4 s (CI −2.7 to +0.3): keep 0–4 s (M2 note in `known_issues.md`).
- **F1 chain experiment and checkpoint × trial overview** ([`finetune_eval/`](../python/analysis/finetune_eval/)): `step0a_sessions.py` (lag search per run), `trial_table.py` (offline Unity vote of every checkpoint on every trial), `f1_analyze.py`, `f1_figures.py`. Each saved finetune vs its parent on later trials (next 48): sub01 A +0.1 pp, sub01 B +3.0 pp, sub05 B +2.3 pp. sub02–sub04 failed the 95% prediction-reproduction gate and are reported separately. Output `python/analysis/results/finetune_eval/`.
- **Docs**: [`plan_finetune_evaluation.md`](plan_finetune_evaluation.md) rewritten (5 sessions, F1 + overview, F1b / F2 design); handoff note updated for the learning rate.

## 2026-10-27 (H4 fix: EEG file start time = first sample)

Check: `python python/tests/test_eeg_reader_timing.py` 5/5. The end-to-end test runs the real reader against a local LSL stream (32 ch, 500 Hz, 50-sample chunks stamped 0.1 s before push): `Record datetime + Timestamp` matches each sample's true Unix time within 0.57 ms max over 1250 rows. The same test on the old code is off by 961 ms. All other test files pass (`test_parallel_finetune` 31/31, `test_session_log` 2/2, `test_cut_compare` 8/8, `test_errp_analysis` 11/11). Not yet checked with the real amplifier.

- **Connect first**: [`CygnusEEGReader.read_eeg`](../python/main/EEG/CygnusEEGReader.py) connects to LSL (`setup_lsl_inlet`, then `inlet.open_stream`) before creating the CSV, and reads `inlet.time_correction()` (0.0 with a WARNING if it fails).
- **Start time from the first sample**: the CSV is created when the first sample arrives, with `Record datetime = fsample_ts = first_ts_lsl + time_correction + (time.time() − local_clock())`. Row `Timestamp` is unchanged (`ts − first_ts_lsl`), so Unix time of a row = `Record datetime + Timestamp`, and trial cuts `START − Record datetime` land on the right samples. One INFO line per run prints `fsample_ts`, the time correction and the arrival delay.
- **Header format**: `Record datetime` is always written with 6 microsecond digits (`str(datetime)` dropped them at exactly 0, which the readers can't parse). Other header lines and the row format are unchanged; readers unchanged.
- **Simulated mode** (configured or fallback): unchanged; the file is created at the start with the current time.
- **No sample in a run**: the CSV is never created; [`readCsvFromPointer`](../python/main/Utils/file_pointer_reader.py) returns "no data" for a missing file instead of raising.
- **Existing recordings** are not changed and still need the per-run lag correction (H4 in [`known_issues.md`](known_issues.md)).

## 2026-10-20 (48 trials per song, `unity_noVR` only)

Plan: [`plan_48_trial_songs.md`](plan_48_trial_songs.md). Check: `python python/else/music_tools/extend_maps_to_48.py` gives OK for all 5 maps (240 notes, 48 groups of 5 by a Python port of `GroupNotesByRules`, median start-to-start 6.8-7.0 s, last note 339-354 s). Unity not compiled or run.

- **Songs**: calibration `ShriekAndOri`, run1 `ClimbingTheGinsoTree`, run2 `NowUseTheLight`, run3 `LumaPools`, run4 `OriAndTheBlindForest` go from 40 to 48 trials with the same trial timing (5 notes over ~3.0 s, ~3.7 s gap).
- **Map script**: new [`extend_maps_to_48.py`](../python/else/music_tools/extend_maps_to_48.py) writes `<map>_48.dat` beside each 40-trial map in `StreamingAssets/{Calibration,run1..run4}` (originals unchanged).
- **Song assets**: `file_names[0]` in the 5 assets under `Assets/SO/Songs/` now points at the `_48.dat` maps.
- **Audio loop**: 48 trials end after the ~300 s audio, so [`BeatSaberInfoLoader.cs`](../unity_noVR/Assets/Script/BeatSaber/SongMapProcess/BeatSaberInfoLoader.cs#L22) gets `loop_audio` (default true), applied at [line 90](../unity_noVR/Assets/Script/BeatSaber/SongMapProcess/BeatSaberInfoLoader.cs#L90) before playback.
- **Not done here**: the "MI trial: 48" label in `SongSelectMenu.cs` was already edited in the working tree (that file also shows line-ending/encoding noise in git).

## 2026-10-19 (Unity: static sabers, charging feedback on the fixation bars, `unity_noVR` only)

Built in the Editor in the new scene `MI_v2.unity`; this entry was written on 2026-10-20 from the saved files. Unity not compiled or run from here, and no tests apply.

- **Static sabers**: new [`Static_Left_saber.mat` / `Static_Right_saber.mat`](../unity_noVR/Assets/Material/Saber/) (`_ColorUp` ≈ `_ColorDown`, so `_scale_range` no longer changes the look). In `MI_v2` the saber mesh `Cylinder.005` of each saber uses them; the originals `Left_saber.mat` / `Right_saber.mat` are untouched by design (`MI`, `MI_Cross`, `Lobby` still reference them).
- **Fixation charging**: the charging fill moved from the sabers to the two side bars of `env/fixation` in `MI_v2`. Cube (1) (left, x = -2) uses new [`Fixation_Left.mat`](../unity_noVR/Assets/Material/env/Fixation_Left.mat) (`_Direction` 1), Cube (3) (right, x = +2) uses new [`Fixation_Right.mat`](../unity_noVR/Assets/Material/env/Fixation_Right.mat) (`_Direction` -1), Cube (2) keeps `Line 2.mat`. Bar scale X is 4.
- **Shader**: new [`Fixation_Shader.shadergraph`](../unity_noVR/Assets/Material/Saber/Fixation_Shader.shadergraph), a copy of `GradientShader` that fills along local X: edge = `_scale_range / 0.5 - 0.5`, `Step(edge, X * _Direction)`, `Lerp(ColorUp, ColorDown, step)` to BaseColor; emission stays `_EmissionHDR`. Since a cube's local X runs from -0.5 to 0.5, the fill is 0-100% of the bar's length at `_scale_range` 0-0.5, whatever the scale. The original `GradientShader` is unchanged.
- **Controller**: `ScaleRangeController` on the TCP object in `MI_v2` is enabled and now points at `Fixation_Left` / `Fixation_Right` (no code change); `speedUp` / `speedDown` 0.2 / 0.2.
- **Side effects seen in git**: `Right_saber.mat` `_scale_range` 0.5 to 0.46951 and `Fixation_Right.mat` saved with `_scale_range` 0.5 (play-mode writes by the controller); reset to 0 before committing if a clean default is wanted. `BeatmapSpawner.cs` and `Lobby.unity` also show many changed lines; for `BeatmapSpawner.cs` the diff is only Chinese comments re-saved as UTF-8 (not checked for the other files).
- **Docs**: [`plan_finetune_evaluation.md`](plan_finetune_evaluation.md) (draft, whether online finetuning helps and whether the gate is right); not an implemented change.

## 2026-10-19 (ErrP analysis, offline)

Plan: [`plan_errp_analysis.md`](plan_errp_analysis.md). Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_errp_analysis.py` gives 11/11 passed (synthetic 32-ch CSV + log with an injected FCz negativity; VRBCIenv has no pytest). No online code changed.

- **Script**: [`errp_analysis.py`](../python/analysis/errp_analysis.py) tests whether wrong votes (`WRONG_CUT`) show an ErrP vs correct ones (`CUT`) on `sub01/ses1` runs 1-7 (alignment check, ERP + cluster/window tests, confound controls, leave-one-run-out decoding). Results in `python/analysis/results/errp/` (`summary.txt`, figures e1-e7).
- **Log parsing**: trial numbers restart inside a log (run1 has trials 0-47 twice), so a new trial opens at every `START`.
- **Tests**: [`test_errp_analysis.py`](../python/tests/test_errp_analysis.py).

## 2026-10-18 (docs only)

- **Known issues** moved from [`plan_ab_online_scoring_gating.md`](plan_ab_online_scoring_gating.md) to the new [`known_issues.md`](known_issues.md) (content unchanged); the plan links to it.

## 2026-10-16 (2026-10-15 recording bug fixes)

Plan: [`plan_ab_online_scoring_gating.md`](plan_ab_online_scoring_gating.md) ("Recording findings and fixes"). Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_parallel_finetune.py` gives 31/31 passed (27 before, 4 new). Replay of a copy of `run1/log_20261015_151841.txt` (start at line 144, read the first 157 lines, then the full file): the first read returns trials 0-3 complete with the pointer on the `Trial 4 START` line (156); the second returns trial 4 complete plus trials 5-47, none repeated. Unity not compiled or run.

- **Log pointer**: [`file_pointer_reader.py` `readLogFromPointer`](../python/main/Utils/file_pointer_reader.py#L64) holds the pointer on the START line of the last trial when it has no END yet.
- **Trials received in Lobby**: [`game_state.py` `LobbyState.update`](../python/main/game_state.py#L87) (clear flag + WARNING), new [`Game.skipLogToEnd`](../python/main/game_state.py#L306), called from [`MIState.start`](../python/main/game_state.py#L185) and [`CalibrationState.start`](../python/main/game_state.py#L134).
- **Connection WARNING**: [`TCPServer.py`](../python/main/Utils/TCPServer.py) ([`start_watchdog`](../python/main/Utils/TCPServer.py#L42), [`on_lsl_connected`](../python/main/Utils/TCPServer.py#L9), [`ever_connected`](../python/main/Utils/TCPServer.py#L29)), [`LSL.py`](../python/main/Utils/LSL.py#L19) (hook after `Connected to LSL stream`).
- **Unity connection handling (`unity_noVR` only)**: [`TCP_Client.cs`](../unity_noVR/Assets/Script/TCP/TCP_Client.cs) (retry loop, pending messages, disconnect detection and reconnect, close on [`OnDestroy`](../unity_noVR/Assets/Script/TCP/TCP_Client.cs#L263)).
- **Status label**: [`TCP_Client.cs` `CreateStatusLabel`](../unity_noVR/Assets/Script/TCP/TCP_Client.cs#L347) (code-created top-right canvas, circle icon, TMP text).
- **AutoSaber lock**: [`TCP_Client.update_history`](../unity_noVR/Assets/Script/TCP/TCP_Client.cs#L252) and [`AutoSaber.cs`](../unity_noVR/Assets/Script/BeatSaber/Effect/AutoSaber.cs) (copy under lock, count outside).
- **Stage message crash**: [`GameManager.cs` `Start`](../unity_noVR/Assets/Script/Manager/GameManager.cs#L82) sends only if `(int)now_stage < Config.Stage.Length`.
- **Tests added** ([`test_parallel_finetune.py`](../python/tests/test_parallel_finetune.py)): [pointer late read](../python/tests/test_parallel_finetune.py#L778), [pointer abandoned trial](../python/tests/test_parallel_finetune.py#L796), [Lobby flag and log skip](../python/tests/test_parallel_finetune.py#L808), [TCP watchdog](../python/tests/test_parallel_finetune.py#L845).
- **Docs**: [plan](plan_ab_online_scoring_gating.md) (known issues, findings section), this entry, [README](../README.md) section 4 (start order, WARNINGs).

## 2026-10-15 (log verification, docs only)

Checked `real_time_data/sub01/ses1` runs 1 to 6 against the plan; no code changes. Everything matched (details in the plan's "Log verification" section). Finetune time growth (2.5 s to 9-14 s) recorded in known issues. Unity asset changes left uncommitted.

## 2026-10-14 (review fixes)

Tests: with the VRBCIenv interpreter, `python python/tests/test_parallel_finetune.py` gives 27/27 passed (24 before, 3 new, 2 existing adjusted). Unity not compiled or run.

- **Calibration song trigger (High)**: Unity sends `Calibration` and then `MI` right away, so the stage string was already `MI` during the calibration song and `MODE: Static` never triggered finetuning. New [`global_value.is_calibration_run`](../python/main/Utils/global_value.py#L48) is set when the `Calibration` stage message arrives ([`UnityMarkerReader.process_message`](../python/main/Utils/UnityMarkerReader.py#L69)), reset in `reset_run_state()`, used by [`should_train_after`](../python/main/Utils/UnityMarkerReader.py#L147). Test replays Calibration, MI, START with `MODE: Static` and ENDs 0-3; and checks no trigger without the Calibration message.
- **No swap when nothing changed**: [`EEGFineTunePipeline.models_changed`](../python/main/EEG/EEG_Train.py#L320) (any model trained or live model changed by [`_decide_live`](../python/main/EEG/EEG_Train.py#L412)); [`Game.swap_in_new_models`](../python/main/game_state.py#L243) returns early otherwise (no vote flag wait, no reload, `swap_time` stays empty in `ab_blocks.csv`).
- **Duplicate import** of `trial_scoring` removed from [`game_state.py`](../python/main/game_state.py).
- **predictions.csv robustness**: new [`Game.dump_predictions()`](../python/main/game_state.py#L256) called after `run_calibration` in `MIState.update` and `CalibrationState.update`, in [`go_to_training`](../python/main/game_state.py#L40), and in [`go_to_lobby`](../python/main/game_state.py#L53). The file is still rewritten completely from `run_start_time`. The in-memory log is still capped at 60000 entries per model.
- **`--start_from subject` visibility**: [`setup_args`](../python/main_start.py#L61) prints the source folder of the A and B start checkpoints and a WARNING when they differ; argparse help and README section 4 document it. Behaviour unchanged (A and B resolved independently).
- **[`create_MInp.py`](../python/else/preprocessing/create_MInp.py)**: `--success_cut_count` (default 1; use 5 for old logs) replaces the hard-coded constant ([`DEFAULT_SUCCESS_CUT_COUNT`](../python/else/preprocessing/create_MInp.py#L25)); threaded through `process_data_dir`, `EEGSelfDataLoader`, `EEGDataLoader`. [`pipeline.py`](../python/else/pipeline.py#L59) got the same option and passes it on; existing callers keep the default.

## 2026-10-14: Unity-based scoring, per-model finetune gate, WRONG_CUT, per-subject checkpoints

Plan: `docs/plan_ab_online_scoring_gating.md`. Branch `ab-online-scoring-gating`. Tests: `C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/tests/test_parallel_finetune.py` gives 24/24 passed. Unity changes are not compiled or run.

- **WRONG_CUT log line (Unity, `unity_noVR` only)**: [`TimeLogger.cs`](../unity_noVR/Assets/Script/FileProcess/TimeLogger.cs#L82) ([`LogType.WrongCut`](../unity_noVR/Assets/Script/FileProcess/TimeLogger.cs#L10), writes `Trial N WRONG_CUT: <ts>`), [`AutoSaber.cs`](../unity_noVR/Assets/Script/BeatSaber/Effect/AutoSaber.cs#L105) (log call before both `wrong_hit += 1; break;`, second at [line 115](../unity_noVR/Assets/Script/BeatSaber/Effect/AutoSaber.cs#L115)).
- **Log parsing accepts WRONG_CUT**: [`UnityMarkerReader.py`](../python/main/Utils/UnityMarkerReader.py#L97), [`file_pointer_reader.py`](../python/main/Utils/file_pointer_reader.py#L78), [`data_process_np.py`](../python/main/EEG/data_process_np.py#L48), [`create_MInp.py`](../python/else/preprocessing/create_MInp.py#L65). Without this the line was dropped from the run log.
- **Unity-based scoring**: [`trial_scoring.py` `score_block`](../python/main/Utils/trial_scoring.py#L83) now returns `(scores, per_trial)`. Vote time is the CUT / WRONG_CUT timestamp (fallback END minus 0.70 s); the broadcasting model gets Unity's result, the other model gets the replica vote at that time. `AT` on END lines is the vote time. `prediction_log_offset_s` 0.47 to 0.70 ([`config.py`](../python/main/Utils/config.py#L88), [`config.json`](../python/config.json)). Whole-trial vote no longer used online ([`window_vote`](../python/main/Utils/trial_scoring.py#L40) kept).
- **Broadcasting model tracking**: [`global_value.py`](../python/main/Utils/global_value.py#L56) (`broadcast_model`, `trial_live`, [`reset_run_state`](../python/main/Utils/global_value.py#L65)), set in [`Game.reload_predictors`](../python/main/game_state.py#L234).
- **Python-side trigger**: [`UnityMarkerReader.should_train_after`](../python/main/Utils/UnityMarkerReader.py#L147); every `trial_train_interval`-th END when `MODE` is known and (Adaptive or Calibration song); Unity's own calibration_start is ignored then; legacy behaviour when no `MODE`.
- **Per-model finetune gate**: [`EEG_Train.py` `_gates`](../python/main/EEG/EEG_Train.py#L382) (at least one wrong trial in the block and run accuracy below `finetune_acc_threshold` 0.75; counters reset when `runCount` changes). Both models always add the block to their own buffer.
- **ReplayBuffer split from training**: [`MI_train.py`](../python/main/EEG/MI_train.py) new [`ReplayBuffer`](../python/main/EEG/MI_train.py#L219); [`OnlineCalibrationTrainer.train(train_set, val_set)`](../python/main/EEG/MI_train.py#L298) trains only. Removed `global_value.replay_buffer` / `replay_buffer_val`; deleted `python/else/pretraining/phase2_calibration_sim.py`.
- **Failure weighting**: `success_cut_count` (new key, 1; [`config.py`](../python/main/Utils/config.py#L101)) and WRONG_CUT in `data_process_np.py` / `create_MInp.py`; `trial_ids` stored in `data_*.pt` ([`game_state.py`](../python/main/game_state.py), [`EEGDataLoader.get_trial_ids_np`](../python/main/EEG/data_process_np.py#L155)) and kept aligned through `arrange_by_label`; the non-broadcasting model's failure flag is "its replica vote was wrong" ([`EEG_Train._failures_per_model`](../python/main/EEG/EEG_Train.py#L396)).
- **Vote flag and swap**: [`trial_scoring`](../python/main/Utils/trial_scoring.py#L120) (`on_trial_start`, `on_trial_vote`, `on_trial_end`, [`vote_flag_on`](../python/main/Utils/trial_scoring.py#L139), [`wait_vote_flag_off`](../python/main/Utils/trial_scoring.py#L144)), new keys [`vote_offset_s`](../python/main/Utils/config.py#L110) (4.25), [`swap_guard_s`](../python/main/Utils/config.py#L111) (1.5); [`Game.swap_in_new_models`](../python/main/game_state.py#L243) waits for the flag (3 s timeout) before `reload_predictors`.
- **ab_blocks.csv**: columns `ft_A, ft_B, ft_seconds, swap_time` appended (existing positions unchanged); [`EEGFineTunePipeline.record_swap`](../python/main/EEG/EEG_Train.py#L445) fills `swap_time`.
- **predictions.csv**: [`trial_scoring.dump_predictions`](../python/main/Utils/trial_scoring.py#L154), written when a run ends (`go_to_lobby`).
- **Checkpoints**: finetuned `cA_NNN` / `cB_NNN` go to `real_time_data/sub{S}/ses{N}/checkpoints/` ([`config.getSessionCheckpointDir`](../python/main/Utils/config.py#L152)); [`get_next_version_path`](../python/main/Utils/some_functions.py#L27) is highest + 1, first file 001. New [`--start_from session|base|subject`](../python/main_start.py#L68) in [`setup_args()`](../python/main_start.py#L61) through [`resolve_start_checkpoint`](../python/main/Utils/some_functions.py#L38).
- **Docs and tests**: [README](../README.md) section 4 rewritten; [`test_parallel_finetune.py`](../python/tests/test_parallel_finetune.py) now 24 tests (was 10), covering scoring, gate matrix, streak, vote flag, swap wait, `--start_from`, numbering, `predictions.csv`, legacy path, failure rule, `trial_ids`, per-model weights.
- **Replay check** on `real_time_data/sub01/ses1/run1` and `run2` logs: CUT counts 23 / 16 equal the live results from `score_block`.

## 2026-10-14 (first commit of this branch): earlier changes in this work

- **`torch.load(..., weights_only=False)`** in the 6 calls in [`python/main`](../python/main) that read `.pt` / checkpoint files with numpy arrays (needed with PyTorch 2.6 and later).
- **`MODE` on START lines**: Unity writes `MODE: Adaptive|Static` on `Trial N START` ([`TimeLogger.cs`](../unity_noVR/Assets/Script/FileProcess/TimeLogger.cs#L80)); Python keeps it in [`global_value.run_mode`](../python/main/Utils/global_value.py#L49) and writes a `mode` column in `run{N}/ab_blocks.csv`.
- **`--subject` / `--session`**: run folders are `real_time_data/sub{S}/ses{N}/run{k}`; run numbering continues in the same session after a Python restart.
- **`--start_model`**: which model controls the game first when `--live_model auto`.
- **`setup_args()`** in [`main_start.py`](../python/main_start.py#L61) holds all command line arguments.
- **README replay buffer numbers**: 384 windows per class, validation cap 153 per class (40%).
