# Known issues

Issues of the A/B online system. Moved here from `docs/plan_ab_online_scoring_gating.md` on 2026-10-18. Add new issues to the summary and details; when one is fixed, move it to [Fixed issues](#fixed-issues) and note the fix in `docs/CHANGELOG.md`.

**Severity**
- **High**: can make the results or the study's conclusion wrong without any visible sign.
- **Medium**: lowers model quality, wastes data or delays adaptation. Visible in the logs, or it has a workaround.
- **Low**: rare, display-only, an edge case or out of scope.

## Summary (most severe first)

| ID | Severity | Issue | Area |
|---|---|---|---|
| H1 | High | Live-model switching is noisy | Python, switching |
| H2 | High | Switching rule not yet tested with a real brain signal | Study |
| H3 | High | Validation data is mostly training data | Python, training |
| M1 | Medium | Finetune is slower than a trial cycle | Python, training |
| M2 | Medium | Training ignores the 0.25 s just before the vote | Python, data |
| M3 | Medium | Replay buffers are empty after a Python restart | Python, training |
| M4 | Medium | A song started while the label is red is not trained (added 2026-10-18) | Unity, connection |
| M5 | Medium | SPRT finetune gate parameters chosen before any SPRT data (added 2026-11-16) | Python, training |
| L1 | Low | Main loop polls once per second | Python, timing |
| L2 | Low | TCP messages have no framing | Unity + Python, TCP |
| L3 | Low | Vote with 0 or 1 predictions | Unity, vote |
| L4 | Low | Messages kept while disconnected can be stale (added 2026-10-18) | Unity, connection |
| L5 | Low | Unlocked reads of `valueHistory` in display scripts | Unity, display |
| L6 | Low | No disconnect panel during a song | Unity, UI |
| L7 | Low | No connection watchdog with simulated EEG | Python, connection |
| L8 | Low | No scoring in Static runs; `unity/` (VR) copy not updated | Scope |

## Details

### H1. Live-model switching is noisy (High)
- **What:** in `--live_model auto`, Python gives control to the other model when it scores strictly higher on 2 blocks in a row (`_decide_live` in `python/main/EEG/EEG_Train.py`, `switch_streak` = 2).
- **Why:** a block has only 4 trials, so each score is 0, 25, 50, 75 or 100% and luck dominates. Two chance-level models trigger a switch in about 13% of block pairs.
- **Effect:**
  - The live model can change because of chance, not because the other model is better.
  - The live model is the one controlling the saber and the one whose CUT/WRONG_CUT is Unity's real result. So noise here changes both what the subject sees and how each model is scored.
- **Considered:** a rolling window over more trials. Dropped for now, because its window size and margin are subjective choices.
- **Fix ideas:** larger blocks, a longer streak, or a statistical test on the accumulated trials (for example a binomial test on the difference).

### H2. Switching rule not yet tested with a real brain signal (High)
- **What:** no recording so far shows whether switching picks the better model.
- **Evidence:**
  - The 2026-10-14 runs did not use a real brain signal. Both models were at or below chance and agreed on about 75% of votes.
  - The 2026-10-15 recording used a real signal, but with `--live_model A`, so switching was off. B scored higher than A in 5 of 7 runs: runs 2, 3, 4, 6 and 7, for example run 7 at 40 vs 46 of 48. A was higher in run 1 (65 vs 56 of 96) and run 5 (38 vs 37).
- **Effect:** the A/B design's main decision is unverified.
- **Next step:** run a real-signal session with `--live_model auto` and check:
  - whether the live model changes;
  - when it changes;
  - whether the accuracy after a switch is higher.

### H3. Validation data is mostly training data (High)
- **What:** each block's finetune runs 4 epochs and keeps the epoch with the lowest validation loss (`OnlineCalibrationTrainer.train` in `python/main/EEG/MI_train.py`). Validation comes from the replay buffer's validation list.
- **Why:**
  - While a class has at most 384 windows, every new window is added to both the training list and the validation list.
  - Only windows pushed out of a full training list are validation-only.
  - 384 windows per class is about one run (48 trials). So during the first run, validation is the same data as training.
- **Effect:**
  - The "best validation epoch" is in practice the best training epoch.
  - Overfitting can't be detected, and checkpoint selection gives no real protection.
  - Combined with 4 epochs over the whole buffer at lr 0.002 per block, models can drift with no warning.
- **Fix ideas:**
  - hold out whole trials (not windows) for validation;
  - or validate on the next block before training on it.
- **Tested (2026-10-28), offline (F1b, sub01 + sub05, 61 finetunes, 5 seeds, H4-correct cut):** both fix ideas did **worse** than the current training, measured as accuracy change vs the parent on the next 48 trials:
  - V1, whole held-out trials, and the parent is kept if no epoch beats it: −1.9 pp vs current (95% CI −3.2 to −0.7). It kept the parent in 65% of retrains, so it blocked useful updates too.
  - V2, validate on the next block before training on it, accept only if not worse: −1.0 pp vs current (CI −1.8 to −0.2).
  - The early drop of A's chain on sub01 was mainly H4, not H3. Keep the current validation for now; the issue stays open because validation still gives no real guard.
  - Output: `python/analysis/results/finetune_eval/f1b/summary.txt`.

### M1. Finetune is slower than a trial cycle (Medium)
- **What:** a block's finetune (both models in parallel) takes longer than one trial (6 s).
- **Evidence:**
  - 2026-10-14: 2.5 s at block 1, growing to 9–10 s (peak 13.86 s, run 4 block 27).
  - 2026-10-15: up to 10.8 s for normal blocks, and 19.2 s for the one 48-trial block.
  - The time grows as the replay buffer fills, because each finetune trains on the whole buffer.
  - It's measured in `ab_blocks.csv` (`ft_seconds`) but not reduced.
- **Effect:**
  - New weights load 1–2 trials into the next block, so those trials still use the old weights.
  - While a finetune runs, the main loop is blocked. A long one delays the next block's save; that's how a trial was lost before the 2026-10-16 pointer fix.
- **Fix ideas:**
  - fewer epochs, or a smaller or capped buffer;
  - train on a subsample of the buffer;
  - run the save on its own thread.

### M2. Training ignores the 0.25 s just before the vote (Medium)
- **Note (2026-10-22):** the premise is wrong because of H4. In wall-clock time the training cut runs to about START + 5 s, past the vote. Fix H4 first, then revisit this.
- **Tested (2026-10-27), offline on sub01 ses1:** leave-one-run-out, 5 seeds, scored by Unity's vote. Training on the 4 s ending at the vote is **not better** than the intended 0–4 s: −1.2 pp pooled (95% CI −2.7 to +0.3; A −1.7, B −0.7). The ses1 online cut (~1–5 s, H4) vs intended: intended +1.4 pp (CI −1.0 to +4.2). All differences are within training-seed noise (sd 1–2 pp). Keep the 0–4 s cut that the H4 fix gives. Output: `python/analysis/results/cut_compare/*/summary.txt`.
- **What:** each trial is cut to its first 2000 samples (4 s at 500 Hz) for training (`python/main/EEG/data_process_np.py:99-100`). Unity votes at START + 4.24–4.26 s.
- **Effect:**
  - The predictions that decide the vote come from windows that include the last 0.25 s, which the models never trained on.
  - It's a systematic mismatch between training and use. Its size is unknown.
- **Fix idea:** cut trials up to the vote time (CUT or WRONG_CUT timestamp) instead of a fixed 4 s.

### M3. Replay buffers are empty after a Python restart (Medium)
- **What:** each model's replay buffer lives only in memory (`ReplayBuffer` in `MI_train.py`). It is never saved, so it's empty after a restart, while the checkpoints are resumed (`--start_from session`).
- **Effect:**
  - The first finetune after a restart trains on one block only: 4 trials = 64 windows, 4 epochs at lr 0.002.
  - That can pull a well-trained resumed model towards a few trials.
- **Workaround:** avoid restarting Python within a session.
- **Fix ideas:**
  - save the buffers with the checkpoints;
  - or refill them from the session's `data_*.pt` files at start.

### M4. A song started while the label is red is not trained (Medium, added 2026-10-18)
- **What:**
  - The `MI` stage message that moves Python into a song comes from the lobby (`SceneLoaderManager.LoadScene`), sent just before the scene changes.
  - If the lobby isn't connected at that moment, the message is held as pending. Then the lobby is destroyed by the scene change, so the message is lost.
  - The song scene doesn't send a stage message: its `now_stage` (2) has no entry in `Config.Stage`, so `GameManager.Start` now skips the send.
- **Effect:**
  - Python stays in Lobby for the whole song. It prints `WARNING: Trial data received in Lobby` every 4 trials, and skips those trials when the next song starts.
  - The song's data is logged but not trained or scored. The subject's time is wasted, though nothing is corrupted.
- **Workaround:** start Python first, open `game.exe` after `歡迎來到遊戲大廳！`, and start a song only when the label is green.
- **Fix ideas:**
  - let the song scene send `MI` itself once connected: give `MI` / `MI_Cross` `now_stage: MI`, or add the stage string, after checking how Python reacts to it;
  - or keep one `TCP_Client` across scenes (`DontDestroyOnLoad`).

### L1. Main loop polls once per second (Low)
- **What:** `Game.start` (`python/main/game_state.py`) checks the state, then sleeps 1 s.
- **Effect:**
  - Up to 1 s passes between a block's last END and the start of its save and finetune.
  - Before the 2026-10-16 pointer fix, this left only about 50 ms of margin before the next START (which comes 1.05 s after END). The fix removed that risk.
  - Now it only delays each finetune by up to 1 s.
- **Fix idea:** wake the loop on an event when the block flag is raised.

### L2. TCP messages have no framing (Low)
- **What:** both sides read up to 1024 bytes per call (`conn.recv(1024)` in `TCPServer.py`, `stream.Read` in `TCP_Client.cs`) with no message separator. Two messages sent close together can arrive as one.
- **Effect:**
  - Unity accepts a prediction only if the read is exactly `"0"` or `"1"`, so a merged read such as `"00"` is dropped.
  - On Python's side, a merged read would hit an exact-match check and break it.
- **Evidence (2026-10-15):**
  - Unity: 2 merged reads out of about 970,000 prediction messages (`Player.log`).
  - Python: 0 merged trial messages.
- **Mitigation:** since 2026-10-16, `TCP_Client` spaces the messages it sends after (re)connecting by 50 ms.
- **Fix idea:** add a newline delimiter on both sides and split on it.

### L3. Vote with 0 or 1 predictions (Low)
- **What:** `AutoSaber` weights the predictions it holds.
  - With 0 predictions, it compares 0 ≥ 0 and always picks the left saber.
  - With 1 prediction, the weight uses `(float)ii / (n - 1)`, a division by zero. The arithmetic then works with NaN and the result is meaningless.
  - Python's replica (`trial_scoring.unity_vote`) returns "no vote" when there are fewer than 2 predictions.
- **Effect:** Unity's result and Python's replica disagree for that trial. This can only happen in the first moments after a connection. Normally the history (100 predictions, about 45 per second) is full long before the first vote at START + 4.25 s.
- **Fix idea:** in `AutoSaber`, skip the vote, or log it as "no vote", when fewer than 2 predictions are available.

### L4. Messages kept while disconnected can be stale (Low, added 2026-10-18)
- **What:** since 2026-10-16, `TCP_Client` keeps up to 50 messages sent while disconnected and sends them after it connects. The list belongs to the scene's client and is lost on a scene change.
- **Effect:** if Python is down during a song and starts mid-song, it receives up to 50 old trial lines. They land in the run log while Python is in Lobby: the Lobby warning fires, and they are skipped when the next song starts. No training effect, just old lines in the log.
- **Fix idea:** drop pending messages older than a few seconds.

### L5. Unlocked reads of `valueHistory` in display scripts (Low)
- **What:** `LSLVisualizer.cs` and `ScaleRangeController.cs` read `valueHistory` with `foreach` every frame without the lock. `AutoSaber` and `update_history` use the lock since 2026-10-16.
- **Effect:** a prediction arriving during the read throws `InvalidOperationException` and skips that frame of the on-screen bar. In the 2026-10-15 `Player.log` this happened 118 times (`LSLVisualizer`) and 2 times (`ScaleRangeController`). It is display only: no effect on votes or data.
- **Note:** `Calibration.cs` has the same pattern but is not used by any scene.
- **Fix:** the same as `AutoSaber`, copy under `lock (valueHistory)`.

### L6. No disconnect panel during a song (Low)
- **What:** if the connection drops during a song, Unity only turns the top-right label red, switches to simulated mode and retries every 1 s. Python prints `WARNING: Connection to unity lost`.
- **Effect:** the subject keeps playing. Votes during the gap are not brain-controlled, and there is no clear prompt to stop.
- **Later improvement:** a pause panel shown only after a connection existed, so UI-only testing never sees it.

### L7. No connection watchdog with simulated EEG (Low)
- **What:** the watchdog starts when `LSL.setup_lsl_inlet` prints `Connected to LSL stream`. With `is_simulated_eeg` (`config.json`), that line never runs.
- **Effect:** no "Waiting for Unity connection" or "Connection to unity lost" warnings in simulated-EEG tests.
- **Fix idea:** also start the watchdog when simulated EEG starts.

### M5. SPRT finetune gate parameters chosen before any SPRT data (Medium, added 2026-11-16, updated 2026-11-17)
- **What:** the default `finetune_gate = "sprt"` uses p0 = 0.65, p1 = 0.85, α = 0.05, β = 0.2 (`online_adaptation` in `python/config.json`; 0.5 / 0.75 until 2026-11-17). No session has run with the SPRT gate yet. The values come only from recorded block-gate data:
  - a simulation with fixed model accuracy;
  - a replay of the recorded sessions;
  - the F2 finetune effect by model accuracy: it helps below about 0.75 and is about 0 at 0.80 or above.
- **Effect:**
  - **Replay** (sub01–05 ses1, `python/analysis/sprt_gate_replay.py`): the SPRT gate would fire 80 finetunes (A 39, B 41) where the block gate fired 166 (A 88, B 78).
  - **Same blocks, F2 counterfactuals:**
    - SPRT finetunes help about as much as block-gate finetunes: +2.6 against +2.5 pp on the next 48 trials.
    - They are no more selective: 60% against 62% of fires help.
    - They catch fewer of the blocks where a finetune would help: 19% against 53%.
  - **Limit:** the replay keeps the recorded models, so it can't show the chained effect of finetuning at different times. Whether the SPRT gate helps or hurts online accuracy is not known.
- **Next step:** after the SPRT data collection, review the parameters with `run{N}/sprt_trials.csv` and `ab_blocks.csv` (trigger, sprt_A, sprt_B): how often each model crossed each line, accuracy after SPRT finetunes vs the F2 results for the block gate.

### L8. No scoring in Static runs; `unity/` (VR) copy not updated (Low, scope)
- **What:**
  - Static runs (other than the calibration song) are not scored or finetuned, by design.
  - All Unity changes of this work (`WRONG_CUT`, `MODE`, the reconnect and status label, the AutoSaber lock, the stage guard) are only in `unity_noVR/`. The VR project `unity/` has none of them.
- **Effect:**
  - Static runs can be compared only offline, from `predictions.csv` and the log.
  - The VR build runs on Python's older path. Without `MODE`, Python follows Unity's own block trigger. Without `WRONG_CUT`, a wrong trial is only known from its END line (vote time estimated as END − 0.70 s).
- **Fix:** port the `unity_noVR` script changes to `unity/` when the VR setup is used again.

## Fixed issues

| ID | Severity | Issue | Fixed | Area |
|---|---|---|---|---|
| H4 | High | Training trials were cut about 1 s late, past the vote | 2026-10-27 (code), recordings before that still affected | Python, recording / data |

### H4. Training trials were cut about 1 s late, past the vote (High, added 2026-10-22, fixed 2026-10-27)
- **Fix (2026-10-27, commit `19e952e`):** the reader connects to LSL first and writes `Record datetime` as the first sample's own Unix time (`fsample_ts`; see `CHANGELOG.md`). Checked with an LSL loopback test (max error 0.57 ms; the old code was off by 961 ms).
- **Still to do:**
  - Confirm with the real amplifier: on the first session after the fix, run the step 0a lag search; the best lag should be about 0 s.
  - Recordings made before the fix (sub01–sub05 ses1) keep the offset. Correct them per run with the lag from `predictions.csv` (step 0a), and regenerate `data_*.pt` from the CSV if they are used for training. Runs without `predictions.csv` can't be corrected exactly.
- **What it was:** the training cut of each trial (`data_*.pt`, every finetune of A and B) covered about START + 1 s to START + 5 s instead of START to START + 4 s.
- **Why:**
  - `CygnusEEGReader.build_csv_file` wrote `Record datetime = datetime.now()` when the file was created, before `read_eeg` connected to the LSL stream (`setup_lsl_inlet`, up to a 2 s timeout).
  - Each row's `Timestamp` is `ts − first_ts_lsl`: 0 is the first sample received, not the file creation.
  - `data_process_np._extract_record_epoch` and `file_pointer_reader.extractRecordEpoch` treat `Record datetime` as the time of row 0. So the trial cut `START − rec_epoch` landed on samples that arrived about 1 s after START.
- **Evidence (sub01 ses1, 2026-10-15):**
  - Re-predicting the EEG file offline matches the logged live predictions only when the file is shifted by +0.93 to +1.05 s (per run; run 7: 0.978 s). At 0 s, agreement is 0.60–0.65 (run 7); at the fitted offset, 97% (all runs: A 96.9%, B 96.8%).
  - The offset is constant within each run (94–99% agreement for every checkpoint), so it's a start offset, not a growing backlog.
  - The offline cut with the 0 s alignment equals `x_data` in `data_*.pt` exactly (12 trials, max difference 0.0).
  - Scripts: `python/analysis/cut_compare/step0_reproduce.py`, output `python/analysis/results/cut_compare/step0a_summary.txt`.
- **Effect:**
  - About 0.75 s of every training trial was EEG after the vote (START + 4.25 s), when the subject sees the saber move. That feedback is correlated with the label whenever the vote is right.
  - The first ~1 s of each trial was never trained on, though the logged predictions are already ~72% correct for windows ending at START.
  - Training and live prediction saw different parts of the trial. Live prediction was not shifted: it reads the live buffer, stamped with `time.time()`.
  - Affects every session recorded before the fix, and any offline analysis that cuts those trials from the CSV with `Record datetime`. The base checkpoints may be affected too if they were trained on data recorded this way (unknown).
  - Supersedes the premise of M2: training did not stop 0.25 s before the vote, it ran ~0.75 s past it.
- **Measured cost (2026-10-28, F1b, sub01 + sub05, 61 finetunes, 5 seeds):** retraining each saved finetune on the correct cut instead of the late one gains +1.5 pp per update on the next 48 trials (95% CI +0.2 to +2.8; A +2.4, B +0.6). With the correct cut, a single finetune helps by +2.9 pp vs its parent (CI +1.6 to +4.3); the saved finetunes gave +1.4 pp (CI −0.1 to +3.1). The drop of A's early finetunes on sub01 (cA_004–006: 0.65–0.67 on run 7) is gone with the correct cut (0.73–0.75). Output: `python/analysis/results/finetune_eval/f1b/summary.txt`.
