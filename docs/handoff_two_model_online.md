# Handoff: two-model (A/B) online system, to continue on the laptop with Unity

Written 2026-10-12 on `ws-4090-02`. Read this first, then `README.md` section "4. 雙模型 A/B 平行微調系統".

## 1. What is done (all tested with synthetic data only)

Branch `verify`, 15 commits ahead of `adenchong/main`, **not pushed** (this workstation has no GitHub SSH key).
The mine-only commits are `655cb5e` (the system), `e60df21` (`switch_streak` config + `cA_` chain prefix), `32261bd` (README).

- A and B finetune in two threads on every block; only the live model sends predictions to Unity.
- A = `cA_000.pth`, pretrained, 22 ch. B = `cB_000.pth`, random, 32 ch. Both live in `python/main/EEG/checkpoint_main/` and are **gitignored** (`*.pth`): copy them to the laptop yourself.
- `python python/main_start.py --live_model A|B|auto` (default A). `auto` switches when the other model has strictly higher trial accuracy on `switch_streak` consecutive blocks (`config.json` → `online_adaptation.switch_streak`, default 2).
- Trial accuracy = majority vote of the model's live predictions between the trial's START and END.
- Each `Trial N END` log line gets `MODEL_PREDICTION: A=0 B=1 LIVE=A AT: <END-0.47>` (Unity-style 100-prediction vote).
- A channel/shape mismatch now raises instead of overwriting the checkpoint.
- Tests: `/opt/anaconda3/envs/AdenVRBCI/bin/python python/tests/test_parallel_finetune.py` (9 tests; adjust the interpreter path on the laptop). Not run anywhere except this workstation.
- Full change list: `python/Aden/report/two_model_online_report.md` (gitignored, on the workstation only) and the README section.

Uncommitted in the working tree: your `unity_noVR` edits (`AutoSaber.cs`, `SaberSlicer.cs`). Not part of any commit.

## 2. The open problem to verify on the laptop (log format change)

Your `unity_noVR` edit moved CUT logging. Reading the code (no new-format log has been seen yet):

- **Before:** `SaberSlicer.SliceCube` logged a CUT for every sliced note, so a successful trial had 5 CUTs (the 5th = the brain-controlled note).
- **Now:** `SaberSlicer` no longer logs. `AutoSaber.cs` logs a CUT right after the brain vote (before the saber moves), only if the vote is correct, and only inside `if (GM.use_LSL_to_controll_saber)`.
  So a trial should have **0 or 1 CUT**. In `unity_noVR/.../MI.unity` the scene default `use_LSL_to_controll_saber` is **0**, which would log no CUT at all. The `auto_saber` object has `swing_time: 1`.

What this breaks in Python (not fixed yet):

1. `python/main/EEG/data_process_np.py:89`: `is_fail = len(cut_list) < config.group_note_num` (5). With the new log every trial is a failure: all windows get weight 2.0, and `pretrain_cross_subject.py --exclude_failed` drops every noVR trial. `python/else/preprocessing/create_MInp.py:96` has the same rule.
   `group_note_num` cannot be reused for the fix: Unity reads it as the note-group size.
2. `MODEL_PREDICTION` uses `AT = END - 0.47`. That offset was measured for the old CUT, which was logged at collision (END minus 5th CUT = 0.466 s, sd 0.017, n = 6,225 five-CUT trials across 378 old logs). The new CUT is logged at the vote itself, so for correct trials the exact time is the CUT timestamp. Wrong trials have no CUT; they need a fallback offset that has to be measured from new logs.
   The old offset also ignores that the vote happens `swingDuration` (0.2 s in the Calibration/MI auto saber) before the collision.

Proposed fix (agreed in principle, not implemented):
- A new config key for the number of CUTs that count as success (5 for old data, 1 for new noVR logs), used by `is_fail`.
- `UnityMarkerReader` remembers the trial's last CUT time; at END, `AT` = that time if it exists, else `END - fallback` (new config value, set from new logs).
- Then update the README section and the report text.

### Checks to do with Unity running
1. Run one calibration/MI session with LSL control on. Open `real_time_data/run{N}/log_*.txt`: count CUT lines per trial (expect 0 or 1) and the time between the CUT and END (this gives the fallback offset; use the median).
2. Confirm `use_LSL_to_controll_saber` is on during the run (the scene default is 0).
3. Confirm only the live model's predictions reach Unity (B is silent), and that Unity still draws a saber swing from the 0/1 stream.
4. Look at `run{N}/ab_blocks.csv` after a calibration: per-block correct counts for A and B, live model, checkpoint paths.
5. Check whether the game pauses until "calibration done" arrives. No repo script reads `calibration_model_finish`; it may be wired in a scene.
6. Clock sync: scoring compares Unity's START/END timestamps with Python's `time.time()`. If Unity runs on another device the clocks must agree, or scoring finds no predictions (the switch streak then never advances).

## 3. Laptop environment (Core 7 240H, 16 GB, RTX 5050)

- The repo pins `torch==2.3.0+cu121` (README). Its GPU architecture list stops at sm_90, so it does **not** support the RTX 5050 (needs PyTorch 2.7+ with CUDA 12.8). Untested on the real GPU.
- Zero-change option: hide the GPU (`$env:CUDA_VISIBLE_DEVICES="-1"` in PowerShell) and run on CPU. Measured on the workstation (limited to 8 or 4 threads): finetune A alone about 1.1–1.7 s, A and B in parallel about 2.5–3.5 s with a full replay buffer. Expect roughly 1–4 s on the laptop.
- Upgrading torch changes the environment (get explicit OK first) and requires `weights_only=False` for the six `torch.load` calls in `python/main` that read `.pt` files with numpy arrays.
- Memory: Python about 1.2 GB, replay buffers about 100 MB for both models. 16 GB is enough unless Unity is a heavy editor session.
- The AdenVRBCI interpreter path in this note is the workstation's. Check the laptop's env.

## 4. How validation works online (for reference)

`python/main/EEG/MI_train.py`, `add_to_buffer` (l.250) and `online_train` (l.275). `use_val` is true in `config.json`.

- Each new window goes into the per-class training buffer (limit 320 windows/class). While the buffer is not full, **the same window is also copied into the validation buffer**. Once the buffer is full, the oldest training window is evicted and moved into the validation buffer. Validation is capped at 128 windows/class (`int(320 * 0.4)`, hardcoded at l.271).
- Measured with synthetic windows: after 1 and 2 blocks 100% of the validation windows are also in the training buffer; once the buffer is full only the evicted windows are new (224/256 still overlapped one block after filling, 32/256 two blocks later).
- Each calibration trains 4 epochs on the whole training buffer (Adam, lr 0.002 = `online_adaptation.learning_rate`; until 2026-10-27 the config held 0.001 and the trainer doubled it, same effective rate; batch 8, failed windows weight 2.0). After each epoch the model is evaluated on the validation buffer (weighted loss). The weights with the lowest validation loss are kept and saved as `ftA-best.pth` or `ftB-best.pth`, then copied to the next `cA_NNN.pth` / `cB_NNN.pth` with `n_chans` added. If there are no validation samples, the training loss is used.
- There is no early stopping and no reported validation accuracy online. Because validation is mostly training data (or older data the model already trained on), it is a weak guard against overfitting. In practice it tends to pick a late epoch. Each model has its own validation buffer.

## 5. Capacity and other numbers

- Replay buffer: 320 windows/class (`eeg_settings.replay_buffer_limit`), separate for A and B. A trial is 11–16 windows (500-sample windows, stride 100; 16 only when the trial is at least 2000 samples). So the training buffer holds about 20–30 trials/class, about 40–60 total. For 48 trials per run, 24 per class × 16 windows = 384.
- Block size is unlimited (trials accumulate until the next trigger). Unity triggers a calibration only when wrong_hit changed and the score is below 0.75, so blocks are 4, 8, 12 … trials and are only scored when the live model is doing poorly (biases the A/B comparison toward switching).
- Prediction log: last 60,000 entries per model (about 11 minutes at about 90 predictions/s).
- Within one trial the CUT gaps are nearly constant (median spread 0.023 s) but differ between songs (about 0.7–0.9 s). START → 1st CUT is 0.54 s on average. 4-CUT trials are not reliably "missing the 5th CUT": END minus (4th CUT + own gap) averages 0.28 s with sd 0.43, so many miss an earlier CUT.

## 6. Pending actions

- **Push.** Run where your GitHub key works. Recommended: `git push adenchong verify`, then merge into `main` on GitHub once the log-format fix is done. `git push adenchong verify:main` would be a fast-forward onto `main` according to the local ref (the remote could not be fetched from the workstation).
  Chungyen's work lives in another repo (probably `stereomp3/VR_BCI`); pushing to `adenchong/BeatSaber` does not change his account's repo. If he pulls from yours he gets the new online behaviour (needs `cA_000.pth` / `cB_000.pth`, mismatch raises).
- The report builder was moved to `python/Aden/report` (gitignored, so no longer on GitHub; still in history, e.g. `git show 72ffa34:python/else/pretraining/report/build_report.py`).
- Stale `c_000.pth` mentions remain in: `python/tools/online_test_lsl_to_unity.py` (default path), `python/main/EEG/generate_random_models.py`, `python/else/pretraining/pretrain_cross_subject.py` docs. Out of scope so far.
- Optional: make the validation fraction (0.4) a config key.

## 7. Suggested first message for the laptop session

> Read docs/handoff_two_model_online.md. I ran a noVR session with the new CUT logging; here is a log from run{N}. Verify the CUT count per trial and END−CUT timing, then implement the Python fix described in section 2 (failure rule config key, MODEL_PREDICTION anchored at the last CUT with a measured fallback), update the README and tests, and run the test file.
