# Plan: Is there an ErrP after wrong trials? (sub01 / ses1, 2026-10-15)

## Context
Question: when the brain vote is wrong (`WRONG_CUT`), does the subject's EEG show an error-related potential (ErrP) compared with correct votes (`CUT`)? If yes, it could later serve as an implicit label for online finetuning.

### What the data contains (checked)
- `python/main/real_time_data/sub01/ses1/run1..run7`: continuous EEG `eeg_record_*.csv` (500 Hz, 32 ch incl. Fz/FCz/Cz/CPz, µV, unfiltered, DC offsets in the thousands), `log_*.txt` (START / CUT or WRONG_CUT / END, unix time), `predictions.csv`, `ab_blocks.csv`. run8 has an empty log and is skipped.
- Trials (counted from the logs): **288 correct / 96 wrong** (run1 65/31, run2 38/10, run3 36/12, run4 36/12, run5 38/10, run6 35/13, run7 40/8). Live model was A throughout (`--live_model A`); the subject only ever saw A's vote, so A's CUT/WRONG_CUT is the error label.
- The "Simulating Cygnus" header is hard-coded text in `CygnusEEGReader.build_csv_file`, not proof of simulated EEG; `docs/known_issues.md` H2 says 2026-10-15 was a real signal. The plan still checks this (step 1).
- Environment `VRBCIenv` already has mne 1.12.1, scikit-learn 1.7.1, scipy, pandas, matplotlib. **No installs needed.**

### Two issues that limit what we can conclude (must be stated in the results)
1. **Visual confound — the feedback is not symmetric.** `AutoSaber.cs:105-117`: on a wrong vote Unity logs `WRONG_CUT` and `break`s — the saber does **not** swing and the cube flies past. On a correct vote the saber swings and the cube is sliced (pieces fly, fade over 2 s, `SaberSlicer.cs`). So "wrong − correct" = error **plus** "no motion vs. motion" **plus** rarity (25% vs 75%, P300-like). We cannot remove this by design; we can only check whether the difference looks like an ErrP (fronto-central negativity ~200–350 ms then positivity ~300–600 ms at FCz/Cz) or like a visual/motion response (occipito-parietal, O1/Oz/O2/P7/P8). Also, on wrong trials the error becomes evident gradually (cube passing), so the ErrP may be later and smeared.
2. **EEG ↔ log time alignment is approximate.** EEG time 0 = first LSL sample, but `Record datetime` = `datetime.now()` taken *before* the LSL inlet is resolved (`CygnusEEGReader.read_eeg`, up to 2 s timeout). That leaves an unknown constant offset per run (same one the training pipeline uses). For ERPs this matters: a few hundred ms shift moves latencies and, if it differs across runs, smears the average. Step 1 measures it.

## Execution
- First, copy this plan to `docs/plan_errp_analysis.md`.
- Then hand the implementation to a **Sonnet** subagent (`model: sonnet`) with this plan as its brief. It writes the script and tests, runs them, and reports results. I check its numbers against the files before reporting to you.

## Approach
One script `python/analysis/errp_analysis.py` (style of `graded_scores.py`: argparse `--session`, `--out`, matplotlib Agg, `summary.txt` + figures), plus a test file. Results to `python/analysis/results/errp/`.

### Step 1 — Load, align, sanity checks
- Read EEG with pandas (`skiprows=11`, much faster than the line loop for 90–210 MB files); record epoch computed exactly as `EEGDataLoader._extract_record_epoch` (`python/main/EEG/data_process_np.py:67`) so we match training; trial parse with the same regex as `data_process_np._parse_log_file`.
- Checks written to `summary.txt`:
  - sample-interval stats / gaps in the Timestamp column (dropped samples);
  - EEG looks real: channel spectra with 1/f + alpha peak (~10 Hz, O1/Oz/O2) and 50/60 Hz line; not white noise (the simulator emits uniform random ints ±10000);
  - CUT/WRONG_CUT count per run vs the counts above; vote at START + ~4.25 s.
- **Alignment check**: per run, ERP time-locked to START (cube spawn is a strong visual onset) and to CUT on correct trials (saber motion). Compare peak latencies across runs. If runs disagree by >~50 ms, report it and add an option `--align xcorr` that shifts each run by the lag maximising cross-correlation with the pooled START-ERP (a relative fix only; absolute latency stays uncertain).

### Step 2 — Preprocess and epoch
- Per run, continuous: drop DC (detrend), band-pass 1–10 Hz (zero-phase FIR, MNE), no notch needed after a 10 Hz low-pass; common average reference (the cap has no mastoid channels). Keep a 0.1–30 Hz version for figures.
- Epochs around the vote (CUT / WRONG_CUT timestamp): −0.5 to +1.0 s, baseline −0.2–0 s.
- Artefacts: reject epochs with peak-to-peak > 150 µV (after filtering) on any channel; report rejections per class; Fp1/Fp2 kept as an eye-movement monitor (on correct trials the eyes may track the saber).
- Output MNE `Epochs` saved as `errp-epo.fif` with metadata (run, trial, label, correct, A/B votes).

### Step 3 — ERP analysis (does an ErrP exist?)
- Grand averages correct vs wrong and difference wave at Fz, FCz, Cz, CPz, Pz, plus O1/Oz/O2 for the visual check; ±95% bootstrap CI over trials.
- Topographies of the difference at 150–250, 250–350, 350–500, 500–700 ms.
- **Statistics:** spatio-temporal cluster permutation test over trials (`mne.stats.spatio_temporal_cluster_test`, 0–1.0 s, 32 ch, 5000 perms; independent groups = trials). Primary hypothesis-driven test: mean amplitude at FCz in a negativity window (200–350 ms) and Cz positivity window (300–600 ms), Welch t-test + permutation p.
- Because it is one subject, the inference is over trials ("within this subject"), not population.

### Step 4 — Controls against confounds
- **Label balance:** repeat the difference within left-MI and within right-MI trials (errors may be unevenly split by label; MI lateralisation could leak into the contrast).
- **Pre-vote window** (−1.0–0 s, no baseline): if correct vs wrong already differ before the vote, part of the post-vote effect is pre-existing brain state, not a response to the error.
- **Run / time-on-task:** run1 has 31 of 96 errors. Repeat the test excluding run1 and with run as a stratum in permutations (shuffle labels within run).
- **Topography check for the visual confound:** report whether the peak of the difference is fronto-central (ErrP-like) or occipito-parietal (VEP-like), and the Fp1/Fp2 difference (EOG-like).
- **Sanity null:** same pipeline with labels shuffled within run → should give no cluster.

### Step 5 — Single-trial detection (is it usable?)
- Features: mean amplitude in 50 ms bins, 150–700 ms, (a) 7 fronto-central channels (Fz, FC3, FCz, FC4, Cz, CPz, Pz), (b) all 32 channels with `mne.preprocessing.Xdawn` (2–4 components) fitted inside each fold.
- Classifier: shrinkage LDA (sklearn `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`), class-weighted.
- CV: leave-one-run-out (7 folds) — no leakage across runs. Metrics: ROC-AUC, balanced accuracy, TPR/TNR. Significance: 1000 within-run label permutations.
- Also: classifier trained on a pre-vote window as a control (should be ~chance if the effect is truly post-feedback).

### Outputs (`python/analysis/results/errp/`)
- `summary.txt` — all numbers with their source (counts, rejections, alignment lags, cluster p-values, window t-tests, AUCs with permutation p).
- `e1_sanity.png` (spectra, sample intervals), `e2_alignment.png` (START/CUT ERPs per run), `e3_erp.png` (correct vs wrong + difference with CI), `e4_topomaps.png`, `e5_cluster.png` (significant cluster mask), `e6_controls.png` (per label, excl. run1, pre-vote), `e7_decoding.png` (ROC per fold, permutation null).
- `epochs_meta.csv` (one row per trial: run, trial, label, correct, rejected).
- Cached `errp-epo.fif` (gitignored with results).

## Files
- New: `python/analysis/errp_analysis.py` (top comment: `# C:\Users\aiden\anaconda3\envs\VRBCIenv\python.exe python/analysis/errp_analysis.py --session python/main/real_time_data/sub01/ses1 --out python/analysis/results/errp`).
- New: `python/tests/test_errp_analysis.py`.
- Reuse: record-epoch / log-parse logic from `python/main/EEG/data_process_np.py`; style helpers (`style`, colours) from `python/analysis/wald_sprt_explainer.py` as `graded_scores.py` does; `LINE_RE` pattern from `graded_scores.py` for A/B votes in metadata.
- Docs: short entry in `docs/CHANGELOG.md`; results interpretation note in `docs/` only if you want one.

## Verification
- Unit tests (pytest) on synthetic data: build a fake 32-ch CSV + log with known header/epoch, inject a Gaussian negativity at FCz 250 ms after WRONG_CUT only → check (1) epochs are aligned to within 1 sample, (2) the window test and cluster test detect it, (3) a no-injection version gives p > 0.05, (4) leave-one-run-out never mixes runs.
- Run the full script on ses1; cross-check counts in `summary.txt` against the log counts above (288/96); re-read figures before reporting.
- Report honestly: if the effect is occipital or present before the vote, say it is not evidence of an ErrP.

## Out of scope / follow-up suggestions
- A clean ErrP test needs symmetric feedback (the saber swings on wrong votes too, into the wrong side). That is a Unity change (`AutoSaber.cs` `break` on wrong) for a future session — proposed, not done here.
- Fixing the EEG time-0 offset (take `Record datetime` at the first sample, or log `local_clock()` vs `time.time()`) — a `CygnusEEGReader` change, also only proposed.
