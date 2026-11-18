# Plan: does online finetuning help? (chain experiment + checkpoint overview)

Status: approved scope 2026-10-27 (all 5 sessions), not started. Replaces the 2026-10-19 draft. Comes before the H1 switching work (`docs/known_issues.md`), because switching compares two models that finetuning keeps changing.

## Why first
- In sub01 ses1, A finetuned in 31 of 85 blocks and B in 20; in runs 3, 4 and 6, A changed in 7, 6 and 8 of 12 blocks.
- A finetuned more and scored lower, but that proves nothing: the gate only fires after bad blocks, so finetuned blocks are bad by selection (regression to the mean). Comparing checkpoints on the same later trials removes that bias.
- First hint (cut comparison, 2026-10-27): on sub01, the untrained pretrained base `cA_000` and the online A chain both got 75.0% of the 384 votes right (agreeing on 288 trials).

## Questions
| ID | Question | Needs |
|---|---|---|
| F1 | Does one finetune make the model better on the trials after it? (chain experiment) | Saved checkpoints |
| OV | How does every checkpoint do on every trial of the session? (overview) | Saved checkpoints |
| F1b | Same as F1, with the finetunes retrained on correctly cut data (H4 fixed) | Offline retraining (step 0b) |
| F2 | Does the gate fire when a finetune would help, and stay off when it wouldn't? | Offline retraining (step 0b) |

This plan runs F1 and OV. F1b and F2 are decided after F1.

Not in this plan: whether gated finetuning beats finetuning every block or never. That is the system's research question, answered by the subject study. The overview shows chain vs frozen models only as a description of these sessions.

## Data (checked 2026-10-27)
| Session | Usable runs | Scored trials | Live model (trials) | A finetunes | B finetunes |
|---|---|---|---|---|---|
| sub01 ses1 | runs 1–7 (run 1 = calibration song + adaptive) | 384 | A 384 | 31 | 20 |
| sub02 ses1 | run 1 (calibration song) | 48 | A 38, B 10 (auto) | 12 | 11 |
| sub03 ses1 | runs 1–3 | 144 | A 120, B 24 (auto) | 28 | 20 |
| sub04 ses1 | runs 1–2 | 96 | A 53, B 43 (auto) | 17 | 17 |
| sub05 ses1 | run 1 (calibration song); run 2 Static, test trials only | 48 + 48 | A 96 | 0 | 10 |
| **Total** | | **720 + 48** | | **88** | **78** |

- Usable = the run has `predictions.csv` (needed for the lag and prediction times). Runs with `ab_blocks.csv` give finetunes; sub05 run 2 (Static, no finetuning, no `ab_blocks.csv`) only adds test trials: no swaps, checkpoints as at the end of run 1. Not usable: sub01 run 8, sub02 run 2, sub03 run 4, sub04 run 3, sub05 run 3 (no `predictions.csv`).

- Per session: `checkpoints/c?_NNN.pth`; bases `python/main/EEG/checkpoint_main/cA_000` (pretrained, 22 ch) and `cB_000` (random, 32 ch).
- Per run: `eeg_record_*.csv`, log (START / CUT / WRONG_CUT / END with `AT:` and `LIVE=`), `predictions.csv`, `ab_blocks.csv` (checkpoint per block, `swap_time`), `data_*.pt`. Only runs with `predictions.csv` and `ab_blocks.csv` are used.
- All five sessions were recorded before the H4 fix (2026-10-27): their EEG files are offset by ~1 s per run and every finetune trained on the late cut. F1 and OV therefore describe the system as it ran.

## Measure
**Unity's vote, recomputed offline**, for any checkpoint on any trial: predict the 1 s windows ending at the evaluated model's logged prediction times in the last 100 before the vote time `AT`, with the online preprocessing, the run's measured lag and 50-sample chunks (`python/analysis/cut_compare/common.py`, validated in step 0a), then Unity's weighted vote → right / wrong. Same EEG and times for every checkpoint compared on a trial.
Secondary: the vote margin (weighted share of predictions that are right).

## Step 0a: lag and reproduction (gate)
- Done for sub01 (2026-10-22): lags 0.93–1.05 s; predictions A 96.9%, B 96.8%; votes A 376/384, B 383/384 (A one trial short of the 98% gate; accepted for relative comparisons, since every checkpoint is scored by the same predictor).
- To do: the same lag search and reproduction for every usable run of sub02–sub05.
- Pass per session: ≥ 95% of predictions identical. Vote agreement is reported; a session clearly below sub01's level (e.g. < 95% of votes) stops the plan for that session until the cause is found.

## F1: chain experiment (one finetune at a time)
- For every finetune k of model M (166 in total): parent `c{M}_{k−1}` ("skip this update") vs child `c{M}_k` ("with it"), both predicting the **same trials after the update**, which neither trained on.
- Windows:
  - **W1 (primary):** the trials the child actually served, from its `swap_time` until M's next swap. Windows don't overlap, so they pool.
  - **W2:** the next 48 trials after the swap (same length for all; overlapping, so CIs resample runs).
- **Closed-loop split:** the subject saw the live model's votes and may adapt to them, which can favour whichever checkpoint was live. Each test trial is tagged by whether M was live on it (`LIVE=` in the log). Results are reported for "M not live" (clean) and "M live" separately, and pooled.
- Statistics, per model, per session and pooled:
  - exact sign test (McNemar) on trials where parent and child votes disagree;
  - accuracy change (child − parent) with a 95% CI (bootstrap over runs, then trials);
  - share of finetunes that made the model worse on W2;
  - effect vs the parent's accuracy just before the finetune.

## OV: overview, every checkpoint × every trial
- For each session and model: every checkpoint (`c?_000` … last) predicts every scored trial of the session (runs in order, trials 1–48 each).
- Saved per trial: `session, model, checkpoint, run, trial, label, vote_ok, margin, in_sample`, where `in_sample` = the trial happened before the checkpoint was created (it may be training data).
- **Figure 1, heatmap per session and model:** rows = checkpoints (oldest at top), columns = blocks of 4 trials in session order with run separators (the 48-trial calibration block is split into 4-trial columns), colour = vote accuracy of that checkpoint on those 4 trials. In-sample cells are shaded. The live chain (the checkpoint live at each block) is outlined. Side panel: each checkpoint's accuracy on later trials vs earlier trials.
- **Figure 2, chain view per session and model:**
  - Panel 1: rolling vote accuracy (24 trials) of the live chain and of frozen references `c?_000` and `c?_001` (`c?_001` from the trials after the calibration song only). Ticks mark finetunes.
  - Panel 2: chain − frozen with a confidence band.
  - Panel 3: one F1 dot per finetune (child − parent on W2) at its time.
- Notes: `cB_000` is a random network; its row is the chance reference. sub05 A has only `cA_000` (one row).
- Check: along the outlined live path, the recomputed votes must match the logged votes at the step 0a level.

## Later (decided after F1): F1b and F2
**Step 0b, retraining check:** rebuild each model's replay buffer block by block from the **original** `data_*.pt` (every block is added, trained or not; FIFO and failure flags per model as online), retrain each saved finetune from its actual parent with the online settings (5 seeds). Pass: the retrained child agrees with the saved child on later trials about as well as two seeds agree with each other. The seed spread is also the noise floor (step 0c).

**F1b, finetunes with the H4-correct cut:**
- Corrected copies of every trial, re-cut from the EEG file with the run's measured lag, saved under `python/analysis/results/finetune_eval/`. The original `data_*.pt` files are never changed.
- For each finetune: same parent, buffer rebuilt from corrected trials, same flags and settings, 5 seeds → child′.
- Compare on the same later trials: child′ vs parent (does a finetune help with correct data?) and child′ vs the saved child (what H4 cost per update).
- Each finetune is retrained on its own from the parent that actually existed; the chain is not rebuilt.

**F2, how good is the current gate?** (approved 2026-11-03; goal: the current gate's performance only. Comparing other gates, including an SPRT gate, is a separate experiment that uses F2's per-block table.)
- Gate under test (`EEG_Train._gates`): fire if the block has ≥ 1 wrong trial and the run accuracy including the block is < 0.75 (live model: Unity's result; other model: Python's replica vote; counters reset per run).
- Decision points: every block of every model, fired or not. sub01 85 blocks per model (A fired 31, B 20), sub05 run 1 12 blocks (A 0, B 10): 194 decisions; blocks with no later trials are dropped.
- Counterfactual per decision: parent = the checkpoint the model had at the end of the block (before any finetune); buffer = the model's replay buffer after adding the block, rebuilt from the H4-correct trials (F1b corrected copies, online FIFO / validation / failure flags); child = V0 finetune of the parent (online settings), 5 seeds (seed 0 replays the online generator, as F1b). Fired blocks are retrained too, so all decisions use one code path; their seeds equal F1b V0 (check: same votes on shared trials).
- Effect = child − parent vote accuracy on the 48 trials after the block's last trial (may span runs; fewer at the session end), 5 seeds averaged. Same window rule for fired and skipped blocks.
- Analysis:
  - A (primary): mean effect of fired vs skipped blocks and the difference, 95% CI (bootstrap over decisions; consecutive windows overlap, so CIs are too narrow), per model and pooled.
  - B: effect vs the gate's inputs (errors in the block, run accuracy at the decision) and vs the parent's accuracy on the previous 24 trials (binned means, Spearman, slope with CI).
  - C: 2 × 2 descriptive table, fired / skipped × help / hurt (seed-mean effect > 0 / < 0, and the stricter count with an exact sign test p < 0.05 per decision).
  - Live / not-live split as in F1. The sub01 calibration block (48 trials, both fired) is reported separately and pooled.
- Checks: the gate recomputed from `ab_blocks.csv` (correct_A/B, n_trials, run resets) reproduces `ft_A` / `ft_B` for every block; fired-block children match F1b V0.
- Limits: each decision judged with the history that actually happened (one step, not a policy); parents are the saved (late-cut) checkpoints; one test window has ~6 pp noise per decision, so group means carry the answer; 2 sessions.
- Outputs: `python/analysis/finetune_eval/f2_counterfactual.py`, `f2_analyze.py`, `f2_figures.py`; results in `python/analysis/results/finetune_eval/f2/`.

## Limits to state in the report
- **Few sessions, one each per subject; H4-era data.** Exploratory. Results describe the system as it ran with the ~1 s late training cut.
- **Closed loop.** Handled only by the live / not-live split; it can't show how the subject would have reacted to other feedback.
- **Short W1 windows.** Some finetunes served 2–4 trials; single finetunes say little, pooling carries the answer.
- **Offline predictor** reproduces ~97% of live predictions; the rest affects all compared checkpoints equally.

## Deliverables
- `python/analysis/finetune_eval/`: lag search for sub02–sub05 (reusing `cut_compare/common.py`, extended to load a run without `ab_blocks.csv` as test-only with no swaps), the trial × checkpoint prediction table, F1 analysis, figures. Each script has a run command at the top.
- Tests in `python/tests/`: finetune list from `ab_blocks.csv`, W1/W2 windows, in-sample flag, live tag, live-path check.
- Results in `python/analysis/results/finetune_eval/` (per-trial CSV, `summary.txt`, figures per session).
- Write-up in `docs/`: answer to F1, what the overview shows, and whether to run F1b / F2.

## Order and stop points
1. Step 0a for sub02–sub05. **Stop and report** if a session fails.
2. Trial × checkpoint table (feeds both F1 and OV), with the live-path check.
3. F1 analysis and the two figures.
4. Report; decide F1b / F2.
