"""
Per-trial scoring of the A/B predictors from Unity's real result.
  - unity_vote:  replica of Unity's AutoSaber vote (last 100 predictions, weights 1 -> 5, oldest -> newest)
  - score_block: vote time = CUT / WRONG_CUT timestamp (fallback END - PREDICTION_LOG_OFFSET_S).
                 The broadcasting model is scored by Unity's result (CUT correct, WRONG_CUT / no CUT wrong),
                 the other model by unity_vote_at() at that same timestamp.
  - vote flag:   "Unity is about to vote" window, used to delay swapping in new weights
  - dump_predictions: all predictions of the current run -> run{N}/predictions.csv
"""
import csv
import os
import time
import main.Utils.config as config
import main.Utils.global_value as global_value

HISTORY_LENGTH = 100  # Unity Config.PredictionHistoryLength
VOTE_WAIT_TIMEOUT_S = 3.0  # wait_vote_flag_off 最久等多久


def snapshot_logs():
    with global_value.prediction_log_lock:
        return {name: list(log) for name, log in global_value.prediction_logs.items()}


def unity_vote(preds):
    """preds: 0/1 predictions, oldest first. Returns 0/1, or None if fewer than 2 predictions."""
    preds = preds[-HISTORY_LENGTH:]
    n = len(preds)
    if n < 2:
        return None
    total, count1 = 0.0, 0
    for i, p in enumerate(preds):
        weight = 1.0 + 4.0 * i / (n - 1)
        total += weight
        if p == 1:
            count1 += round(weight)  # C# Mathf.RoundToInt also rounds half to even
    return 1 if count1 >= total / 2.0 else 0


def window_vote(entries):
    """entries: [(t, pred, prob1)]. Majority vote; tie -> class with higher mean probability. None if empty.
    Not used online any more (whole-trial vote); kept for recomputing from predictions.csv."""
    if not entries:
        return None
    n1 = sum(1 for _, p, _ in entries if p == 1)
    n0 = len(entries) - n1
    if n1 != n0:
        return 1 if n1 > n0 else 0
    mean1 = sum(q for _, _, q in entries) / len(entries)
    return 1 if mean1 > 0.5 else 0


def unity_vote_at(log, t):
    """Unity-style vote from the model's predictions made up to time t."""
    return unity_vote([p for ts, p, _ in log if ts <= t])


def vote_info(t):
    """trial dict from the log -> (vote_ts, unity_correct). vote_ts falls back to END - offset; None if no END either.
    unity_correct: True with a CUT, False with a WRONG_CUT or no CUT (the broadcasting model's exact result)."""
    if t.get('cut'):
        return t['cut'][-1], True
    if 'wrong_cut' in t:
        return t['wrong_cut'], False
    if 'end' in t:
        return t['end'] - config.PREDICTION_LOG_OFFSET_S, False
    return None, False


def model_prediction_suffix(end_ts, logs=None, vote_ts=None, live=None):
    """Text appended to a trial's END log line. AT = the trial's vote time when known, else END - offset."""
    logs = logs if logs is not None else snapshot_logs()
    at = vote_ts if vote_ts is not None else end_ts - config.PREDICTION_LOG_OFFSET_S
    live = live or global_value.broadcast_model or global_value.live_model
    vals = {}
    for name in ("A", "B"):
        v = unity_vote_at(logs[name], at)
        vals[name] = "NA" if v is None else v
    return (f"MODEL_PREDICTION: A={vals['A']} B={vals['B']} "
            f"LIVE={live} AT: {at:.3f}")


def score_block(trials, logs=None, trial_live=None):
    """
    trials: {idx: {'end','label','cut'?,'wrong_cut'?}} from the log; trial_live: {idx: broadcasting model at the vote}.
    Returns (scores, per_trial): scores = {name: (n_correct, n_scored)};
    per_trial = {idx: {'vote_ts', 'live', 'A': bool|None, 'B': bool|None}} (None = could not be scored).
    """
    logs = logs if logs is not None else snapshot_logs()
    trial_live = trial_live if trial_live is not None else global_value.trial_live
    default_live = global_value.broadcast_model or global_value.live_model
    per_trial = {}
    for idx in sorted(trials):
        t = trials[idx]
        if 'end' not in t or 'label' not in t:
            continue
        vote_ts, unity_correct = vote_info(t)
        live = trial_live.get(idx, default_live)
        res = {'vote_ts': vote_ts, 'live': live}
        for name in ("A", "B"):
            if name == live:
                res[name] = unity_correct
            else:
                v = unity_vote_at(logs[name], vote_ts)
                res[name] = None if v is None else (v == t['label'])
        per_trial[idx] = res
    scores = {}
    for name in ("A", "B"):
        results = [r[name] for r in per_trial.values() if r[name] is not None]
        scores[name] = (sum(results), len(results))
    return scores, per_trial


# ---- vote flag：Unity 快要投票的時間窗，這段時間不載入新權重 ----
def current_vote_offset():
    m = global_value.measured_vote_offset
    return m if m is not None else config.VOTE_OFFSET_S


def on_trial_start(idx, ts):
    global_value.trial_start_ts[idx] = ts
    global_value.vote_on_at = ts + current_vote_offset() - config.SWAP_GUARD_S


def on_trial_vote(idx, ts):
    """CUT / WRONG_CUT 到達：旗標關閉，並用 START -> 投票的實測秒數更新 vote offset"""
    start = global_value.trial_start_ts.get(idx)
    if start is not None and ts > start:
        global_value.measured_vote_offset = ts - start
    global_value.vote_on_at = None
    global_value.trial_live[idx] = global_value.broadcast_model or global_value.live_model


def on_trial_end(idx):
    global_value.vote_on_at = None  # 保險：沒有投票行 (舊版 Unity) 時也要關
    global_value.trial_live.setdefault(idx, global_value.broadcast_model or global_value.live_model)


def vote_flag_on(now=None):
    on_at = global_value.vote_on_at
    return on_at is not None and (time.time() if now is None else now) >= on_at


def wait_vote_flag_off(timeout=VOTE_WAIT_TIMEOUT_S, poll=0.02):
    """等到 vote flag 關閉才回傳 True；逾時回傳 False (呼叫端仍會繼續換模型)"""
    deadline = time.time() + timeout
    while vote_flag_on():
        if time.time() >= deadline:
            return False
        time.sleep(poll)
    return True


def dump_predictions(path=None, since=None):
    """把本 run 的所有預測 (A、B) 寫成 run{N}/predictions.csv：model, time, pred, prob1"""
    path = path or os.path.join(config.getRunDataDir(), "predictions.csv")
    since = global_value.run_start_time if since is None else since
    logs = snapshot_logs()
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "time", "pred", "prob1"])
        for name in ("A", "B"):
            for t, pred, prob in logs[name]:
                if t >= since:
                    w.writerow([name, f"{t:.3f}", pred, f"{prob:.4f}"])
    return path
