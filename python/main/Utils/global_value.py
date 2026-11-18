"""
會變動的變數存放位置，global
"""
import numpy as np
import main.Utils.config as config
import threading

unity_marker_string_stage = config.GameSTATE.LOBBY.value  # unity 傳送 marker 到這個變數，儲存 config GameSTATE 對應的 str，一開始再 Lobby
unity_marker_string_log = ""  # unity 傳送 marker 到這個變數，儲存 Start End CUT 的時間點，在 Utils UnityMarkerReader 改，game state 使用
unity_marker_string_change_model = "20251101_13c_shollownet_g_train-epoch10"  # 一樣在 Utils UnityMarkerReader，預設為標準模型
unity_marker_string_calibration = ""  # 用在 Calibration State，偵測 unity 是否傳送 calibration 字串，收到開始 online training，並存 pt
subject = ""  # main_start.py --subject，例如 "01"；有值時 run 資料夾放在 real_time_data/sub{subject}/ses{session}/
session = ""  # main_start.py --session，例如 "1"
runCount = 0  #追蹤目前 run 的編號，每次進入 Lobby 時 +1，用於 real_time_data/run{N}/ 資料夾
# ----------------------------
# Global buffer & concurrency
# ----------------------------
eeg_buffer = np.empty((config.N_CHANNELS, 500), dtype=float)  # 用在 CygnusEEGReader 和 EEGPrediction
data = np.empty((config.N_CHANNELS, 1), dtype=float)  # 用在 CygnusEEGReader 和 EEGPrediction
ts = 0  # 用在 CygnusEEGReader 和 EEGPrediction timestamp
buffer_lock = threading.Lock()  # 使用 lock 保護共享資源 eeg_buffer

data_lookup_table = {  # 暫時不使用
    config.GameSTATE.LOBBY.value: [],  # save tuple in list, (csv, log)
    config.GameSTATE.MI.value: [],
    config.GameSTATE.Calibration.value: [],  # use in EEG_Calibration.py
    config.GameSTATE.BeatSaber.value: [],
}

train_np_data = []  # 取代 data_lookup_table
# train_np_data = [
#     "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\real_time_data\\run6\\data_20260629_204218.pt",
#     "D:\\CECNL_lab\\lab_project\\VR\\VR-BCI_beat_saber_python\\main\\real_time_data\\run6\\data_20260629_204245.pt",]

# model_trained.pth 會用在最開始的模型，model.pth 用於當作 base model ft 的模型
models_name = []  # 顯示在 unity 的模型 # 會讀取 config.EEG_CHECKPOINT_MAIN_BASE_FILE 底下所有模型名稱，然後在切換場景重新載入 (game_state)
update_model = False  # 在 unity 傳輸字串後，會更新這個 (UnityMarkerReader.py)，然後在 game_state.py 使用，使用過後還原成 False (update model)
unity_update_model_str = None  # 跟上面的那個為一組的，都從 unity 那邊接收，接收後，根據模型名稱替換模型
# 訓練 (EEG_train FT train) 使用的位置，訓練完成後，會更新為訓練模型，在 lobby 選擇新模型也會 update
import os
NOW_TRAINED_CHECKPOINT = os.path.join(config.EEG_CHECKPOINT_MAIN_BASE_FILE, "cA_000.pth")  # 現在線上使用 (live) 的模型

# ---- 雙模型 A/B：各自微調，只有 live 的那個把預測送給 Unity ----
from collections import deque
A_CHECKPOINT = NOW_TRAINED_CHECKPOINT  # A：預訓練模型（例如 22 通道）
B_CHECKPOINT = os.path.join(config.EEG_CHECKPOINT_MAIN_BASE_FILE, "cB_000.pth")  # B：隨機初始化，不存在時自動產生
LIVE_MODE = "A"  # --live_model：A / B / auto
is_calibration_run = False  # 本 run 的歌曲是 Calibration (收到 Calibration stage 訊息時為 True；之後 Unity 會立刻送 MI，不能靠 stage 字串判斷)
run_mode = ""  # Unity START 行的 MODE: Adaptive / Static，由 UnityMarkerReader 更新，寫入 ab_blocks.csv
live_model = "A"  #目前實際 live 的模型（auto 時會依表現切換）
last_block_trials = None  # saveDataAsPtFromPointer 成功存檔時放入該 block 的 trial 時間與標籤；run_calibration 用完即清空
prediction_logs = {"A": deque(maxlen=60000), "B": deque(maxlen=60000)}  # (time.time(), pred, prob_class1)
prediction_log_lock = threading.Lock()

# ---- Unity 式計分與模型切換時機 (UnityMarkerReader / trial_scoring / game_state) ----
broadcast_model = ""  # 實際把預測送給 Unity 的模型 (predictor.live 為 True 的那個)；只在 game.reload_predictors 套用 live 旗標時改變
trial_live = {}  # {trial idx: 該 trial 投票當下的 broadcast_model}，每個 run 重設
trial_start_ts = {}  # {trial idx: START 時間}，用來量 START -> 投票的秒數
measured_vote_offset = None  # 本 run 實測的 START -> 投票秒數；None 時用 config.VOTE_OFFSET_S
vote_on_at = None  # 投票旗標：Unity 時間 >= 這個值就代表「Unity 快要投票了」；CUT / WRONG_CUT / END 到達後設回 None
last_swap_time = None  # 最近一次載入新權重 (reload_predictors) 的時間
run_start_time = 0.0  # 本 run 開始的 time.time()，predictions.csv 只存這之後的預測


def reset_run_state():
    """進入新 run (Lobby) 時重設與單一 run 相關的狀態"""
    global is_calibration_run, run_mode, trial_live, trial_start_ts, measured_vote_offset, vote_on_at, run_start_time, unity_marker_string_calibration
    import time
    is_calibration_run = False
    run_mode = ""
    trial_live = {}
    trial_start_ts = {}
    measured_vote_offset = None
    vote_on_at = None
    unity_marker_string_calibration = ""
    run_start_time = time.time()