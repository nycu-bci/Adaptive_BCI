"""
ERD + Riemannian online pipeline (full flow) + CSV logging + Marker stream
Author: ChatGPT
Date: 2025-09-13
Notes:
 - Requires pylsl, numpy, scipy, sklearn, pyriemann, mne (optional for ICA)
 - Set N_CHANNELS to actual number of channels your LSL stream provides
"""
import csv
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet, resolve_streams
import sys
from datetime import datetime
from functools import wraps
import main.Utils.config as config
import main.Utils.global_value as global_value
import main.Utils.LSL as LSL
import re
from main.Utils.some_functions import rename_file_with_time
from main.Utils.TCPServer import TCPServer
from main.Utils import trial_scoring
import time


class UnityTCPReader:
    def __init__(self):
        self.save_csv = config.SAVE_CSV
        self.filename = config.LOG_FILENAME
        self.csv_file = None  # CSV 檔案物件
        self.tcp_server = None  # TCPServer(host, port, on_message=self.process_message)
        self.vote_ts = {}  # {trial idx: CUT / WRONG_CUT 時間}，END 行用來算 AT

    def setup_tcp_server(self, tcp_server):
        self.tcp_server = tcp_server

    def start_write_log(self):
        self.filename = rename_file_with_time(config.getRunLogFilename())  # config.LOG_FILENAME
        self.vote_ts.clear()
        if self.save_csv:
            try:
                self.csv_file = open(self.filename, "w", encoding="utf-8", newline="")
            except Exception as e:
                print(f"[CSV] Failed to open file: {e}")

    def stop_and_save_log(self):
        # self.tcp_server.stop()
        self.close_csv_file()

    def flushLog(self):
        """強制將 LOG 檔案 buffer 寫入磁碟，供 pointer-based 讀取前呼叫"""
        if self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.flush()
                print(f"{config.TAGS.INFO.value} [DEBUG] flushLog: LOG buffer 已寫入磁碟")
            except Exception as e:
                print(f"{config.TAGS.WARNING.value} flushLog failed: {e}")

    def close_csv_file(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                print("[CSV] Saved EEG data to", self.filename)
            except Exception:
                print("[CSV] Saved fail")
            finally:
                self.csv_file = None

    def process_message(self, msg: str, tcp_server: TCPServer):  # 在 game_state 裡面加入到 TCPServer 的 on message
        # 處理 stage
        for state in config.GameSTATE:
            if state.value == msg:
                global_value.unity_marker_string_stage = msg
                if msg == config.GameSTATE.Calibration.value:
                    global_value.is_calibration_run = True
                print(f"{config.TAGS.INFO.value} set unity_marker_string_stage")
        if msg == config.RECEIVE_UNITY_MODEL_STR:  # send exist model
            name = config.SENT_UNITY_MODEL_STR
            for i in global_value.models_name:
                name += config.SEPARATE_STR
                name += i
            tcp_server.broadcast(name)
            print(f"{config.TAGS.INFO.value} SENT_UNITY_MODEL_STR {name}")
        text = msg.split("@@@")
        if text[0] == config.RECEIVE_UNITY_SELECT_MODEL_STR:  # send_python_tcp_select_model_str@@@model_name # 選擇使用的模型
            if len(text) == 2:
                global_value.update_model = True
                global_value.unity_update_model_str = f"{config.EEG_CHECKPOINT_MAIN_BASE_FILE}{text[1]}"
                print(f"{config.TAGS.INFO.value} Model selected {global_value.unity_update_model_str}")

        # 處裡 Calibration 字串
        # 已知 MODE 時由 Python 在每 N 個 trial 的 END 自己觸發 (見下面)，忽略 Unity 的 calibration_start；舊版 Unity (沒有 MODE) 照舊
        if msg == config.RECEIVE_UNITY_CALIBRATION_START_STR and not global_value.run_mode:
            global_value.unity_marker_string_calibration = config.RECEIVE_UNITY_CALIBRATION_START_STR

        # 處理 log
        pat = re.compile(r'Trial\s+(\d+)\s+(START|CUT|WRONG_CUT|END):\s*([\d\.]+)(?:\s+LABEL:\s*(\d+))?', re.IGNORECASE)
        pat_eyes = re.compile(r'(Close eyes|Open eyes):\s*([\d\.]+)', re.IGNORECASE)
        m = pat.match(msg)

        if not m:  # 沒有比對到 MI，判斷 eyes 邏輯
            m = pat_eyes.match(msg)
        if m:
            global_value.unity_marker_string_log = msg

            print(f"{config.TAGS.INFO.value} set unity_marker_string_log")
            if config.verbose:
                try:
                    if m.group(3):
                        tms = (float(m.group(3)) - time.time()) * 1000  # ms = 1000s
                        print(f"{config.TAGS.INFO.value} [TCP Latency] {tms} ms")
                except IndexError:
                    pass

            typ, idx, ts = None, None, None
            if m.re is pat:
                typ, idx, ts = m.group(2).upper(), int(m.group(1)), float(m.group(3))
            if typ == "START":
                mode = re.search(r'MODE:\s*(\w+)', msg)
                if mode:
                    global_value.run_mode = mode.group(1)
                trial_scoring.on_trial_start(idx, ts)
            elif typ in ("CUT", "WRONG_CUT"):
                trial_scoring.on_trial_vote(idx, ts)
                self.vote_ts[idx] = ts

            line = msg
            if typ == "END":  # 只在 END 行附加 A/B 模型在投票時間 (CUT / WRONG_CUT，沒有就 END-0.70s) 的 Unity 式投票結果
                trial_scoring.on_trial_end(idx)
                try:
                    line = f"{msg} {trial_scoring.model_prediction_suffix(ts, vote_ts=self.vote_ts.pop(idx, None), live=global_value.trial_live.get(idx))}"
                except Exception as e:
                    print(f"{config.TAGS.WARNING.value} MODEL_PREDICTION failed: {e}")

            if self.save_csv and self.csv_file:
                try:
                    self.csv_file.write(f"{line}\n")
                    self.csv_file.flush()  # 即時寫入
                except Exception as e:
                    print(f"[CSV] Write failed: {e}")

            # END 行已寫入 log 之後才觸發，main loop 讀到 flag 時 log 一定包含這個 trial
            if typ == "END" and self.should_train_after(idx):
                global_value.unity_marker_string_calibration = config.RECEIVE_UNITY_CALIBRATION_START_STR

    @staticmethod
    def should_train_after(idx):
        """已知 MODE 時：每 trial_train_interval 個 trial (trial 編號從 0 起算) 檢查一次，只在 Adaptive 或 Calibration 歌曲 (Unity 送 Calibration 之後立刻送 MI，所以用 is_calibration_run 旗標而不是 stage 字串)"""
        if not global_value.run_mode:
            return False
        interval = 1 if config.FINETUNE_GATE == "sprt" else config.trial_train_interval  # SPRT：每個 trial 都計分
        if (idx + 1) % interval != 0:
            return False
        return global_value.run_mode == "Adaptive" or global_value.is_calibration_run


def send_marker(label: int, marker_outlet):
    """Send integer marker through LSL, update latest_marker, and log to CSV."""
    try:
        marker_outlet.push_sample([int(label)])
    except Exception as e:
        print(f"{config.TAGS.MARKER.value} Push failed:", e)

    print(f"{config.TAGS.MARKER.value} Sent: {label}")

# def main_flow():
#     unity_reader = UnityLSLReader()
#     unity_reader.start_read_marker_thread()
#     while True:
#         pass
#
#
# if __name__ == "__main__":
#     try:
#         main_flow()
#     except KeyboardInterrupt:
#         print("Interrupted by user. Exiting.")
