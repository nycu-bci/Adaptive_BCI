import main.Utils.config as config
import csv
import threading
import numpy as np
from datetime import datetime
import os
import re


def rename_file_with_time(name):
    # 格式化時間為 'YYYYMMDD_HHMMSS'
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 取得檔案名稱
    file_name, file_extension = os.path.splitext(name)
    # 生成新的檔案名稱
    return f"{file_name}_{formatted_time}{file_extension}"


def highest_version(directory, prefix, ext=".pth"):
    """directory 底下 {prefix}NNN{ext} 的最大編號，沒有則回傳 None"""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = [int(m.group(1)) for f in (os.listdir(directory) if os.path.isdir(directory) else [])
            for m in [pat.match(f)] if m]
    return max(nums) if nums else None


def get_next_version_path(base_path, prefix="c_"):
    """
    回傳 base_path 所在資料夾裡「最大編號 + 1」的路徑 (例如已有 cA_001、cA_004 -> cA_005.pth)，
    資料夾沒有任何 {prefix}NNN 時從 001 開始。這樣最大的編號永遠是最新的，刪掉中間的檔案也不會被補回。
    """
    directory, filename = os.path.split(base_path)
    ext = os.path.splitext(filename)[1]
    top = highest_version(directory, prefix, ext)
    return os.path.join(directory, f"{prefix}{(top or 0) + 1:03d}{ext}")


def resolve_start_checkpoint(prefix, start_from="session"):
    """
    決定 A / B 各自的起點 checkpoint (prefix 為 "cA_" 或 "cB_")。
      base    : checkpoint_main/{prefix}000.pth
      session : 這個 session 的 checkpoints/ 裡編號最大的，沒有就用 base
      subject : 這個受試者編號最大、且有 checkpoint 的 session 裡編號最大的，沒有就用 base
    """
    base = os.path.join(config.EEG_CHECKPOINT_MAIN_BASE_FILE, f"{prefix}000.pth")
    if start_from == "base":
        return base
    session_ckpt = os.path.join(config.getSessionDir(), "checkpoints")
    candidates = [session_ckpt]
    if start_from == "subject":
        subject_dir = os.path.dirname(config.getSessionDir())
        sessions = [d for d in os.listdir(subject_dir) if d.startswith("ses") and d[3:].isdigit()] if os.path.isdir(subject_dir) else []
        candidates = [os.path.join(subject_dir, d, "checkpoints") for d in sorted(sessions, key=lambda d: -int(d[3:]))]
    for directory in candidates:
        top = highest_version(directory, prefix)
        if top is not None:
            return os.path.join(directory, f"{prefix}{top:03d}.pth")
    return base
