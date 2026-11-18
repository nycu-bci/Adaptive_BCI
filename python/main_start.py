# python python/main_start.py --subject 01 --session 1 --live_model auto --start_from session   (live_model: A | B | auto, default A; start_from: session | base | subject, default session)
import os
import sys
import argparse
from functools import wraps
import main.Utils.config as config
import main.Utils.global_value as global_value
from main.Utils.some_functions import resolve_start_checkpoint
from main.game_state import Game


# 自動儲存 log
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def tee_log(log_file=None):
    """裝飾器：將 print 輸出同時存檔與顯示；預設存到 sub{S}/ses{N}/session_log_<時間>.txt (呼叫時才決定路徑，需在 setup_args 之後)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            path = log_file or config.getSessionLogPath()
            with open(path, "w", encoding="utf-8") as f:
                sys.stdout = Tee(original_stdout, f)
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout
            print(f"✅ 輸出已保存到 {path}")
            return result

        return wrapper

    return decorator


# ----------------------------
# Main flow
# ----------------------------
@tee_log()
def main_flow():
    # marker_outlet = LSL.setup_lsl_outlet(config.TO_UNITY_LSL_STREAM)  # to unity
    game = Game()
    game.start()


def setup_args():
    """解析命令列參數並寫入 global_value (live 模型、受試者、session、接續的 run 編號、A/B 起點 checkpoint)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--live_model", choices=["A", "B", "auto"], default="A",
                        help="哪個模型把預測送給 Unity；auto = 另一個模型連續 2 個 block trial accuracy 嚴格較高才切換")
    parser.add_argument("--start_model", choices=["A", "B"], default="A",
                        help="--live_model auto 時最初由哪個模型控制遊戲 (A / B 固定時忽略)")
    parser.add_argument("--start_from", choices=["session", "base", "subject"], default="session",
                        help="A、B 各自從哪個 checkpoint 開始：session = 這個 session 最新的，沒有就用 base；"
                             "base = 永遠用 checkpoint_main 的 cA_000 / cB_000；subject = 這個受試者編號最大、有 checkpoint 的 session 的最新一個 "
                             "(A、B 各自找，所以可能來自不同 session，例如 A 來自 ses2、B 來自 ses1；啟動時會印出各自來源，不同時警告)")
    parser.add_argument("--subject", required=True, help="受試者代號，例如 01、02")
    parser.add_argument("--session", required=True, help="session 代號，例如 1、2")
    args = parser.parse_args()
    if not (args.subject.isdigit() and args.session.isdigit()):
        parser.error("--subject 與 --session 必須是數字，例如 --subject 01 --session 1")
    global_value.subject, global_value.session = args.subject, args.session
    session_dir = config.getSessionDir()
    os.makedirs(session_dir, exist_ok=True)
    # 同一個 session 重開 Python 時接續 run 編號，避免新資料混進舊的 run 資料夾
    global_value.runCount = max([int(d[3:]) for d in os.listdir(session_dir) if d.startswith("run") and d[3:].isdigit()], default=0)
    print(f"{config.TAGS.INFO.value} Subject {args.subject} Session {args.session} -> {session_dir} (next run: {global_value.runCount + 1})")
    global_value.LIVE_MODE = args.live_model
    global_value.live_model = args.start_model if args.live_model == "auto" else args.live_model
    # A、B 分開決定起點 (例如 cA_009 和 cB_012)，與 --live_model / --start_model 無關
    global_value.A_CHECKPOINT = resolve_start_checkpoint("cA_", args.start_from)
    global_value.B_CHECKPOINT = resolve_start_checkpoint("cB_", args.start_from)
    global_value.NOW_TRAINED_CHECKPOINT = getattr(global_value, f"{global_value.live_model}_CHECKPOINT")
    print(f"{config.TAGS.INFO.value} start_from={args.start_from}: A={global_value.A_CHECKPOINT} B={global_value.B_CHECKPOINT}")
    folder_a, folder_b = os.path.dirname(global_value.A_CHECKPOINT), os.path.dirname(global_value.B_CHECKPOINT)
    for name, path in (("A", global_value.A_CHECKPOINT), ("B", global_value.B_CHECKPOINT)):
        print(f"{config.TAGS.INFO.value} 起點 {name}: {os.path.basename(path)} 來自 {os.path.dirname(path)}")
    if folder_a != folder_b:
        print(f"{config.TAGS.WARNING.value} A 與 B 的起點來自不同資料夾 (A: {folder_a} / B: {folder_b})；各自取最新的 checkpoint，請確認這是預期的")


if __name__ == "__main__":
    setup_args()
    try:
        main_flow()
    except KeyboardInterrupt:
        print(f"{config.TAGS.WARNING.value} Interrupted by user. Exiting.")
