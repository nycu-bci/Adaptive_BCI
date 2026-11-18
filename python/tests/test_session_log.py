"""
Tests that the overall stdout log (tee_log in main_start.py) lands in sub{S}/ses{N}/ as session_log_*.txt,
not in the cwd and not as a log_*.txt that the per-run analysis globs could match.

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_session_log.py   (VRBCIenv has no pytest; pytest also works)
"""
import glob
import os
import sys
import tempfile

PYTHON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PYTHON_DIR)
import main.Utils.config as config
import main.Utils.global_value as global_value
import main_start


def _with_session(tmp, fn):
    old = global_value.subject, global_value.session
    config.__dict__["__REALTIME_BASE_FILE"] = tmp
    global_value.subject, global_value.session = "05", "1"
    try:
        return fn()
    finally:
        global_value.subject, global_value.session = old


def test_log_saved_in_session_dir():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cwd:
        old_cwd, old_base = os.getcwd(), config.__dict__["__REALTIME_BASE_FILE"]
        os.chdir(cwd)

        @main_start.tee_log()
        def flow():
            print("hello session log")

        try:
            _with_session(tmp, flow)
        finally:
            os.chdir(old_cwd)
            config.__dict__["__REALTIME_BASE_FILE"] = old_base
        logs = glob.glob(os.path.join(tmp, "sub05", "ses1", "session_log_*.txt"))
        assert len(logs) == 1, logs
        assert "hello session log" in open(logs[0], encoding="utf-8").read()
        assert not glob.glob(os.path.join(tmp, "sub05", "ses1", "log_*.txt"))
        assert os.listdir(cwd) == []


def test_explicit_path_still_works():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "x.txt")

        @main_start.tee_log(target)
        def flow():
            print("explicit")

        flow()
        assert "explicit" in open(target, encoding="utf-8").read()


if __name__ == "__main__":
    test_log_saved_in_session_dir()
    test_explicit_path_still_works()
    print("2/2 passed")
