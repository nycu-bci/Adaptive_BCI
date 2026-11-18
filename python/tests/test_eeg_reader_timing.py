"""
Tests that the Cygnus EEG CSV is time-aligned: Record datetime == Unix time of the first sample (Timestamp 0), so
Record datetime + row Timestamp == the Unix time of every row (known issue H4: it used to be ~1 s earlier).
The end-to-end test uses a real local LSL stream (SKIP if LSL cannot resolve it, e.g. a firewall blocks it).

Usage: C:\\Users\\aiden\\anaconda3\\envs\\VRBCIenv\\python.exe python/tests/test_eeg_reader_timing.py   (VRBCIenv has no pytest; pytest also works)
"""
import contextlib
import io
import os
import re
import sys
import tempfile
import threading
import time
import uuid

PYTHON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PYTHON_DIR)
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_byprop

import main.Utils.config as config
import main.Utils.global_value as global_value
from main.EEG.CygnusEEGReader import EEGReader
from main.EEG.data_process_np import EEGDataLoader
from main.Utils.file_pointer_reader import extractRecordEpoch, readCsvFromPointer

FS = 500
CHUNK = 50  # samples per push -> one push every 100 ms
N_CH = 32
RECORD_RE = re.compile(r"^Record datetime: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}$")


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


@contextlib.contextmanager
def reader_env(stream_name=None):
    """Save and restore every config / global_value attribute a reader run touches."""
    saved = (config.RECEIVE_CYGNUS_LSL_STREAM, config.SAVE_CSV, global_value.eeg_buffer)
    if stream_name is not None:
        config.RECEIVE_CYGNUS_LSL_STREAM = stream_name
    config.SAVE_CSV = True
    global_value.eeg_buffer = np.empty((config.N_CHANNELS, 500), dtype=float)
    try:
        yield
    finally:
        config.RECEIVE_CYGNUS_LSL_STREAM, config.SAVE_CSV, global_value.eeg_buffer = saved


class FakeCygnus:
    """Local LSL outlet pushing 50-sample chunks every 100 ms stamped as 'recorded' 100 ms before the push."""

    def __init__(self, push=True):
        self.name = "TestCygnus-" + uuid.uuid4().hex[:8]
        info = StreamInfo(self.name, "EEG", N_CH, FS, "float32", self.name)
        self.outlet = StreamOutlet(info)
        self.push = push
        self.stamps = []  # stamps[i] = local-clock time of global sample i
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        idx, k, t0 = 0, 0, time.perf_counter()
        while not self.stop.is_set():
            if self.push:
                now = local_clock()
                st = now - CHUNK / FS + np.arange(CHUNK) / FS
                x = np.zeros((CHUNK, N_CH), dtype=np.float32)
                x[:, 0] = np.arange(idx, idx + CHUNK)
                x[:, 1:] = np.random.default_rng(idx).normal(size=(CHUNK, N_CH - 1))
                self.outlet.push_chunk(x, st.tolist())
                self.stamps.extend(st.tolist())
                idx += CHUNK
            k += 1
            time.sleep(max(0.0, t0 + k * CHUNK / FS - time.perf_counter()))

    def __enter__(self):
        self.thread.start()
        if not resolve_byprop("name", self.name, timeout=5.0):
            self.close()
            skip("local LSL stream could not be resolved within 5 s (firewall / no LSL multicast?)")
        return self

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        del self.outlet

    def __exit__(self, *exc):
        self.close()


def run_reader(reader, seconds):
    """Run reader.read_eeg in a thread for `seconds`, return its stdout."""
    reader.read_eeg_stop_event.clear()
    t = threading.Thread(target=reader.read_eeg, daemon=True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        t.start()
        time.sleep(seconds)
        reader.read_eeg_stop_event.set()
        t.join(timeout=15)
    assert not t.is_alive(), "read_eeg did not stop"
    return out.getvalue()


def test_csv_timestamps_match_unix_time_of_each_sample():
    with FakeCygnus() as src, tempfile.TemporaryDirectory() as tmp, reader_env(src.name):
        reader = EEGReader()
        reader.is_simulated = False
        reader.save_csv = True
        reader.filename = os.path.join(tmp, "eeg.csv")
        log = run_reader(reader, 3.5)
        assert not reader.is_simulated, "reader fell back to simulated mode:\n" + log
        assert os.path.exists(reader.filename)
        offset = time.time() - local_clock()  # local LSL clock -> Unix time
        rec = EEGDataLoader([], [], [2])._extract_record_epoch(reader.filename)
        assert rec == extractRecordEpoch(reader.filename)
        ts, eeg, ptr = readCsvFromPointer(reader.filename, [2], 0)
        assert len(ts) == ptr >= 500, len(ts)
        idx = eeg[:, 0].astype(np.int64)  # channel 0 carries the global sample index
        assert (np.diff(idx) == 1).all(), "duplicated or skipped rows"
        assert ts[0] == 0.0
        expected = np.array(src.stamps)[idx] + offset
        err_ms = np.abs(rec + ts - expected) * 1000
        m = re.search(r"arrival delay=([\d.\-]+) ms", log)
        print(f"      rows={len(ts)}  max|err|={err_ms.max():.3f} ms  mean|err|={err_ms.mean():.3f} ms  "
              f"arrival delay={m.group(1) if m else '?'} ms  record-vs-first-sample={abs(rec - expected[0]) * 1000:.3f} ms")
        assert err_ms.max() <= 2.0, err_ms.max()


def test_record_datetime_round_trip():
    base = float(int(time.time()))
    with tempfile.TemporaryDirectory() as tmp, reader_env():
        for i, rec_ts in enumerate([base, base + 0.999999, base + 0.5, base + 0.000001, base + 0.123456, time.time()]):
            reader = EEGReader()
            reader.filename = os.path.join(tmp, f"h{i}.csv")
            reader.build_csv_file(rec_ts)
            reader.close_csv_file()
            with open(reader.filename) as f:
                lines = f.read().splitlines()
            assert len(lines) == 11 and "Simulating Cygnus" in lines[0] and lines[10].startswith("Timestamp,Serial Number,Fp1")
            assert RECORD_RE.match(lines[2]), lines[2]
            a = EEGDataLoader([], [], [2])._extract_record_epoch(reader.filename)
            b = extractRecordEpoch(reader.filename)
            assert abs(a - rec_ts) < 1e-6 and abs(b - rec_ts) < 1e-6, (rec_ts, a, b)


def test_read_csv_from_pointer_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        ts, eeg, ptr = readCsvFromPointer(os.path.join(tmp, "nope.csv"), [2, 3, 4], 7)
        assert ts.size == 0 and eeg.shape == (0, 3) and ptr == 7


def test_simulated_mode_still_writes_file():
    with tempfile.TemporaryDirectory() as tmp, reader_env():
        reader = EEGReader()
        reader.is_simulated = True
        reader.save_csv = True
        reader.filename = os.path.join(tmp, "sim.csv")
        t0 = time.time()
        run_reader(reader, 0.5)
        rec = extractRecordEpoch(reader.filename)
        assert abs(rec - t0) < 1.0, rec - t0
        ts, eeg, ptr = readCsvFromPointer(reader.filename, [2], 0)
        assert len(ts) >= 5 and ts[0] == 0.0 and (np.diff(ts) >= 0).all()


def test_stop_before_first_sample_creates_no_file():
    with FakeCygnus(push=False) as src, tempfile.TemporaryDirectory() as tmp, reader_env(src.name):
        reader = EEGReader()
        reader.is_simulated = False
        reader.save_csv = True
        reader.filename = os.path.join(tmp, "empty.csv")
        log = run_reader(reader, 2.5)
        assert not reader.is_simulated, log
        assert not os.path.exists(reader.filename)
        reader.flushCsv()
        ts, eeg, ptr = readCsvFromPointer(reader.filename, [2], 0)
        assert ts.size == 0 and ptr == 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # reader prints Chinese; a cp1252 console would raise
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.1f}s)")
        except Skip as e:
            print(f"SKIP {name}: {e}")
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
