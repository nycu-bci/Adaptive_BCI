#!/bin/bash
# 用法：run_worker_as.sh <假的主機名> <log> <simulate_gates.py 的參數...>
# 讓這台機器的 worker 以另一台主機的名義領工作 (session 綁定用 socket.gethostname)，不改任何既有檔案。
AS=$1; LOG=$2; shift 2
cd ~/gate_study/Adaptive_BCI/python
python - "$AS" "$@" > "$LOG" 2>&1 <<'PY'
import runpy, socket, sys
as_host = sys.argv[1]
socket.gethostname = lambda: as_host
sys.argv = ["analysis/gate_study/simulate_gates_r4.py"] + sys.argv[2:]
print(f"acting as {as_host} for session binding", flush=True)
runpy.run_path("analysis/gate_study/simulate_gates_r4.py", run_name="__main__")
PY
echo "EXIT $?" >> "$LOG"
