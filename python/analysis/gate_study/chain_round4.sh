#!/bin/bash
# 在一台主機上：等這台的第三輪 worker (tmux g3_0..3) 都結束、GPU 上沒有其他程序，就開 4 個第四輪 worker (延遲 0)。
# 第四輪的 session -> 主機綁定事先從第二輪複製，所以同一個 session 跟第二輪在同一台 GPU 上跑，可以直接比較。
OUT=~/gate_study/results/round4_shadow_20261123
H=$(hostname)
sleep 120  # 讓第三輪先啟動
while tmux ls 2>/dev/null | grep -q '^g[23]_[0-3]'; do sleep 60; done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "$(date) GPU busy (not ours?), round 4 not started on $H" >> ~/gate_study/results/chain_round4_$H.log; exit 1
fi
for i in 0 1 2 3; do
  tmux new-session -d -s g4_$i "bash -ic 'conda activate chinghao_tseng && cd ~/gate_study/Adaptive_BCI/python && python analysis/gate_study/simulate_gates.py --start_ckpts ~/gate_study/results/pretrain_4fold_20261123/starts.json --out $OUT --queue $OUT/queue.txt > $OUT/worker_${H}_$i.log 2>&1; echo EXIT \$? >> $OUT/worker_${H}_$i.log'"
  sleep 2
done
echo "$(date) round 4 started on $H" >> ~/gate_study/results/chain_round4_$H.log
