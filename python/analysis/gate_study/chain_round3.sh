#!/bin/bash
# 在一台主機上：等這台的第二輪 worker (tmux g2_0..3) 都結束，GPU 上沒有其他程序，就開 4 個第三輪 worker (--delay 2)
OUT=~/gate_study/results/round3_delay2_20261123
H=$(hostname)
while tmux ls 2>/dev/null | grep -q '^g2_'; do sleep 60; done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "$(date) GPU busy (not ours?), round 3 not started on $H" >> ~/gate_study/results/chain_round3_$H.log; exit 1
fi
mkdir -p $OUT
[ -f $OUT/queue.txt ] || (cd ~/gate_study/Adaptive_BCI/python/analysis/gate_study && python make_queue.py round3 > $OUT/queue.txt.tmp && mv -n $OUT/queue.txt.tmp $OUT/queue.txt)
for i in 0 1 2 3; do
  tmux new-session -d -s g3_$i "bash -ic 'conda activate chinghao_tseng && cd ~/gate_study/Adaptive_BCI/python && python analysis/gate_study/simulate_gates.py --start_ckpts ~/gate_study/results/pretrain_4fold_20261123/starts.json --out $OUT --queue $OUT/queue.txt --seeds 0,1,2 --delay 2 > $OUT/worker_${H}_$i.log 2>&1; echo EXIT \$? >> $OUT/worker_${H}_$i.log'"
  sleep 2
done
echo "$(date) round 3 started on $H" >> ~/gate_study/results/chain_round3_$H.log
