#!/bin/bash
# ws-4090-02：等接手的第二輪 worker (g2b_*) 結束 -> 第三輪 (延遲 2，用自己的主機名) -> 第四輪 (以 ws-4090-01 的名義，接第二輪綁在 ws-4090-01 的 session)
G=~/gate_study/Adaptive_BCI/python/analysis/gate_study
CK=~/gate_study/results/pretrain_4fold_20261123/starts.json
R3=~/gate_study/results/round3_delay2_20261123
R4=~/gate_study/results/round4_shadow_20261123
LOG=~/gate_study/results/chain_ws-4090-02.log
gpu_free() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while tmux ls 2>/dev/null | grep -q '^g2b_'; do sleep 60; done
gpu_free || { echo "$(date) GPU busy, round 3 not started" >> $LOG; exit 1; }
for i in 0 1 2 3; do
  tmux new-session -d -s g3b_$i "bash -ic 'conda activate chinghao_tseng && $G/run_worker_as.sh $(hostname) $R3/worker_ws-4090-02_$i.log --start_ckpts $CK --out $R3 --queue $R3/queue.txt --seeds 0,1,2 --delay 2'" < /dev/null
  sleep 3
done
echo "$(date) round 3 started" >> $LOG
sleep 120
while tmux ls 2>/dev/null | grep -q '^g3b_'; do sleep 60; done
gpu_free || { echo "$(date) GPU busy, round 4 not started" >> $LOG; exit 1; }
for i in 0 1 2 3; do
  tmux new-session -d -s g4b_$i "bash -ic 'conda activate chinghao_tseng && $G/run_worker_as.sh ws-4090-01 $R4/worker_ws-4090-02_$i.log --start_ckpts $CK --out $R4 --queue $R4/queue.txt'" < /dev/null
  sleep 3
done
echo "$(date) round 4 started" >> $LOG
