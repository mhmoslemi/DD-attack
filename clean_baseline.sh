#!/usr/bin/env bash
# Clean-data baseline: train ConvNetBN / VGG13BN / ResNet20BN from scratch on
# the full clean CIFAR-10 train set (same victim recipe as main_FC.sh:
# 40 epochs, lr 0.1, bs 256, x0.1 decay @35), 5 runs per model, then print
# the ACC mean / var / std per model.

# --- logging: send everything (stdout + stderr) to a timestamped log file while
#     still printing to the terminal. Set LOG_FILE=... to override the path. ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/clean_baseline_$(date +%Y%m%d_%H%M%S).log"}
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== clean_baseline.sh started $(date '+%Y-%m-%d %H:%M:%S') — logging to $LOG_FILE ==="

RUNS=5
MODELS=(ConvNetBN VGG13BN ResNet20BN)

for model in "${MODELS[@]}"; do
    echo ""
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] clean baseline: $model ($RUNS runs) ==="
    python -u clean_acc.py \
        --model "$model" \
        --runs $RUNS \
        --data_path /home/mmoslem3/scratch/data \
        --victim_epochs 40 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 \
        --seed 0
done

echo ""
echo "=== summary ==="
grep "ACC mean" "$LOG_FILE"

echo "=== clean_baseline.sh finished $(date '+%Y-%m-%d %H:%M:%S') — log: $LOG_FILE ==="
