#!/usr/bin/env bash
# Run the standard from-scratch (no-distill) poison eval on full real data.
# Surrogates share the victim arch (--model). Edit the vars below and launch:
#   bash run_exact.sh
set -euo pipefail

# ---- GPUs ------------------------------------------------------------------
# list the GPUs to use; >=2 enables --parallel_victims (round-robin over them)
export CUDA_VISIBLE_DEVICES=0

# ---- experiment knobs ------------------------------------------------------
MODEL=ConvNetBN                       # victim + surrogate arch
DATASET=CIFAR10
DATA_PATH=/home/mmoslem3/scratch/data
ATTACK=gradmatch                      # fc | gradmatch | influence
CLASS_PAIRS="dog-bird frog-airplane"  # 'poison-target'
BUDGET=0.01                           # fraction of the 50k train set
EPSILON=0.0313725                     # 8/255

NUM_SURROGATES=10
SURROGATE_EPOCHS=1000
NUM_TARGETS=10
NUM_VICTIMS=6

SEED=0
OUT_DIR=result/standard_nodistill
CACHE_DIR=cache
POISON_CACHE_DIR=cache/poisons
LOG_FILE=logs/exact_${ATTACK}_${MODEL}_seed${SEED}.log

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$POISON_CACHE_DIR" "$(dirname "$LOG_FILE")"

# enable multi-GPU parallel training when >1 GPU is visible
PARALLEL=""
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    PARALLEL="--parallel_victims"
fi

python Exact.py \
    --dataset "$DATASET" --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --attack "$ATTACK" \
    --class_pairs $CLASS_PAIRS \
    --budget "$BUDGET" --epsilon "$EPSILON" \
    --pgd_steps 250 --pgd_alpha 0.0039216 --restarts 8 \
    --num_surrogates "$NUM_SURROGATES" --surrogate_epochs "$SURROGATE_EPOCHS" \
    --num_targets "$NUM_TARGETS" --num_victims "$NUM_VICTIMS" \
    --victim_epochs 200 --victim_lr 0.1 --victim_bs 125 --victim_decay 100 150 \
    --target_select random \
    --clean_baseline \
    --exact_select \
    --cache_dir "$CACHE_DIR" \
    --poison_cache_dir "$POISON_CACHE_DIR" \
    --out_dir "$OUT_DIR" \
    --log_file "$LOG_FILE" \
    --seed "$SEED" \
    $PARALLEL
