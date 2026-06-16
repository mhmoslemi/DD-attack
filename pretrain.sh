#!/bin/bash
# Run this locally before sbatch run_sweep.sh.
# Trains surrogates + clean victims for all 3 models and saves them to result/cache/.
# The sweep job copies result/ to SLURM_TMPDIR and loads from cache instead of retraining.

CACHE_DIR="result/cache"
SEED=0

SURROGATE_MODEL=ConvNet
NUM_SURROGATES=10
SURROGATE_EPOCHS=1000

VICTIM_EPOCHS=60
VICTIM_LR=0.1
VICTIM_BS=125

# CUDA_DEVICE=6          # <── change this to select the GPU (0, 1, 2, …)
# export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE


# ── 1. Surrogates (attack-independent; saved once) ───────────────────────────
echo "=== precomputing surrogates ==="
python -u main_IF.py \
    --precompute_only \
    --cache_dir "$CACHE_DIR" \
    --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
    --surrogate_model $SURROGATE_MODEL \
    --num_surrogates $NUM_SURROGATES \
    --surrogate_epochs $SURROGATE_EPOCHS \
    --attack fc \
    --class_pairs frog-airplane \
    --seed $SEED

# ── 2. Clean victims (one set per victim model) ───────────────────────────────
for MODEL in ConvNetBN ResNet20 VGG13; do
    echo ""
    echo "=== precomputing clean victims for $MODEL ==="
    python -u main_IF.py \
        --precompute_only \
        --clean_baseline \
        --cache_dir "$CACHE_DIR" \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
        --surrogate_model $SURROGATE_MODEL \
        --num_surrogates $NUM_SURROGATES \
        --surrogate_epochs $SURROGATE_EPOCHS \
        --model "$MODEL" \
        --num_victims 6 \
        --victim_epochs $VICTIM_EPOCHS \
        --victim_lr $VICTIM_LR \
        --victim_bs $VICTIM_BS \
        --victim_decay 40 \
        --attack fc \
        --class_pairs frog-airplane \
        --seed $SEED
done

echo ""
echo "=== pretrain done — cache is in $CACHE_DIR ==="
