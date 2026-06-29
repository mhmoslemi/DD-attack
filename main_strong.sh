#!/usr/bin/env bash
# ============================================================================
# Strong-victim run: same clean-label poison attack, but the VICTIM is a real
# ResNet18 trained with a proper recipe so its CLEAN accuracy reaches ~95%.
#
# What makes it ~95% (vs the ~85% ConvNet runs):
#   * real net           : ResNet18BN  (3-layer ConvNet caps at ~85%)
#   * strong aug + wd     : pipeline "diffaug:wd=5e-4"  (DSA aug + weight decay)
#   * long schedule       : 200 epochs, LR x0.1 @ {100,150}, lr 0.1, bs 128
#
# The poison (selection + crafting on the surrogates) is optimized ONCE per
# target and cached in $POISON_CACHE, then every pipeline trains the strong
# victim from scratch on that same poisoned set. --clean_baseline trains clean
# victims with each recipe, so the ~95% number is reported as the per-method
# "clean CTA" right next to the poisoned CTA/ASR.
#
# Usage:  bash main_strong.sh [GPU]      # GPU defaults to 7 (the idle one)
#
# NOTE: ResNet18 @ 200 ep is ~15-20x a ConvNet @ 50 ep, so keep the grid SMALL.
# ============================================================================
set -u


# Keep it small -- a strong victim is expensive. Add budgets once you've timed one.
# BUDGETS=(0.01)
BUDGETS=(0.02 0.01 0.005)

# 95% recipe is "diffaug:wd=5e-4" (aug + weight decay). "standard" is kept for
# contrast (no aug/no wd -> overfits to ~89-91%, NOT 95).
PIPELINES=(standard diffaug:wd=5e-4)

MODEL=ResNet20BN          # victim arch
SURR=ResNet20BN             # surrogate arch (non-BN, trained on distilled S; mirrors ConvNet/ConvNetBN)
PAIR=dog-bird
ATTACK=gradmatch
POISON_CACHE=result/poison_cache_strong
LOG_DIR=logs/ablation_strong
mkdir -p "$LOG_DIR"

for budget in "${BUDGETS[@]}"; do
  # scored selection (ours), then random selection (ablation); each caches its own poison.
  for SELECT in scored random; do
    if [ "$SELECT" = random ]; then
        SELECT_FLAG="--random_select"; TAG="random"
    else
        SELECT_FLAG="";                TAG="scored"
    fi
    LOG_FILE="$LOG_DIR/${MODEL}_${PAIR}_${ATTACK}_b${budget}_strong_${TAG}_seed0.log"

    CUDA_VISIBLE_DEVICES=6 python -u main_IF.py \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
        --surrogate_model "$SURR" --model "$MODEL" \
        --class_pairs "$PAIR" \
        --attack "$ATTACK" --restarts 5 \
        --budget "$budget" --epsilon 0.0313725 --pgd_steps 75 --pgd_alpha 0.0039216 \
        --lambda_margin 1 \
        --num_surrogates 5 --surrogate_epochs 40 \
        --num_targets 8 --num_victims 5 --num_clean_victims 1 \
        --victim_epochs 200 --victim_lr 0.1 --victim_bs 128 --victim_decay 100 150 \
        --target_select random --seed 0 \
        --multilayer --single_surrogate \
        --cache_dir result/cache_strong \
        --out_dir "result/strong_${TAG}" \
        --victim_pipelines "${PIPELINES[@]}" \
        --poison_cache_dir "$POISON_CACHE" \
        --log_file "$LOG_FILE" --clean_baseline  --surrogate_on_full_data --multilayer \
        $SELECT_FLAG
  done
done
