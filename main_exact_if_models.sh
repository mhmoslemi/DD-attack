#!/usr/bin/env bash
# ============================================================================
# Selection-TIME comparison across ARCHITECTURES (select-only).
#
# Extends the smart vs exact-IF selection ablation to ResNet20BN and VGG13BN.
# Here we ONLY run the base-SELECTION step and time it -- NO crafting, NO victim
# training (--select_only). The point is how the EXACT influence selection (CG
# over Hessian-vector products on the surrogate params) scales with model size;
# smart stays cheap (feature distance + margin).
#
# The surrogate IS the model (so gradients/Hessian are on that architecture),
# trained on the distilled S like the ConvNet run. If those surrogates are not
# cached yet they are trained once and cached (NOT part of the timed selection).
#
# Usage:  bash main_exact_if_models.sh [GPU]      # GPU defaults to 7
# ============================================================================
set -u


MODELS=(ResNet20BN VGG13BN)
BUDGETS=(0.01)
PAIR=dog-bird
NUM_SURR=1            # match the ConvNet run so total sel-time is comparable
SURR_EP=1000
TARGS=10
LOG_DIR=logs/select_cmp
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    LOG_FILE="$LOG_DIR/${model}_${PAIR}_b${budget}_smart_vs_exactIF_selectonly_seed0.log"

    CUDA_VISIBLE_DEVICES=6 python -u main_IF_exact.py \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
        --surrogate_model "$model" --model "$model" \
        --class_pairs "$PAIR" \
        --budget "$budget" --epsilon 0.0313725 \
        --num_surrogates "$NUM_SURR" --surrogate_epochs "$SURR_EP" \
        --lambda_margin 1 --base_dist l2 --multilayer \
        --num_targets "$TARGS" --target_select random --seed 0 \
        --cache_dir result/cache \
        --out_dir result/select_cmp \
        --methods smart exact --select_only \
        --if_hess_source syn --if_damping 0.01 --if_cg_iters 100 --if_cg_tol 1e-4 \
        --if_fd_h 0.01 --if_hess_size 512 \
        --log_file "$LOG_FILE" --verbose

        # --if_last_layer        # final-linear-only IF (much cheaper) if full-param is too slow/OOM
        # --if_max_surrogates 1  # time 1 surrogate instead of all 10
  done
done
