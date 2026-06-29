#!/usr/bin/env bash
# ============================================================================
# Selection-only ablation:  SMART-select (ours)  vs  EXACT influence function.
#
# Runs main_IF_exact.py, which reuses the SAME cached surrogates, the SAME fc
# craft, and the SAME from-scratch victim training as main_IF.py -- only the
# base-SELECTION rule changes:
#   smart : first-order gradient alignment (drops H^{-1}); your ICLR method.
#   exact : keeps the curvature term, score = grad l(x_t)^T H^{-1} grad l(z),
#           with H^{-1}g solved by conjugate gradients over Hessian-vector
#           products (H = Hessian of the surrogate risk on the distilled S).
#
# Reports, per method: final ASR, CTA, and the SELECTION wall-clock time
# (the only computation that differs). Cached surrogates from main.sh are
# reused (same --cache_dir / --surrogate_model / epochs / seed), so no retrain.
#
# Usage:  bash main_exact_if.sh [GPU]      # GPU defaults to 7 (idle)
# ============================================================================
set -u

# GPU=${1:-7}

BUDGETS=(0.01 0.02)
# BUDGETS=(0.005 0.01 0.02)

MODEL=ConvNetBN
SURR=ConvNet
PAIR=dog-bird
LOG_DIR=logs/select_cmp
mkdir -p "$LOG_DIR"

for budget in "${BUDGETS[@]}"; do
    LOG_FILE="$LOG_DIR/${MODEL}_${PAIR}_fc_b${budget}_smart_vs_exactIF_seed0.log"

    CUDA_VISIBLE_DEVICES=6 python -u main_IF_exact.py \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
        --surrogate_model "$SURR" --model "$MODEL" \
        --class_pairs "$PAIR" \
        --budget "$budget" --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216 \
        --num_surrogates 10 --surrogate_epochs 1000 --single_surrogate \
        --lambda_margin 1 --base_dist l2 \
        --num_targets 10 --num_victims 6 \
        --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 \
        --target_select random --seed 0 \
        --cache_dir result/cache \
        --out_dir result/select_cmp \
        --methods smart exact \
        --if_hess_source syn --if_damping 0.01 --if_cg_iters 100 --if_cg_tol 1e-4 \
        --if_fd_h 0.01 \
        --log_file "$LOG_FILE" --verbose

        # add 'random' to --methods for a lower-bound baseline
        # --if_last_layer            # cheaper classic IF (final linear only)
        # --if_max_surrogates 1      # cap exact-IF to 1 surrogate to cut its time
done
