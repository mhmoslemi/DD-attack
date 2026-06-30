

#!/usr/bin/env bash
# Call the project venv interpreter directly. `source ENV/bin/activate` is broken
# here because the venv was created at the old path (/work/mohammad/DD-attack/ENV)
# and activate hardcodes it; the bare `python`/system python3 have no torch. The
# venv python still works when invoked by path. cd first so relative paths resolve.
cd "$(dirname "$0")"
PY=ENV/bin/python

BUDGETS=(0.1 0.05 0.02 0.01 0.005 0.002)
# BUDGETS=(0.02 0.01)
BUDGETS=(0.01)

for budget in "${BUDGETS[@]}"; do


    
# --surrogate_model ResNet20BN --model ResNet20BN \
# ConvNetBN

CUDA_VISIBLE_DEVICES=0,1,2,3,5 "$PY" main_IF.py \
    --surrogate_model ResNet20BN --model ResNet20BN \
    --class_pairs dog-bird \
    --attack fc --restarts 1 \
    --budget "$budget" --epsilon 0.0313725 --pgd_steps 100 --pgd_alpha 0.0039216 \
    --lambda_margin 1 \
    --num_surrogates 4 --surrogate_epochs 80 \
    --num_targets 8 --num_victims 5 \
    --victim_epochs 40 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 --multilayer \
    --target_select random --seed 0 --surrogate_on_full_data --victim_pipelines standard \
    --parallel_victims --single_surrogate --easy_targets \
    --target_cache_dir result/target_cache \
    --exact_select --if_hess_source syn --if_damping 0.01 --if_cg_iters 150 \
    --if_cg_tol 1e-4 --if_fd_h 0.01 --if_hess_size 512 --if_max_surrogates 1 --verbose

    # --random_select 
     
# 

    # ^ matches the FAST main_exact_if_models.sh recipe: 512-image subsampled
    #   Hessian + 1 surrogate (was slow before: 2000 imgs over all 4 surrogates).
    # --if_last_layer   # add for a much cheaper final-layer-only influence (classic IF)
    #--random_select    # NOTE: bypasses scored/exact selection (mutually exclusive)
    # --curv_select     # the OTHER curvature option (|| C_x^T H^-1 g_t ||_1 leverage)
done



