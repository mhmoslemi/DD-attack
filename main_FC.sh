

#!/usr/bin/env bash
# Call the project venv interpreter directly. `source ENV/bin/activate` is broken
# here because the venv was created at the old path (/work/mohammad/DD-attack/ENV)
# and activate hardcodes it; the bare `python`/system python3 have no torch. The
# venv python still works when invoked by path. cd first so relative paths resolve.
cd "$(dirname "$0")"
PY=ENV/bin/python

# BUDGETS=(0.1 0.05 0.02 0.01 0.005 0.002)
BUDGETS=(0.02)

for budget in "${BUDGETS[@]}"; do


    
# --surrogate_model ResNet20BN --model ResNet20BN \
# ConvNetBN

CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PY" main_IF.py \
    --surrogate_model ResNet20BN --model ResNet20BN \
    --class_pairs dog-bird \
    --attack fc --restarts 1 \
    --budget "$budget" --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216 \
    --lambda_margin 1 \
    --num_surrogates 4 --surrogate_epochs 40 \
    --num_targets 8 --num_victims 5 \
    --victim_epochs 40 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 --multilayer \
    --target_select random --seed 0 --surrogate_on_full_data --victim_pipelines standard \
    --parallel_victims --single_surrogate --easy_targets \
    --curv_select --lambda_curv 1.0 --curv_damping 1.0 --curv_cg_iters 10 \
    --curv_hessian_bs 512 --curv_cand_bs 128
    #--random_select   # NOTE: --random_select bypasses scored selection -> disables --curv_select

    # --easy_targets
    #  --random_select
done



