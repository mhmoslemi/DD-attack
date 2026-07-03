

#!/usr/bin/env bash
# Call the project venv interpreter directly. `source ENV/bin/activate` is broken
# here because the venv was created at the old path (/work/mohammad/DD-attack/ENV)
# and activate hardcodes it; the bare `python`/system python3 have no torch. The
# venv python still works when invoked by path. cd first so relative paths resolve.
# cd "$(dirname "$0")"
# PY=ENV/bin/python

# --- logging: send everything (stdout + stderr) to a timestamped log file while
#     still printing to the terminal. Set LOG_FILE=... to override the path. ---
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/main_FC_$(date +%Y%m%d_%H%M%S).log"}
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== main_FC.sh started $(date '+%Y-%m-%d %H:%M:%S') — logging to $LOG_FILE ==="

BUDGETS=(0.1 0.05 0.02 0.01 0.005 0.002)
BUDGETS=(0.1 0.05 0.02 0.01 0.005 0.002 0.001)
BUDGETS=(0.05 0.02 0.01 0.005 0.002 0.001)
BUDGETS=(0.001)
for budget in "${BUDGETS[@]}"; do

    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] budget=$budget ==="

# 0.02 0.05 0.002 0.005 0.01
    
# --surrogate_model ResNet20BN --model ResNet20BN \
# ConvNetBN
# ResNet20BN dog-bird frog-airplane

# python main_IF.py \
#     --surrogate_model ConvNetBN --model ConvNetBN \
#     --class_pairs frog-airplane \
#     --data_path /home/mmoslem3/scratch/data \
#     --attack gradmatch --restarts 4 \
#     --budget "$budget" --epsilon 0.0313725 --pgd_steps 100 --pgd_alpha 0.0039216 \
#     --lambda_margin 1 \
#     --num_surrogates 4 --surrogate_epochs 80 \
#     --num_targets 10 --num_victims 5 \
#     --victim_epochs 40 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 --multilayer \
#     --target_select random --seed 0 --surrogate_on_full_data --victim_pipelines standard \
#     --single_surrogate \
#     --exact_select --if_hess_source syn --if_damping 0.01 --if_cg_iters 100 \
#     --if_cg_tol 1e-4 --if_fd_h 0.01 --if_hess_size 512 --if_max_surrogates 1 --verbose \
#     --easy_targets 0.7 \
#     --target_cache_dir result/target_cache \

# --parallel_victims 
    # --surrogate_model VGG13BN --model VGG13BN \
#   


python main_IF.py \
    --surrogate_model ConvNetBN --model ConvNetBN \
    --class_pairs frog-airplane \
    --data_path /home/mmoslem3/scratch/data \
    --attack gradmatch --restarts 4 \
    --budget "$budget" --epsilon 0.0313725 --pgd_steps 100 --pgd_alpha 0.0039216 \
    --lambda_margin 1 \
    --num_surrogates 4 --surrogate_epochs 80 \
    --num_targets 10 --num_victims 5 \
    --victim_epochs 40 --victim_lr 0.1 --victim_bs 256 --victim_decay 35 --multilayer \
    --target_select random --seed 0 --surrogate_on_full_data --victim_pipelines standard \
    --parallel_victims --single_surrogate \
    --target_cache_dir result/target_cache   --easy_targets 0.7 \
    --random_select 
    
     

    # ^ matches the FAST main_exact_if_models.sh recipe: 512-image subsampled
    #   Hessian + 1 surrogate (was slow before: 2000 imgs over all 4 surrogates).
    # --if_last_layer   # add for a much cheaper final-layer-only influence (classic IF)
    #--random_select    # NOTE: bypasses scored/exact selection (mutually exclusive)
    # --curv_select     # the OTHER curvature option (|| C_x^T H^-1 g_t ||_1 leverage)
done

echo "=== main_FC.sh finished $(date '+%Y-%m-%d %H:%M:%S') — log: $LOG_FILE ==="



