#!/bin/bash



LOG_DEST="logs"
mkdir -p "$LOG_DEST"

# Background log sync not needed locally — logs written directly to logs/
# ── Parallelism ───────────────────────────────────────────────────────────────
NUM_GPUS=1

# ── Pre-selected targets (from select_targets.py) ────────────────────────────
TARGET_IDX_FILE=""  # set to "result/selected_targets.json" to re-enable JSON target loading

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED=0
export PYTHONHASHSEED=$SEED

# ── Sweep grid ───────────────────────────────────────────────────────────────
# PAIRS=(dog-bird frog-airplane)
PAIRS=(frog-airplane)
# ATTACKS=(fc gradmatch)
ATTACKS=(fc)
# MODELS=(ConvNetBN ResNet20 VGG13)
MODELS=(ConvNetBN)

# BUDGETS=(0.00002 0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
# BUDGETS=(0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0001 0.00002)
BUDGETS=(0.05)

TOTAL=$(( ${#MODELS[@]} * ${#PAIRS[@]} * ${#ATTACKS[@]} * ${#BUDGETS[@]} ))
RUN=0
SLOT=0
declare -a PIDS

# run_combo <gpu_id> <tag> <extra python args...>
run_combo() {
    local GPU=$1; local TAG=$2; shift 2
    local RUNLOG="logs/${TAG}.log"

    echo "──────────────────────────────────────────────────────────" >> "$RUNLOG"
    echo "[GPU ${GPU}] ${TAG}" >> "$RUNLOG"
    echo "──────────────────────────────────────────────────────────" >> "$RUNLOG"

    CUDA_VISIBLE_DEVICES=$GPU python -u main_IF.py "$@" \
        2>&1 | stdbuf -oL tee -a "$RUNLOG"
}

for MODEL in "${MODELS[@]}"; do
for PAIR in "${PAIRS[@]}"; do
for ATTACK in "${ATTACKS[@]}"; do
for BUDGET in "${BUDGETS[@]}"; do

    RUN=$(( RUN + 1 ))
    TAG="${MODEL}_${PAIR}_${ATTACK}_b${BUDGET}_seed${SEED}"
    GPU=$(( SLOT % NUM_GPUS ))

    if [[ "$ATTACK" == "gradmatch" ]]; then RESTARTS=8; else RESTARTS=1; fi

    EXTRA_ARGS=(--cache_dir result/cache)
    [[ -n "$TARGET_IDX_FILE" ]] && EXTRA_ARGS+=(--target_idx_file "$TARGET_IDX_FILE")

# --data_path /home/mmoslem3/scratch/data
    # Set to "--surrogate_on_full_data" to train surrogates on real data instead of distilled S
    SURROGATE_DATA_FLAG=""
    # SURROGATE_DATA_FLAG="--surrogate_on_full_data"

    COMMON_ARGS=(
        --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt
        --surrogate_model ConvNet --model "$MODEL"
        --class_pairs "$PAIR"
        --attack "$ATTACK" --restarts "$RESTARTS"
        --budget "$BUDGET" --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216
        --lambda_margin 1
        --num_surrogates 2 --surrogate_epochs 1000
        --num_targets 10 --num_victims 5 
        --victim_epochs 60 --victim_lr 0.1 --victim_bs 125 --victim_decay 40
        --target_select random --seed "$SEED" --single_surrogate # --clean_baseline
        "${EXTRA_ARGS[@]}"
        ${SURROGATE_DATA_FLAG}
    )

    echo "[${RUN}/${TOTAL}] ${TAG} → GPU ${GPU}"

    # main run + its baseline run sequentially on the same GPU, in background
    (
        # run_combo "$GPU" "$TAG"              "${COMMON_ARGS[@]}"
        run_combo "$GPU" "${TAG}_random_bl"  "${COMMON_ARGS[@]}" --random_select
    ) &
    PIDS[$SLOT]=$!

    SLOT=$(( SLOT + 1 ))

    # once we've filled all GPU slots, wait for all to finish before continuing
    if (( SLOT % NUM_GPUS == 0 )); then
        wait "${PIDS[@]}"
        PIDS=()
    fi

done
done
done
done

# wait for any remaining jobs (last batch may be smaller than NUM_GPUS)
wait

echo "══════════════════════════════════════════════════════════"
echo "Sweep done. ${TOTAL} combos (x2 with baseline) — logs in $LOG_DEST"
echo "══════════════════════════════════════════════════════════"
