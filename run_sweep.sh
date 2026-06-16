#!/usr/bin/env bash
set -euo pipefail

# ── CUDA device ──────────────────────────────────────────────────────────────
CUDA_DEVICE=0          # <── change this to select the GPU (0, 1, 2, …)
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# ── Pre-selected targets (from select_targets.py) ────────────────────────────
# Run once:  python select_targets.py --model ConvNetBN --seed 0
# Then set the path here; leave empty ("") to fall back to random selection.
TARGET_IDX_FILE="result/selected_targets.json"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED=0
export PYTHONHASHSEED=$SEED   # deterministic hash randomisation (Python 3.3+)

mkdir -p logs

# ── Sweep grid ───────────────────────────────────────────────────────────────
MODELS=(ConvNetBN ResNet20 VGG13)
PAIRS=(dog-bird frog-airplane)
ATTACKS=(fc gradmatch)
BUDGETS=(0.00002 0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1)

TOTAL=$(( ${#MODELS[@]} * ${#PAIRS[@]} * ${#ATTACKS[@]} * ${#BUDGETS[@]} ))
RUN=0

for MODEL in "${MODELS[@]}"; do
for PAIR in "${PAIRS[@]}"; do
for ATTACK in "${ATTACKS[@]}"; do
for BUDGET in "${BUDGETS[@]}"; do

    RUN=$(( RUN + 1 ))
    TAG="${MODEL}_${PAIR}_${ATTACK}_b${BUDGET}_seed${SEED}"
    LOGFILE="logs/${TAG}.log"

    echo "──────────────────────────────────────────────────────────"
    echo "[${RUN}/${TOTAL}] model=${MODEL}  pair=${PAIR}  attack=${ATTACK}  budget=${BUDGET}"
    echo "  log → ${LOGFILE}"
    echo "──────────────────────────────────────────────────────────"

    # restarts only matter for gradmatch, but the flag is accepted by both
    if [[ "$ATTACK" == "gradmatch" ]]; then
        RESTARTS=8
    else
        RESTARTS=1
    fi

    # python -u  → unbuffered stdout/stderr (zero delay in log)
    # stdbuf -oL → line-buffer tee's stdout so the log is updated immediately
    EXTRA_ARGS=()
    [[ -n "$TARGET_IDX_FILE" ]] && EXTRA_ARGS+=(--target_idx_file "$TARGET_IDX_FILE")

    python -u main_IF.py \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
        --surrogate_model ConvNet --model "$MODEL" \
        --class_pairs "$PAIR" \
        --attack "$ATTACK" --restarts "$RESTARTS" \
        --budget "$BUDGET" --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216 \
        --lambda_margin 0.1 \
        --num_surrogates 10 --surrogate_epochs 1000 \
        --num_targets 10 --num_victims 6 \
        --victim_epochs 80 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 60 \
        --target_select random --seed "$SEED" --single_surrogate \
        "${EXTRA_ARGS[@]}" \
        2>&1 | stdbuf -oL tee "$LOGFILE"

done
done
done
done

echo "══════════════════════════════════════════════════════════"
echo "Sweep done. ${TOTAL} runs — logs in logs/"
echo "══════════════════════════════════════════════════════════"
