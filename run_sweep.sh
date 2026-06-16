#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --time=0-0:45:00       # Time limit (DD-HH:MM:SS)
#SBATCH --gpus-per-node=h100:1  # Request 2 full H100 GPUs
#SBATCH --cpus-per-task=2      # Request CPU cores (adjust as needed; 12-24 is common for 2 GPUs)
#SBATCH --mem=6GB               # Request memory (adjust as needed)
#SBATCH --mail-user=mhmoslemi2338@gmail.com
#SBATCH --mail-type=ALL

SCRATCH_DIR="/home/mmoslem3/scratch/DD-attack"
LOG_DEST="$SCRATCH_DIR/logs"

module load python/3.11.5 cuda/12.6 cudnn

cp -r "$SCRATCH_DIR" "$SLURM_TMPDIR"
cd "$SLURM_TMPDIR/DD-attack"

mkdir -p "$LOG_DEST"

# Background log sync: rsync logs/ → scratch every 60 s so logs are live-readable from scratch
(while true; do
    rsync -a --update logs/ "$LOG_DEST/" 2>/dev/null
    sleep 10
done) &
SYNC_PID=$!

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

pip install --no-index --upgrade pip
pip install -r "$SLURM_TMPDIR/DD-attack/requirements.txt"



# ── CUDA device ──────────────────────────────────────────────────────────────
# CUDA_DEVICE=0          # <── change this to select the GPU (0, 1, 2, …)
# export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# ── Pre-selected targets (from select_targets.py) ────────────────────────────
# Run once:  python select_targets.py --model ConvNetBN --seed 0
# Then set the path here; leave empty ("") to fall back to random selection.
TARGET_IDX_FILE="result/selected_targets.json"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED=0
export PYTHONHASHSEED=$SEED   # deterministic hash randomisation (Python 3.3+)

mkdir -p logs

# ── Sweep grid ───────────────────────────────────────────────────────────────
# PAIRS=(dog-bird frog-airplane)
# ATTACKS=(fc gradmatch)

PAIRS=(frog-airplane)
ATTACKS=(fc)
MODELS=(ConvNetBN ResNet20 VGG13)

BUDGETS=(0.00002 0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1)

TOTAL=$(( ${#MODELS[@]} * ${#PAIRS[@]} * ${#ATTACKS[@]} * ${#BUDGETS[@]} ))
RUN=0
LOGFILE="logs/sweep_${SLURM_JOB_ID}.log"

for MODEL in "${MODELS[@]}"; do
for PAIR in "${PAIRS[@]}"; do
for ATTACK in "${ATTACKS[@]}"; do
for BUDGET in "${BUDGETS[@]}"; do

    RUN=$(( RUN + 1 ))
    TAG="${MODEL}_${PAIR}_${ATTACK}_b${BUDGET}_seed${SEED}"

    echo "──────────────────────────────────────────────────────────" | stdbuf -oL tee -a "$LOGFILE"
    echo "[${RUN}/${TOTAL}] ${TAG}" | stdbuf -oL tee -a "$LOGFILE"
    echo "──────────────────────────────────────────────────────────" | stdbuf -oL tee -a "$LOGFILE"

    # restarts only matter for gradmatch, but the flag is accepted by both
    if [[ "$ATTACK" == "gradmatch" ]]; then
        RESTARTS=8
    else
        RESTARTS=1
    fi

    EXTRA_ARGS=(--cache_dir result/cache)
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
        --victim_epochs 30 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 60 \
        --target_select random --seed "$SEED" --single_surrogate --clean_baseline \
        "${EXTRA_ARGS[@]}" \
        2>&1 | stdbuf -oL tee -a "$LOGFILE"

done
done
done
done

# Stop background sync and do a final copy to scratch
kill $SYNC_PID 2>/dev/null
cp "$LOGFILE" "$LOG_DEST/"

echo "══════════════════════════════════════════════════════════"
echo "Sweep done. ${TOTAL} runs — logs synced to $LOG_DEST"
echo "══════════════════════════════════════════════════════════"
