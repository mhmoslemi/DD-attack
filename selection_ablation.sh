#!/usr/bin/env bash
# Selection ablation: compare alternative base-selection rules against the
# proposed rule, with EVERYTHING else fixed. The surrogate ensemble is trained
# once and cached, then reused by every criterion, so the only thing that varies
# is how the N_p bases are chosen. (Cheap too: 10 surrogates trained once.)
#
# Criteria (selection_strategies.CRITERIA):
#   target-aware   : pixel_l2  feat_l2  feat_cos  grad_cos  ours  anti
#   target-agnostic: gradnorm  el2n  margin
#   control        : random
#
# feat_l2 is exactly 'ours' with lambda=0, so it doubles as the lambda=0 point of
# the margin sweep. 'anti' (worst bases by the proposed score) is the sanity
# control: it should sink ASR if the score is meaningful.
#
# Default config = the FC 1% setting where the proposed method is strongest, and
# where crafting is cheap (no gradmatch restarts). Switch ATTACK/BUDGET below to
# match Table 1 (gradmatch, 0.5%) if you want parity with that table instead.

set -euo pipefail

SYN=result/res_DM_CIFAR10_ConvNet_50ipc.pt
OUT=result/sel_abl
LOG="${OUT}/selection_ablation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
echo "Logging to $LOG"
CACHE=$OUT/surrogates_K10.pt          # shared ensemble for ALL criteria

ATTACK=fc
BUDGET=0.01

COMMON=(
    --syn_data_path "$SYN"
    --surrogate_cache "$CACHE"
    --surrogate_model ConvNet --model ConvNetBN
    --class_pairs dog-bird
    --attack "$ATTACK"
    --budget "$BUDGET" --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216
    --num_surrogates 10 --surrogate_epochs 1000
    --num_targets 10 --num_victims 6
    --victim_epochs 80 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 60
    --lambda_margin 1.0 --single_surrogate
    --target_select random --seed 0
)

# order matters only in that the FIRST run trains + writes the cache; the rest
# load it. Put a fast/cheap criterion first.
# for CRIT in random pixel_l2 el2n gradnorm margin feat_cos grad_cos feat_l2 ours anti; do
for CRIT in random pixel_l2 el2n gradnorm margin; do
    echo ""
    echo "================= selection = ${CRIT} ================="
    python eval_selection.py "${COMMON[@]}" \
        --select_criterion "${CRIT}" \
        --out_dir "${OUT}/${CRIT}"
done

# order = ['random','pixel_l2','el2n','gradnorm','margin']


#

# ---- collate the comparison table -----------------------------------------
echo ""
echo "===== ASR / CTA by selection criterion ====="
python - "$OUT" <<'PY'
import sys, os, glob, json, numpy as np
out = sys.argv[1]
order = ['random','pixel_l2','el2n','gradnorm','margin']
rows = {}
for jf in glob.glob(os.path.join(out, '*', 'results_*.json')):
    d = json.load(open(jf))
    c = d.get('criterion')
    if not d['rows']:
        continue
    asr = np.array([r['poison_asr'] for r in d['rows']])
    cta = np.array([r['poison_cta'] for r in d['rows']])
    # per-target ASR mean already aggregates victims inside each row's poison_asr,
    # which is the per-target value; mean/sem over targets:
    rows[c] = (asr.mean(), asr.std(ddof=0) / np.sqrt(len(asr)), cta.mean() * 100)
print('  %-10s %8s %8s %8s' % ('criterion', 'ASR(%)', 'SEM', 'CTA(%)'))
for c in order:
    if c in rows:
        a, se, ct = rows[c]
        print('  %-10s %8.1f %8.1f %8.1f' % (c, a, se, ct))
PY
