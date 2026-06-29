#!/usr/bin/env bash
# Ablation: isolate the contribution of sample selection vs. surrogate ensemble.
#
# 2x2 design:
#   rows: random select (baseline) vs. scored select (ours)
#   cols: single surrogate (baseline) vs. full ensemble (ours)
#
# Cell A — plain FC baseline:     random select + single surrogate
# Cell B — +selection only:       scored select + single surrogate
# Cell C — +ensemble only:        random select + full ensemble
# Cell D — full method:           scored select + full ensemble
#
# The claim is that selection (A→B) drives most of the gain over (A→C).

# COMMON="python eval_standard_nodistill.py
#     --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt
#     --surrogate_model ConvNet --model ConvNetBN
#     --class_pairs dog-bird
#     --budget 0.01 --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216
#     --lambda_margin 1.0
#     --num_surrogates 10 --surrogate_epochs 1000
#     --num_targets 10 --num_victims 6
#     --victim_epochs 60 --victim_lr 0.1 --victim_bs 125 --victim_decay 40
#     --target_select random --seed 0 --clean_baseline"



# model ConvNetBN, ResNet20, VGG13
# class_pairs dog-bird frog-airplane
# attack FC gradmatch
# budget 0.00002 0.0001 0.001 0.002 0.005 0.01 0.02 0.05 0.1
# --surrogate_model ResNet20BN --model ResNet20BN \

BUDGETS=(0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005)
BUDGETS=(0.02 0.01 0.005 0.002)

# BUDGETS=(0.005 0.002 0.001 0.0005)


# BUDGETS=(0.0005)

# ----------------------------------------------------------------------------
# Training-pipeline ablations.
#
# The poison (selection + crafting) is optimized ONCE per target and cached in
# $POISON_CACHE; every pipeline below trains victims from scratch on that SAME
# poisoned set, so we only measure how robust each training recipe is.
# Plain/standard training is already done, so it is NOT in this list.
#
# Tune any pipeline inline, e.g. advtrain:eps=0.0314,steps=7  mixup:alpha=1.0
# ----------------------------------------------------------------------------
# PIPELINES=(diffaug mixup cutmix advtrain dpsgd labelsmooth)
PIPELINES=(standard diffaug cutmix advtrain:alpha=0.0156863,steps=7 dpsgd)

MODEL=ConvNetBN
PAIR=dog-bird
ATTACK=fc
POISON_CACHE=result/poison_cache
LOG_DIR=logs/ablation
mkdir -p "$LOG_DIR"

for budget in "${BUDGETS[@]}"; do
  # Run each budget twice: scored selection (ours), then random selection (ablation).
  # The poison cache key already distinguishes the two, so each crafts its own
  # poison once and reuses it across every pipeline.
  for SELECT in scored random; do
    if [ "$SELECT" = random ]; then
        SELECT_FLAG="--random_select"
        TAG="random"
    else
        SELECT_FLAG=""
        TAG="scored"
    fi
    LOG_FILE="$LOG_DIR/${MODEL}_${PAIR}_${ATTACK}_b${budget}_ablation_${TAG}_seed0.log"

    CUDA_VISIBLE_DEVICES=6 python -u main_IF.py \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
        --surrogate_model ConvNet --model ConvNetBN \
        --class_pairs "$PAIR" \
        --attack "$ATTACK" --restarts 1 \
        --budget "$budget" --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216 \
        --lambda_margin 1 \
        --num_surrogates 10 --surrogate_epochs 1000 \
        --num_targets 8 --num_victims 5 --num_clean_victims 1 \
        --victim_epochs 45 --victim_lr 0.1 --victim_bs 125 --victim_decay 35 \
        --target_select random --seed 0  \
        --cache_dir result/cache \
        --out_dir "result/ablation_${TAG}" \
        --victim_pipelines "${PIPELINES[@]}" \
        --poison_cache_dir "$POISON_CACHE" \
        --log_file "$LOG_FILE" --single_surrogate --clean_baseline \
        $SELECT_FLAG
  done



        #  --single_surrogate  
        #  --random_select
        # --multilayer
        
        #  --surrogate_on_full_data
        
        

        # --random_select
    # CUDA_VISIBLE_DEVICES=5 python main_IF.py \
    #     --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
    #     --surrogate_model VGG13BN --model VGG13BN \
    #     --class_pairs dog-bird \
    #     --attack fc --restarts 1 \
    #     --budget "$budget" --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216 \
    #     --lambda_margin 1 \
    #     --num_surrogates 1 --surrogate_epochs 40 \
    #     --num_targets 10 --num_victims 5  \
    #     --victim_epochs 40 --victim_lr 0.1 --victim_bs 125 --victim_decay 35 \
    #     --target_select random --seed 0  --single_surrogate --surrogate_on_full_data --random_select


        # --surrogate_on_full_data --base_dist cosine

    # python main_IF.py \
    #     --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
    #     --surrogate_model ResNet20BN --model ResNet20BN \
    #     --class_pairs dog-bird \
    #     --attack gradmatch --restarts 8 \
    #     --budget "$budget" --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216 \
    #     --lambda_margin 1 \
    #     --num_surrogates 1 --surrogate_epochs 1000 \
    #     --num_targets 10 --num_victims 5  \
    #     --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 \
    #     --target_select random --seed 0  --single_surrogate --random_select
done

# --base_dist cosine

# echo "===== Cell A: plain FC (random select + single surrogate) ====="
# $COMMON --random_select --single_surrogate --out_dir result/abl_A_plain_fc

# echo "===== Cell B: +selection (scored select + single surrogate) ====="
# $COMMON --single_surrogate --out_dir result/abl_B_selection_only

# echo "===== Cell C: +ensemble (random select + full ensemble) ====="
# $COMMON --random_select --out_dir result/abl_C_ensemble_only

# echo "===== Cell D: full method (scored select + full ensemble) ====="
# $COMMON --out_dir result/abl_D_full_method


# for grad match
# margin 10  --> ASR = 43.3%
# margin 2.0 --> ASR = 43.3%
# margin 1.0 --> ASR = 46.7%
# margin 0.1 --> ASR = 23.3%

