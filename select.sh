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



python eval_standard_nodistill.py \
    --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
    --surrogate_model ConvNet --model ConvNetBN \
    --class_pairs dog-bird \
    --attack gradmatch --restarts 8 \
    --budget 0.005 --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216 \
    --lambda_margin 0.1 \
    --num_surrogates 10 --surrogate_epochs 1000 \
    --num_targets 5 --num_victims 6 \
    --victim_epochs 60 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 \
    --target_select random --seed 0 

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

