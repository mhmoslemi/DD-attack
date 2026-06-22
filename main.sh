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

# BUDGETS=(0.005 0.002 0.001 0.0005)


# BUDGETS=(0.0005)

for budget in "${BUDGETS[@]}"; do
    # CUDA_VISIBLE_DEVICES=5 python main_IF.py \
    #     --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
    #     --surrogate_model ResNet20BN --model ResNet20BN \
    #     --class_pairs dog-bird \
    #     --attack gradmatch --restarts 4 \
    #     --budget "$budget" --epsilon 0.0313725 --pgd_steps 75 --pgd_alpha 0.0039216 \
    #     --lambda_margin 1 \
    #     --num_surrogates 1 --surrogate_epochs 45 \
    #     --num_targets 10 --num_victims 5  \
    #     --victim_epochs 45 --victim_lr 0.1 --victim_bs 125 --victim_decay 30 \
    #     --target_select random --seed 0  --single_surrogate --multilayer --surrogate_on_full_data --random_select
        
        # --random_select
        
    CUDA_VISIBLE_DEVICES=5 python main_IF.py \
        --syn_data_path result/res_DM_CIFAR10_ConvNet_100ipc.pt \
        --surrogate_model ResNet20BN --model ResNet20BN \
        --class_pairs dog-bird \
        --attack fc --restarts 1 \
        --budget "$budget" --epsilon 0.0313725 --pgd_steps 150 --pgd_alpha 0.0039216 \
        --lambda_margin 1 \
        --num_surrogates 6 --surrogate_epochs 40 \
        --num_targets 10 --num_victims 5  \
        --victim_epochs 40 --victim_lr 0.1 --victim_bs 125 --victim_decay 35 \
        --target_select random --seed 0 --multilayer --surrogate_on_full_data \
        --cache_dir result/cache
        

        #  --single_surrogate  
        #  --random_select
        
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

