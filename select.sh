python eval_standard_nodistill.py \
    --syn_data_path result/res_DM_CIFAR10_ConvNet_10ipc.pt \
    --surrogate_model ConvNet --model ConvNetBN \
    --class_pairs dog-bird \
    --budget 0.01 --epsilon 0.0313725 --pgd_steps 50 --pgd_alpha 0.0039216 \
    --lambda_margin 1.0 \
    --num_surrogates 10 --surrogate_epochs 500 \
    --num_targets 5 --num_victims 4 \
    --victim_epochs 50 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 \
    --target_select random --seed 0 --clean_baseline  # --random_select
    # --random_select --target_select random --seed 0
    


# python eval_standard_nodistill.py \
#     --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
#     --surrogate_model ConvNet --model ConvNetBN \
#     --class_pairs dog-bird \
#     --budget 0.01 --epsilon 0.0313725 --pgd_steps 250 --pgd_alpha 0.0039216 \
#     --lambda_margin 1.0 \
#     --num_surrogates 5 --surrogate_epochs 1000 \
#     --num_targets 10 --num_victims 5 \
#     --victim_epochs 25 --victim_lr 0.1 --victim_bs 125 --victim_decay 40 \
#     --random_select --target_select random --seed 0
    