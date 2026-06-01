CUDA_VISIBLE_DEVICES=5 python poison_condensation_multi.py \
    --syn_data_path result/res_DM_CIFAR10_ConvNet_50ipc.pt \
    --out_dir result/attack_pgd_eps32 \
    --y_adv 3 --target_class dog --num_targets 4 \
    --attack pgd --epsilon 0.1255 --pgd_steps 500 --pgd_alpha 0.00784 \
    --N_p 1000 --screen_agree 3 \
    --num_clean_models 5 --num_surrogates 5 --surrogate_epochs 500 \
    --Iteration 3000 --num_victims 6 --victim_epochs 500





# ================================ SWEEP RESULTS ================================
#   clean baseline CTA (pool B) = 0.6218
#      idx       true  clean_ASR   pois_CTA   pois_ASR     Linf
#     2459        dog         0%     0.5887        83%   0.1255
#     2259        dog         0%     0.5872       100%   0.1255
#     9654        dog         0%     0.5939       100%   0.1255
#     1774        dog         0%     0.5880       100%   0.1255
#   ----------------------------------------------------------------------------
#   mean poison ASR = 95.8% +/- 7.2%   mean clean ASR = 0.0%
#   mean poison CTA = 0.5895 +/- 0.0026 (clean CTA 0.6218, drop +0.0324)
#   attack increased ASR on 4/4 targets