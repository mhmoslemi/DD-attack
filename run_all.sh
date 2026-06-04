#!/bin/bash

# SCRIPTS=(main.py main_DM.py)
SCRIPTS=(main_DM.py)
DATASETS=(SVHN FashionMNIST STL10)
IPCS=(10 50 100)

for script in "${SCRIPTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for ipc in "${IPCS[@]}"; do
            echo "=== Running: python $script --dataset $dataset --ipc $ipc ==="
            CUDA_VISIBLE_DEVICES=6 python "$script" --dataset "$dataset" --ipc "$ipc"
        done
    done
done
