#!/bin/bash

SCRIPTS=(main_DM.py main.py)
DATASETS=(CIFAR10 SVHN FashionMNIST STL10 MNIST)
IPCS=(10 50 100)

declare -A METHOD_MAP
METHOD_MAP["main.py"]="DC"
METHOD_MAP["main_DM.py"]="DM"

for script in "${SCRIPTS[@]}"; do
    method="${METHOD_MAP[$script]}"
    for dataset in "${DATASETS[@]}"; do
        for ipc in "${IPCS[@]}"; do
            pt="result/res_${method}_${dataset}_ConvNet_${ipc}ipc.pt"
            if [ -f "$pt" ]; then
                echo "=== Skipping: $pt already exists ==="
                continue
            fi
            echo "=== Running: python $script --dataset $dataset --ipc $ipc ==="
            python "$script" --dataset "$dataset" --ipc "$ipc"
        done
    done
done
