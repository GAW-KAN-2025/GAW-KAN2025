#!/bin/bash

# 支持传入--dataset参数，默认PEMS-BAY
DATASET=${1:-PEMS-BAY}
METHOD=${2:-LIME}

if [ "$DATASET" == "PEMS-BAY" ]; then
    MODEL_PATH=./checkpoints/WGAK_PEMS-BAY_12_bs512_completed.pt
    ADJ_PATH=./data/PEMS-BAY/adj_mx_bay.pkl
elif [ "$DATASET" == "ST-EVCDP" ]; then
    MODEL_PATH=./checkpoints/WGAK_ST-EVCDP_12_bs512_completed.pt
    ADJ_PATH=./data/ST-EVCDP/adj.csv
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

if [ "$METHOD" == "LIME" ] || [ "$METHOD" == "BOTH" ]; then
    python wgak_lime_analysis.py\
     --model_path $MODEL_PATH\
     --adj_path $ADJ_PATH \
     --dataset $DATASET \
     --seq_len 12 \
     --pre_len 12 \
     --batch_size 512 \
     --sample_idx 0
    if [ $? -eq 0 ]; then
        echo "LIME Run finished!"
    else
        echo "LIME Run failed."
    fi
fi

if [ "$METHOD" == "SHAP" ] || [ "$METHOD" == "BOTH" ]; then
    python shap.py\
     --model_path $MODEL_PATH\
     --adj_path $ADJ_PATH \
     --dataset $DATASET \
     --seq_len 12 \
     --pre_len 12 \
     --batch_size 512 \
     --sample_idx 0
    if [ $? -eq 0 ]; then
        echo "SHAP Run finished!"
    else
        echo "SHAP Run failed."
    fi
fi