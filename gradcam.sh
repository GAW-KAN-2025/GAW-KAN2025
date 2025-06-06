#!/bin/bash

# 支持传入--dataset参数，默认PEMS-BAY
DATASET=${1:-PEMS-BAY}
SAMPLE_IDX=${2:-0}
TARGET_LAYER=${3:-layers.0}

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

python GradCAM.py \
 --model_path $MODEL_PATH \
 --adj_path $ADJ_PATH \
 --dataset $DATASET \
 --seq_len 12 \
 --pre_len 12 \
 --batch_size 512 \
 --sample_idx $SAMPLE_IDX \
 --target_layer $TARGET_LAYER

if [ $? -eq 0 ]; then
    echo "GradCAM run finished!"
else
    echo "GradCAM run failed."
fi
