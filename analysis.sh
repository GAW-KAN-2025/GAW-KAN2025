#!/bin/bash
python wgak_lime_analysis.py\
 --model_path ./checkpoints/WGAK_PEMS-BAY_12_bs512_completed.pt\
 --adj_path ./data/PEMS-BAY/adj_mx_bay.pkl \
 --dataset PEMS-BAY \
 --seq_len 12 \
 --pre_len 12 \
 --batch_size 512 \
 --sample_idx 0

if [ $? -eq 0 ]; then
    echo "Run finished!"
else
    echo "Run failed."
fi