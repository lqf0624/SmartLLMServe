#!/bin/bash

python main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' --hardware 'RTX3090' --npu_num 1 --npu_group 1 --npu_mem 40 \
    --remote_bw 512 --link_bw 256 --fp 16 --block_size 4 \
    --dataset 'dataset/BurstGPT_1.csv' --output 'output/example_run.csv' \
    --verbose --req_num 10