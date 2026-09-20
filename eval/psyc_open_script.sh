#!/bin/bash

N=100

for i in $(seq 1 $N); do
    TAG=$(printf "%04d" $i)
    CUDA_VISIBLE_DEVICES=1 uv run eval/psyc_open_judge_rep.py judge \
        --seed 42 \
        --gpus 1 \
        --temperature 1.0 \
        --runs 1 \
        --language en \
        --model-selection-file judges.json \
        --tag "$TAG" \
        --max-sequence-length 32000 \
        --max-concurrent 1
done