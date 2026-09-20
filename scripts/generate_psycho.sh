#!/bin/bash

max_concurrent=20
language="zh"
tag="final"
models=(
    "final_local"
    # "final_openrouter_batch1"
    # "final_openrouter_batch2"
    # "final_openrouter_batch3"
    # "final_openrouter_batch4"
    # "final_openrouter_batch5"
    # "final_openrouter_batch6"
)

sessionName=psycho-zh
tmux kill-session -t $sessionName
tmux new -s $sessionName -d

for model in "${models[@]}"; do
    tmux new-window -t "$sessionName" -n "$model"

    model_file="${model}.json"

    cmd="uv run main.py \
        --model_selection_file $model_file \
        --runs 10 \
        --output_dir_prefix_tag $tag \
        --max_concurrent $max_concurrent \
        --language $language"

    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}_${model}.log" Enter
done
