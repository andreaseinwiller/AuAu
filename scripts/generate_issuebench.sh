#!/bin/bash

num_prompts=1000
max_concurrent=20
language="en"
tag="issuebench_gen_jan13_$language"
models=(
    # "final_local"
    # "final_openrouter_batch1"
    # "final_openrouter_batch2"
    # "final_openrouter_batch3"
    # "final_openrouter_batch4"
    "final_openrouter_batch5"
    "final_openrouter_batch6"
)

# sessionName=ibgen
sessionName=ibgen-or
tmux kill-session -t $sessionName
tmux new -s $sessionName -d

for model in "${models[@]}"; do
    tmux new-window -t "$sessionName" -n "$model"

    model_file="${model}.json"

    cmd="uv run issuebench.py gen \
        --model-selection-file $model_file \
        --tag $tag \
        --num-prompts $num_prompts \
        --max-concurrent $max_concurrent \
        --language $language"

    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}_${model}.log" Enter
done
