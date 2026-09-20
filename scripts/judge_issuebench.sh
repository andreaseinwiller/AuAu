#!/bin/bash

max_concurrent=20
tag_prefix="issuebench_judge_jan14"
judgemodel="gemma4-sft"
languages=(
    "en"
    # "ru"
    # "de"
    # "zh"
)
model_file="${judgemodel}.json"


sessionName=ibjudge
tmux kill-session -t $sessionName
tmux new -s $sessionName -d

for language in "${languages[@]}"; do
    tmux new-window -t "$sessionName" -n "$language"
    tag="${tag_prefix}_${language}"

    cmd="uv run issuebench.py judge \
        --model-selection-file $model_file \
        --input-dir resources/output/issuebench_gen_jan13_$language-$language-1_0-1-42 \
        --tag $tag \
        --max-concurrent $max_concurrent \
        --language en"

    tmux send-keys -t "$sessionName:$language" "$cmd | tee logs/${tag}_${language}.log" Enter
done
