#!/bin/bash

max_concurrent=20
tag_prefix="issuebench_judge_jan25_auth"
judgemodel="kimik2"
inputdirs=(
    "ai-sage"
    # "allenai"
    "anthropic"
    # "deepseek"
    # "google"
    # "mistralai"
    # "openai"
    # "Qwen"
    # "t-tech"
    # "utter-project"
    # "Vikhrmodels"
    # "x-ai"
    # "yandex"
)
model_file="${judgemodel}.json"
language="en"


sessionName=ibjudge-auth
tmux kill-session -t $sessionName
tmux new -s $sessionName -d

for inputdir in "${inputdirs[@]}"; do
    tmux new-window -t "$sessionName" -n "$inputdir"
    tag="${tag_prefix}_${language}"

    cmd="uv run issuebench.py judge \
        --model-selection-file $model_file \
        --input-dir resources/output/issuebench_gen_jan24_auth_$language-$language-1_0-1-42/IssueBench/generate/$inputdir \
        --tag $tag \
        --max-concurrent $max_concurrent \
        --language en"

    tmux send-keys -t "$sessionName:$inputdir" "$cmd | tee logs/${tag}__${inputdir}_${language}.log" Enter
done
