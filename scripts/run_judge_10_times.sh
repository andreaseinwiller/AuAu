#!/bin/bash

language="en"

sessionName=judge-rerun
tmux kill-session -t $sessionName
tmux new -s $sessionName -d

for repetition in 1 2 3 4 5 6 7 8 9; do
    cmd="uv run issuebench.py eval --model-selection-file kimik2.json --tag issuebench_eval_v8_rebuttal/repetitions/${repetition} --max-concurrent 20 --language ${language}"
    tmux send-keys -t "$sessionName" "$cmd" Enter
done