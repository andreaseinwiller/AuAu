#!/usr/bin/env bash
set -euo pipefail

SESSION_PREFIX="or6"
OUTPUT_DIR_PREFIX_TAG="final"
MODEL_SELECTION_FILE="final_openrouter_batch6.json"
LANGUAGE="de"
REVERSE_SCALE="False"
AUTH_SYS_PROMPT="False"

DATASETS=(F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS)

PROJECT_DIR="$HOME/llm-audit"
VENV_ACTIVATE=".venv/bin/activate"

for DATASET in "${DATASETS[@]}"; do
    SESSION_NAME="${SESSION_PREFIX}-${DATASET}-${LANGUAGE}"

    # Skip if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session already exists: $SESSION_NAME (skipping)"
        continue
    fi

    echo "Starting tmux session: $SESSION_NAME"

    tmux new-session -d -s "$SESSION_NAME" "
        conda deactivate || true
        cd \"$PROJECT_DIR\"
        source \"$VENV_ACTIVATE\"
        CUDA_VISIBLE_DEVICES=0,1 uv run main.py \
            --seed 42 \
            --gpus 2 \
            --temperature 1.0 \
            --runs 10 \
            --language $LANGUAGE \
            --model_selection_file $MODEL_SELECTION_FILE \
            --judge_selection_file judges.json \
            --datasets $DATASET \
            --output_dir_prefix_tag $OUTPUT_DIR_PREFIX_TAG \
            --reverse_scale $REVERSE_SCALE \
            --auth_sys_prompt $AUTH_SYS_PROMPT \
            --max_concurrent 20
    "
done

echo "Done. All tmux sessions started."