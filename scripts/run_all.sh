#!/bin/bash

set -e  # Exit on error (can be disabled for specific commands)

# Configuration
language="en"
max_concurrent=20  # parallel requests to LLM API
repetitions=10
tag=""  # Leave empty to integrate with existing results

# Check if models are provided as arguments
if [ $# -eq 0 ]; then
    echo "Error: No models provided"
    echo "Usage: $0 <model1> <model2> ..."
    echo "Example: $0 myModel1 myModel2 myModel3"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first."
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  macOS: brew install tmux"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Store all arguments as models array
models=("$@")

echo "Running experiments for models:" "${models[@]}"
echo "Configuration: language=$language, repetitions=$repetitions, max_concurrent=$max_concurrent"

sessionName=auau-$language

# Kill existing session if it exists (suppress error if it doesn't)
tmux kill-session -t $sessionName 2>/dev/null || true

# Create new session
tmux new -s $sessionName -d

for model in "${models[@]}"; do
    echo "Setting up experiments for model: $model"
    tmux new-window -t "$sessionName" -n "$model"

    model_file="${model}.json"

    # ========================================================================
    # PSYCHOMETRIC EXPERIMENTS (closed and open)
    # Comment out any section to skip it
    # ========================================================================

    # Standard psychometric experiments
    cmd="uv run main.py \
        --model_selection_file $model_file \
        --runs $repetitions \
        --output_dir_prefix_tag ${tag}final \
        --max_concurrent $max_concurrent \
        --datasets F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS \
        --language $language"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}final_${model}.log; echo '=== Finished: final ===' " Enter

    # Reversed option order
    cmd="uv run main.py \
        --model_selection_file $model_file \
        --runs $repetitions \
        --output_dir_prefix_tag ${tag}final-reverse \
        --max_concurrent $max_concurrent \
        --datasets F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS \
        --language $language \
        --reverse_scale True"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}final-reverse_${model}.log; echo '=== Finished: final-reverse ===' " Enter

    # With authoritarian system prompt
    cmd="uv run main.py \
        --model_selection_file $model_file \
        --runs $repetitions \
        --output_dir_prefix_tag ${tag}final-authsys \
        --max_concurrent $max_concurrent \
        --datasets F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS \
        --language $language \
        --auth_sys_prompt True"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}final-authsys_${model}.log; echo '=== Finished: final-authsys ===' " Enter

    # ========================================================================
    # BEHAVIORAL EXPERIMENTS
    # Comment out any section to skip it
    # ========================================================================

    # Vignette RWA3D
    cmd="uv run main.py \
        --model_selection_file $model_file \
        --runs $repetitions \
        --output_dir_prefix_tag ${tag}final-vignette-rwa3d \
        --max_concurrent $max_concurrent \
        --datasets VignetteRWA3D \
        --language $language"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}final-vignette-rwa3d_${model}.log; echo '=== Finished: vignette-rwa3d ===' " Enter

    # Vignette RWA3D with authoritarian system prompt
    cmd="uv run main.py \
        --model_selection_file $model_file \
        --runs $repetitions \
        --output_dir_prefix_tag ${tag}final-authsys-vignette-rwa3d \
        --max_concurrent $max_concurrent \
        --datasets VignetteRWA3D \
        --language $language \
        --auth_sys_prompt True"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}final-authsys-vignette-rwa3d_${model}.log; echo '=== Finished: authsys-vignette-rwa3d ===' " Enter

    # ========================================================================
    # ISSUEBENCH (realistic user prompts) EXPERIMENTS
    # Comment out any section to skip it
    # ========================================================================

    # Generate responses
    cmd="uv run issuebench.py gen \
        --model-selection-file $model_file \
        --tag ${tag}issuebench_gen_jan13_${language} \
        --num-prompts 1000 \
        --max-concurrent $max_concurrent \
        --language $language"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}issuebench_gen_jan13_${language}_${model}.log; echo '=== Finished: issuebench gen ===' " Enter

    # Judge generated responses
    cmd="uv run issuebench.py judge \
        --model-selection-file $model_file \
        --input-dir resources/output/${tag}issuebench_gen_jan13_${language}-$language-1_0-1-42 \
        --tag ${tag}issuebench_judge_jan14_${language} \
        --max-concurrent $max_concurrent \
        --language en"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}issuebench_judge_jan14_${language}_${model}.log; echo '=== Finished: issuebench judge ===' " Enter

    # Generate responses with authoritarian system prompt
    cmd="uv run issuebench.py gen \
        --model-selection-file $model_file \
        --tag ${tag}issuebench_gen_jan24_auth_${language} \
        --num-prompts 1000 \
        --max-concurrent $max_concurrent \
        --language $language \
        --auth-sys-prompt"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}issuebench_gen_jan24_auth_${language}_${model}.log; echo '=== Finished: issuebench gen auth ===' " Enter

    # Judge authoritarian generations
    cmd="uv run issuebench.py judge \
        --model-selection-file $model_file \
        --input-dir resources/output/${tag}issuebench_gen_jan24_auth_${language}-$language-1_0-1-42 \
        --tag ${tag}issuebench_judge_jan14_${language} \
        --max-concurrent $max_concurrent \
        --language en"
    tmux send-keys -t "$sessionName:$model" "$cmd | tee logs/${tag}issuebench_judge_jan14_${language}_${model}.log; echo '=== Finished: issuebench judge auth ===' " Enter

    echo "All commands queued for model: $model"
done

echo ""
echo "============================================"
echo "All experiments queued successfully!"
echo "To attach to the session: tmux attach -t $sessionName"
echo "To list windows: tmux list-windows -t $sessionName"
echo "To switch windows: Ctrl+b then window number"
echo "To detach: Ctrl+b then d"
echo "============================================"