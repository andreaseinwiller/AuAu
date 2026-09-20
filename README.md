![AuAuBenchmark](figs/fig01-overview.png)
# AuAu: A Benchmark for Auditing Authoritarian Alignment in Large Language Models

You can use the AuAu Benchmark to audit authoritarian alignment in LLMs.
In the following, we explain:
1. How to setup the benchmark.
2. How to use the benchmark with your own model.
3. Where to download the data.
4. How to replicate the LLM inference and reproduce the results from the paper (with frozen artefacts).

Further resources:
+ Finetuned Gemma4 model: https://huggingface.co/mklabunde/gemma4-31b-autho-detector
+ Annotated ground truths: https://huggingface.co/datasets/mklabunde/auditing-authoritarianism-annotations-reduction and https://huggingface.co/datasets/mklabunde/auditing-authoritarianism-annotations-issuebench
+ All outputs: https://zenodo.org/records/20786456

## Setup

1. Clone this repository
```bash
git clone git@github.com:andreaseinwiller/AuAu.git
cd AuAu
```

2. Install dependencies.
If you do not have *uv* installed, install it from [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).
```bash
uv sync
```

3. Create a `.env` file in the project root and configure the following variables:

```env
# Required (to run our model selection)
OR_API_KEY="<your_openrouter_api_key>"     # OpenRouter API key.
HF_TOKEN="<your_huggingface_token>"        # HuggingFace (READ) token.

# Optional
ALT_BASE_URL="<your_base_url>"             # Alternative (OpenAI compatible API) LLM inference endpoint.
ALT_API_KEY="<your_api_key>"               # API key for alternative endpoint.
```

## Using the Benchmark

To run the benchmark with a new model, do the following:
1. Follow the setup steps above.
2. Add a `JSON` file to `resources/input/models`. Note that you can have multiple models in one file. It should have the following structure:
    ```json
    [
        {
            "setup": "OpenRouter",
            "name": "myModelName",
            "format": "text-only",
            "context_window": 163840
        },
    ]
    ```

    **Fields**:
    - `setup` Specifies how LLM inference will be performed. Supported values:
        - `"OpenRouter"` - Use models via https://openrouter.ai/.
        - `"vLLM"` - Run Hugging Face models locally via vLLM.
        - `"Alternative"` - Use custom OpenAI compatible API endpoint.
    - `name` Name (or id) of the model.
    - `format` Defines the expected input format:
        - `"text-only"` - content
        - `"multimodal"` - [{"type": "text", "text": content}]
    - `context_window` The context window of the model.

<br/>

3. Run `scripts/run_all.sh myAddedFileName` where `myAddedFileName` is replaced by the name of the model file (without .json suffix) you added in the previous step. This will run all parts of the benchmark. Feel free to comment out blocks of the scripts to run only specific parts of the benchmark. **Note: running the experiments via OpenRouter incurs significant costs. You may want to reduce the number of repetitions for testing.**
    ```bash
    ./scripts/run_all.sh myAddedFileName
    ```
4. If you have previously [downloaded our results](#downloading-data), your new results will be in the same directories.
5. To analyze generated outputs or reproduce the paper from our frozen outputs, continue with [Reproducing the paper artifacts](#5-reproduce-paper-artifacts-from-frozen-outputs). 

## Downloading Data

All of our model responses are publicly available in this zenodo repository:
* https://zenodo.org/records/20786456

All other content (including prompts and datasets) is available in this GitHub repository.

For reproducing the paper results, use the following downloader and extractor. 
Note that building the docker image will take some time.

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo service docker start
sudo usermod -aG docker $USER
# Restart shell session
chmod +x run.sh
./run.sh build download extract
```

The extraction step restores the frozen model and judge outputs under `resources/output/` and the released tidy analysis files under `eval/data/tidy/`.
To create the tidy datasets `construct_scores.csv` und `vignette_scores.csv` for results of a new model, simply execute both cells in `eval/tidy_df_preprocessing.ipynb`.

## LLM Inference and Reproducing our Results

Request access for this model: https://huggingface.co/utter-project/EuroLLM-9B-Instruct.


```bash
# GPU selection
watch -n0.1 nvidia-smi
```

- U may want to change `CUDA_VISIBLE_DEVICES=0` and `--gpus 1` to something like `CUDA_VISIBLE_DEVICES=4,5` and `--gpus 2` depending on your available GPU resources.
- Note that you can replace `final_complete.json` with `...local_batch...`, `...openrouter_batch...`, or your own model configuration JSON file (under `resources/input/models/`).


### 1. Construct experiments (closed and open)

- resources/output/final-en-1_0-10-42/
- resources/output/final-reverse-en-1_0-10-42/
- resources/output/final-authsys-en-1_0-10-42/

```bash
# Construct run (final-en-1_0-10-42): DefaultSys, NonReverse, en
CUDA_VISIBLE_DEVICES=0 uv run main.py \
    --seed 42 \
    --gpus 1 \
    --temperature 1.0 \
    --runs 10 \
    --language en \
    --model_selection_file final_complete.json \
    --judge_selection_file judges_or_alt.json \
    --datasets F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS \
    --output_dir_prefix_tag final \
    --reverse_scale False \
    --auth_sys_prompt False \
    --max_concurrent 20

# Construct run (final-reverse-en-1_0-10-42): DefaultSys, Reverse, en
CUDA_VISIBLE_DEVICES=0 uv run main.py \
    --seed 42 \
    --gpus 1 \
    --temperature 1.0 \
    --runs 10 \
    --language en \
    --model_selection_file final_complete.json \
    --judge_selection_file judges_or_alt.json \
    --datasets F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS \
    --output_dir_prefix_tag final-reverse \
    --reverse_scale True \
    --auth_sys_prompt False \
    --max_concurrent 20

# Construct run (final-authsys-en-1_0-10-42): AuthSys, NonReverse, en
CUDA_VISIBLE_DEVICES=0 uv run main.py \
    --seed 42 \
    --gpus 1 \
    --temperature 1.0 \
    --runs 10 \
    --language en \
    --model_selection_file final_complete.json \
    --judge_selection_file judges_or_alt.json \
    --datasets F RWA RWA3D AA A D CSM ACT VSA SDO7 DW BDW CW PISD ASC PI APC BFI10 KSA3 LAS \
    --output_dir_prefix_tag final-authsys \
    --reverse_scale False \
    --auth_sys_prompt True \
    --max_concurrent 20
```

The open response generations need the ensemble judge, which requires running more judges:
```
uv run rejudge_open_responses.py  # use the command line arguments to point towards the outputs of the previous steps as input (e.g., 'final-en')

# Examples, which also need to be adapted to final-reverse and final-authsys
uv run rejudge_open_responses.py --model-selection-file gemma4-or.json --input-dir resources/output/final-en-1_0-10-42 --output-dir resources/output/final-en-1_0-10-42-gemma4-rejudge --max-concurrent 16

uv run rejudge_open_responses.py --model-selection-file dsv4flash.json --input-dir resources/output/final-en-1_0-10-42 --output-dir resources/output/final-en-1_0-10-42-dsv4flash-rejudge --max-concurrent 16

uv run rejudge_open_responses.py --model-selection-file falconh1r7b.json --input-dir resources/output/final-en-1_0-10-42 --output-dir resources/output/final-en-1_0-10-42-falconh1r7b-rejudge --max-concurrent 16
```
Since the Falcon model does not use correct reasoning parsing out of the box, extract the reasoning afterwards with this:
```
uv run scripts/reprocess_falconh1r7b_rejudge.py --root resources/output/final-en-1_0-10-42-falconh1r7b-rejudge/
```

Afterwards, create the ensemble construct csv:
```
uv run eval/create_construct_tidydf.py
```
Now you can run open response adjusted bootstrapping.

### 2. Vignette experiments

- resources/output/final-vignette-rwa3d-en-1_0-10-42/
- resources/output/final-authsys-vignette-rwa3d-en-1_0-10-42/


```bash
# Vignette (RWA3D) run (final-vignette-rwa3d-en-1_0-10-42): DefaultSys, NonReverse, en
CUDA_VISIBLE_DEVICES=0 uv run main.py \
    --seed 42 \
    --gpus 1 \
    --temperature 1.0 \
    --runs 10 \
    --language en \
    --model_selection_file final_complete.json \
    --judge_selection_file judges_or_alt.json \
    --datasets VignetteRWA3D \
    --output_dir_prefix_tag final-vignette-rwa3d \
    --reverse_scale False \
    --auth_sys_prompt False \
    --max_concurrent 20

# Vignette (RWA3D) run (final-authsys-vignette-rwa3d-en-1_0-10-42): AuthSys, NonReverse, en
CUDA_VISIBLE_DEVICES=0 uv run main.py \
    --seed 42 \
    --gpus 1 \
    --temperature 1.0 \
    --runs 10 \
    --language en \
    --model_selection_file final_complete.json \
    --judge_selection_file judges_or_alt.json \
    --datasets VignetteRWA3D \
    --output_dir_prefix_tag final-authsys-vignette-rwa3d \
    --reverse_scale False \
    --auth_sys_prompt True \
    --max_concurrent 20
```

### 3. Issuebench experiments

You can use the following scripts.
Each script will start parallel jobs for the specified models.
This means you could run out of local GPU resources or get rate limited when using OpenRouter.
We thus recommend running only a few models in parallel.

```bash
./scripts/generate_issuebench.sh
./scripts/generate_issuebench_auth.sh
./scripts/judge_issuebench.sh
./scripts/judge_issuebench_auth.sh
```

Since this repo has vllm pinned to an old version to have variable output for the same prompt, the finetuned Gemma4 judge must be run in a different environment; it requires a newer vllm version for inference.
Before judging with this model, create a different venv with vllm>=0.21 and then run the vllm server with a command like this (assuming the model is saved in `merged_exports/...`):
```
CUDA_VISIBLE_DEVICES=4,5 uv run vllm serve merged_exports/2026-05-22_103509_final_fold-0/   --max-model-len 64000   --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 0, "audio": 0}' --max-num-batched-tokens 4096 -tp 2
```
Tensor Parallel is only necessary for long context, you can use data paralell (`-dp`) instead to get more throughput and rerun the missing judgements later with the different setup. Running the judgement script multiple times will only run the inputs with missing outputs.

After you started the server, patch your `ALTERNATIVE_BASE_URL` in the `.env` to the URL of the vllm server (typically 127.0.0.1). The API key does not matter for this.
Only then can you judge with the `gemma4-sft.json` model file.

### 4. Bootstrap and score adjustment

Create the bootstrap summaries required for the main results table.

```bash
./scripts/run_bootstrap.sh
```

### 5. Reproduce paper artifacts from frozen outputs

Run notebook steps with the repository root as the working directory.

| Paper artifact | Code |
| --- | --- |
| Figure 1 | Static overview figure: `figs/fig01-overview.png` |
| Table 1 and Table 20 | `eval/main_res_table_gen.ipynb` |
| Figure 2 and Figure 5; Tables 29--31 | `eval/authsys_ablation.ipynb` |
| Figure 3 | `uv run create_issuebench_authsys_prompt_figures.py` |
| Figure 4 | `uv run issuebench_high_risk_vs_low_risk_figure.py` |
| Tables 21--24 | Run `eval/cfa_preprocessing.ipynb`, then `R script eval/reliability_psych.r`, `R script eval/cfa_sem.r`, and finally `eval/reliability_measures.ipynb` |
| Tables 25--28 | `eval/stability_over_reps.ipynb` |
| Table 34 | `reduction_ensemble_eval.ipynb` |
| Table 35 | `eval/issuebench_descriptive_stats.ipynb` (final resolved-label counts) |
| Table 36 | `uv run create_gemma4_eval_table.py --output eval/data/gemma4_eval_table.tex` |
| Table 37 | Produced by the separate Gemma 4 finetuning repository linked below |

Tables 2--19 are related to psychoemtric instruments and Tables 32--33 document the evaluated and judge models; these are static tables rather than derived analysis outputs.

Additional numerical analyses reported in the text can be reproduced with:

```bash
uv run regression_analysis.py
# Reverse-order robustness: run all cells in reverse_order_effect.ipynb

uv run annotation/issuebench/check_agreement.py
uv run annotation/open_response/check_agreement.py
```

### 6. Gemma4 finetuning

TODO: Add repo link after review.

## License

This project is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the material for any purpose, provided appropriate credit is given and any changes are indicated.

## Citation

TODO: Add bibtex after review.