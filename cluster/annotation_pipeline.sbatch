#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --job-name=annotation_pipeline
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail

export UV_CACHE_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/uv_cache
export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export LOCAL_RESULTS_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/annotation_pipeline/results
export SAFETYTOOLING_CACHE_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/safetytooling_cache
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

# file sourced from ~repo/annotation_pipeline/entity_annotation.prompt
export ENTITY_ANNOTATION_PROMPT_TEMPLATE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/annotation_pipeline/entity_annotation.prompt

# private keys
export HF_TOKEN="$(cat ~/keys/.hf_token)"
if [[ -f ~/keys/.wandb_key ]]; then
    export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
fi
if [[ -f ~/keys/.hf_token_write ]]; then
    export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
fi
if [[ -f ~/keys/.anthropic_key ]]; then
    export ANTHROPIC_API_KEY="$(cat ~/keys/.anthropic_key)"
fi
if [[ -f ~/keys/.openai_key ]]; then
    export OPENAI_API_KEY="$(cat ~/keys/.openai_key)"
fi

# change this to your repo path
cd /workspace/hallucination_probes/

git pull
git checkout  compare_models
git pull



export CUDA_VISIBLE_DEVICES=0
rm -rf /usr/lib/python3.*/EXTERNALLY-MANAGED


uv pip install --no-deps  . --system

uv pip install "git+https://github.com/safety-research/safety-tooling@100cf43b4935465e6bb5428011734fa6c622d5a7" --system

# newest version spans some errors
uv pip install  opencv-python==4.8.0.74 --system
uv pip install --upgrade pyarrow --system

uv pip install --upgrade openai --system --no-cache-dir


python annotation_pipeline/run.py 

# srun --account=large-sc-2 -p normal  --time=01:00:00 --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml --pty bash
