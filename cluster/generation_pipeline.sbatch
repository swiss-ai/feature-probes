#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --job-name=generation_pipeline
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail


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
export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
if [[ -f ~/keys/.anthropic_key ]]; then
    export ANTHROPIC_API_KEY="$(cat ~/keys/.anthropic_key)"
fi

# change this to your repo path
cd /workspace/hallucination_probes/

git pull
git checkout  compare_models
git pull


uv pip install --no-deps  . --system
uv pip install transformers==4.56.0 --system

python generation_pipeline/generation.py


# srun --account=large-sc-2 -p normal  --time=01:00:00 --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml --pty bash
