#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --job-name=apertus-lora-comparison
#SBATCH --output=log/%x-%j_%a.out
#SBATCH --error=log/%x-%j_%a.err
#SBATCH --array=0-3
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail


export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

# private keys
export HF_TOKEN="$(cat ~/keys/.hf_token)"
if [[ -f ~/keys/.wandb_key ]]; then
    export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
fi

# don't require to have hf_write_token for read-only operations, if this is missing, field upload_to_hf should be false
if [[ -f ~/keys/.hf_token_write ]]; then
    export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
fi

# change this to your repo path
cd /workspace/hallucination_probes/

git pull
git checkout  transferability
git pull



export CUDA_VISIBLE_DEVICES=0
uv pip install --no-deps  . --system

# multiarray job setup - 4 jobs
MODELS=("apertus" "apertus" "llama" "llama")
TRAINING=("no_lora" "lora" "no_lora" "lora")

CURRENT_MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CURRENT_TRAINING=${TRAINING[$SLURM_ARRAY_TASK_ID]}

echo "Running Hydra job: model=$CURRENT_MODEL training=$CURRENT_TRAINING"

# Instead of passing a file path, we override the Hydra config modules
python probe/train.py model=$CURRENT_MODEL training=$CURRENT_TRAINING dataset=our_long_form

# srun --account=large-sc-2 -p normal  --time=01:00:00 --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml --pty bash
