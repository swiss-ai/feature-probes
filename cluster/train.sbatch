#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=feature-probes-train
#SBATCH --output=/iopsstor/scratch/cscs/%u/feature-probes/log/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/feature-probes/log/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail

mkdir -p /iopsstor/scratch/cscs/$USER/feature-probes/log

# Read keys from home — mounted via env.toml
export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"

# Optional write token (only needed if upload_to_hf=true)
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

# --environment must be passed to srun, not as #SBATCH directive (CSCS recommendation)
srun --environment=/users/$USER/feature-probes/cluster/env.toml \
    python scripts/train_probe.py \
        model=llama \
        training=no_lora \
        dataset=our_long_form
