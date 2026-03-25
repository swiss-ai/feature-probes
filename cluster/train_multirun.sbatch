#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --job-name=feature-probes-sweep
#SBATCH --output=/iopsstor/scratch/cscs/%u/feature-probes/log/%x-%j_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/feature-probes/log/%x-%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --array=0-5   # adjust to ceil(total_runs / 4) - 1

set -euo pipefail

mkdir -p /iopsstor/scratch/cscs/$USER/feature-probes/log

export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

# ── Sweep definition — edit this block to change the search space
MODELS=(llama llama apertus apertus llama llama apertus apertus)
TRAINING=(no_lora lora no_lora lora no_lora lora no_lora lora)
DATASETS=(our_long_form our_long_form our_long_form our_long_form triviaqa triviaqa triviaqa triviaqa)
# ──────────────────────────────────────────────────────────────────

TASKS_PER_JOB=4
START=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
TOTAL=${#MODELS[@]}

for local_gpu in 0 1 2 3; do
    idx=$(( START + local_gpu ))
    [[ $idx -ge $TOTAL ]] && break

    echo "GPU $local_gpu → model=${MODELS[$idx]} training=${TRAINING[$idx]} dataset=${DATASETS[$idx]}"

    srun --environment=/users/$USER/feature-probes/cluster/env.toml \
        env CUDA_VISIBLE_DEVICES=$local_gpu \
        python scripts/train_probe.py \
            model=${MODELS[$idx]} \
            training=${TRAINING[$idx]} \
            dataset=${DATASETS[$idx]} &
done

wait
echo "Array task $SLURM_ARRAY_TASK_ID done."
