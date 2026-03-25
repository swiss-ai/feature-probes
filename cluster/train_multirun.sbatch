#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --job-name=feature-probes-sweep
#SBATCH --output=/iopsstor/scratch/cscs/%u/feature-probes/cluster/log/%x-%j_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/feature-probes/cluster/log/%x-%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --array=0   # ceil(total_runs / 4) - 1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/feature-probes

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Array task: $SLURM_ARRAY_TASK_ID"

export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

# ── Sweep definition ────────────────────────────────────────────────────────
MODELS=(llama llama apertus apertus)
TRAINING=(no_lora lora no_lora lora)
DATASETS=(our_long_form our_long_form our_long_form our_long_form)
# ───────────────────────────────────────────────────────────────────────────

TASKS_PER_JOB=4
START=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
TOTAL=${#MODELS[@]}

# Build a small wrapper script that runs all 4 tasks inside the container
TMPSCRIPT=$REPO/cluster/log/sweep_task_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh

cat > $TMPSCRIPT << 'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail
SCRIPT_EOF

for local_gpu in 0 1 2 3; do
    idx=$(( START + local_gpu ))
    [[ $idx -ge $TOTAL ]] && break

    echo "GPU $local_gpu → model=${MODELS[$idx]} training=${TRAINING[$idx]} dataset=${DATASETS[$idx]}"

    cat >> $TMPSCRIPT << EOF
CUDA_VISIBLE_DEVICES=$local_gpu python scripts/train_probe.py \\
    model=${MODELS[$idx]} \\
    training=${TRAINING[$idx]} \\
    dataset=${DATASETS[$idx]} &
EOF
done

echo "wait" >> $TMPSCRIPT
echo 'echo "All tasks done."' >> $TMPSCRIPT
chmod +x $TMPSCRIPT

srun --environment=$REPO/cluster/env.toml bash $TMPSCRIPT

rm -f $TMPSCRIPT
echo "Array task $SLURM_ARRAY_TASK_ID done."

