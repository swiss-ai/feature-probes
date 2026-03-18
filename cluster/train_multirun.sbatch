#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=3:00:00
#SBATCH --job-name=hallucination-probe-ablation
#SBATCH --output=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.err
#SBATCH --array=0-23
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

if [[ -f ~/keys/.hf_token_write ]]; then
    export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
fi

cd /workspace/hallucination_probes/

git pull
git checkout improve_stability
git pull

uv pip install --no-deps . --system
uv pip install hydra-core --system

mkdir -p log

# Define search space (Apertus only, two probe variants)
MODEL="apertus"
EVEN_LAYERS=(10 20 26 30)
SEEDS=(42 43 44)
LRS=(1e-3 3e-4 1e-4 3e-5)
# VARIANTS=(ln_fp32 fp32_only ln_bf16 bf16_only)
VARIANTS=(ln_fp32 fp32_only)

# Build a flat list of all combinations
RUN_LAYERS=()
RUN_SEEDS=()
RUN_LRS=()
RUN_VARIANTS=()

for v in "${VARIANTS[@]}"; do
  for l in "${EVEN_LAYERS[@]}"; do
    for s in "${SEEDS[@]}"; do
      for lr in "${LRS[@]}"; do
        RUN_VARIANTS+=("$v")
        RUN_LAYERS+=("$l")
        RUN_SEEDS+=("$s")
        RUN_LRS+=("$lr")
      done
    done
  done
done

TOTAL=${#RUN_LAYERS[@]}          # Will be 4 * 4 * 3 * 4 = 192 runs total
TASKS_PER_JOB=4                  # 4 GPUs per SLURM job
START=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB))

# Launch exactly 4 tasks in the background, mapping to GPU 0, 1, 2, 3
for local_gpu in 0 1 2 3; do
  idx=$((START + local_gpu))
  
  # Safety check just in case TOTAL isn't perfectly divisible by 4
  if [[ $idx -ge $TOTAL ]]; then
    break
  fi

  VARIANT="${RUN_VARIANTS[$idx]}"
  LAYER="${RUN_LAYERS[$idx]}"
  SEED="${RUN_SEEDS[$idx]}"
  LR="${RUN_LRS[$idx]}"

  if [[ "$VARIANT" == "ln_fp32" || "$VARIANT" == "ln_bf16" ]]; then
    NORM="layernorm"
  else
    NORM="none"
  fi

  if [[ "$VARIANT" == "ln_bf16" || "$VARIANT" == "bf16_only" ]]; then
    PROBE_DTYPE="bfloat16"
  else
    PROBE_DTYPE="float32"
  fi

  LR_TAG="${LR//./p}"
  LR_TAG="${LR_TAG//-/m}"
  PROBE_ID="apertus_no_lora_${VARIANT}_layer${LAYER}_seed${SEED}_lr${LR_TAG}"

  echo "Launching Run $idx/$TOTAL -> Model: $MODEL | Variant: $VARIANT | Layer: $LAYER | Seed: $SEED | LR: $LR | GPU: $local_gpu"

  # Override probe_id on the fly so runs don't overwrite each other in output dirs / wandb
  CUDA_VISIBLE_DEVICES=$local_gpu python probe/train.py \
    model=$MODEL \
    training=no_lora \
    dataset=our_long_form \
    probe_config.layer=$LAYER \
    model_dtype=bfloat16 \
    probe_config.probe_dtype=$PROBE_DTYPE \
    probe_config.normalize_before_head=$NORM \
    probe_head_lr=$LR \
    seed=$SEED \
    "probe_config.probe_id=${PROBE_ID}" &
done

# Wait for all 4 background tasks on this node to finish
wait
echo "All 4 runs for Array ID $SLURM_ARRAY_TASK_ID completed!"
