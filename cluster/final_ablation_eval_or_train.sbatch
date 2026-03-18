#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=6:00:00
#SBATCH --job-name=probe-seed-ablation-eval-or-train
#SBATCH --output=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.err
#SBATCH --array=0-47
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

# For each task, probe/run_probe.py checks whether the probe checkpoint exists:
#   - probe_head.bin FOUND   → evaluate only, log metrics to W&B
#   - probe_head.bin MISSING → run full training (probe/train.py via Hydra)
#
# Array mapping (variants-outermost, same order as final_ablation.sbatch):
#   Jobs  0-23  → no-LoRA variants (no_lora_no_ln, no_lora_ln)
#   Jobs 24-47  → LoRA variants    (lora_no_ln, lora_ln)
# Each array job runs 4 tasks in parallel (one per GPU).

set -euo pipefail

export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

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

# Fixed hyperparameters
LR="3e-4"

# Grid — identical to final_ablation.sbatch
MODELS=(apertus llama)
VARIANTS=(no_lora_no_ln no_lora_ln lora_no_ln lora_ln)
PROBE_DTYPES=(float32 bfloat16)
EVEN_LAYERS=(10 20 26 30)
SEEDS=(42 43 44)

# Build flat list: variants -> models -> dtypes -> layers -> seeds
RUN_MODELS=()
RUN_VARIANTS=()
RUN_DTYPES=()
RUN_LAYERS=()
RUN_SEEDS=()

for v in "${VARIANTS[@]}"; do
  for m in "${MODELS[@]}"; do
    for d in "${PROBE_DTYPES[@]}"; do
      for l in "${EVEN_LAYERS[@]}"; do
        for s in "${SEEDS[@]}"; do
          RUN_MODELS+=("$m")
          RUN_VARIANTS+=("$v")
          RUN_DTYPES+=("$d")
          RUN_LAYERS+=("$l")
          RUN_SEEDS+=("$s")
        done
      done
    done
  done
done

TOTAL=${#RUN_LAYERS[@]}   # 192
TASKS_PER_JOB=4
START=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB))

for local_gpu in 0 1 2 3; do
  idx=$((START + local_gpu))

  if [[ $idx -ge $TOTAL ]]; then
    break
  fi

  MODEL="${RUN_MODELS[$idx]}"
  VARIANT="${RUN_VARIANTS[$idx]}"
  PROBE_DTYPE="${RUN_DTYPES[$idx]}"
  LAYER="${RUN_LAYERS[$idx]}"
  SEED="${RUN_SEEDS[$idx]}"

  # Decode norm flag
  case "$VARIANT" in
    no_lora_ln|lora_ln) NORM="layernorm" ;;
    *)                  NORM="none" ;;
  esac

  # Decode lora flag
  if [[ "$VARIANT" == lora_* ]]; then
    LORA_LAYERS="all"
  else
    LORA_LAYERS="none"
  fi

  # Short dtype suffix for probe ID and tags
  case "$PROBE_DTYPE" in
    float32)  DTYPE_SHORT="fp32" ;;
    bfloat16) DTYPE_SHORT="bf16" ;;
    *)        DTYPE_SHORT="$PROBE_DTYPE" ;;
  esac

  PROBE_ID="${MODEL}_seed_ablation_${VARIANT}_${DTYPE_SHORT}_layer${LAYER}_seed${SEED}"

  echo "Run $idx/$TOTAL -> Model: $MODEL | Variant: $VARIANT | dtype: $PROBE_DTYPE | Layer: $LAYER | Seed: $SEED | GPU: $local_gpu"

  CUDA_VISIBLE_DEVICES=$local_gpu python -u probe/run_probe.py \
    --probe-id   "${PROBE_ID}" \
    --model      "${MODEL}" \
    --layer      "${LAYER}" \
    --probe-dtype "${PROBE_DTYPE}" \
    --norm        "${NORM}" \
    --lora-layers "${LORA_LAYERS}" \
    --variant     "${VARIANT}" \
    --model-dtype bfloat16 \
    --seed        "${SEED}" \
    --lr          "${LR}" \
    --wandb-tags  "final_ablation,${MODEL},${VARIANT},probe${DTYPE_SHORT}" &
done

wait
echo "All tasks for Array ID $SLURM_ARRAY_TASK_ID completed."
