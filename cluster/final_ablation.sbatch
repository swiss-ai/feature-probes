#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=6:00:00
#SBATCH --job-name=probe-seed-ablation
#SBATCH --output=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.err
#SBATCH --array=0-47
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

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

# 2 models x 4 variants (norm x lora) x 2 probe dtypes x 4 layers x 3 seeds = 192 total
MODELS=(apertus llama)
VARIANTS=(no_lora_no_ln no_lora_ln lora_no_ln lora_ln)
PROBE_DTYPES=(float32 bfloat16)
EVEN_LAYERS=(10 20 26 30)
SEEDS=(42 43 44)

# Build flat list: variants -> models -> dtypes -> layers -> seeds
# Variant outermost ensures all no-LoRA tasks precede all LoRA tasks,
# giving a clean job-range split when submitting with different time limits.
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

TOTAL=${#RUN_LAYERS[@]}   # 4 variants * 2 models * 2 dtypes * 4 layers * 3 seeds = 192
# Submit in two batches to use different wall-clock limits:
#   No-LoRA (variants 0-1, jobs 0-23):  sbatch --array=0-23  --time=2:00:00 final_ablation.sbatch
#   LoRA    (variants 2-3, jobs 24-47): sbatch --array=24-47 --time=6:00:00 final_ablation.sbatch
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

  # Short dtype suffix for probe ID
  case "$PROBE_DTYPE" in
    float32)  DTYPE_SHORT="fp32" ;;
    bfloat16) DTYPE_SHORT="bf16" ;;
    *)        DTYPE_SHORT="$PROBE_DTYPE" ;;
  esac

  PROBE_ID="${MODEL}_seed_ablation_${VARIANT}_${DTYPE_SHORT}_layer${LAYER}_seed${SEED}"

  echo "Run $idx/$TOTAL -> Model: $MODEL | Variant: $VARIANT | dtype: $PROBE_DTYPE | Layer: $LAYER | Seed: $SEED | GPU: $local_gpu"

  CUDA_VISIBLE_DEVICES=$local_gpu echo probe/train.py \
    model=$MODEL \
    training=no_lora \
    dataset=our_long_form \
    probe_config.layer=$LAYER \
    model_dtype=bfloat16 \
    probe_config.probe_dtype=$PROBE_DTYPE \
    probe_config.normalize_before_head=$NORM \
    probe_config.lora_layers=$LORA_LAYERS \
    probe_head_lr=$LR \
    seed=$SEED \
    "probe_config.probe_id=${PROBE_ID}" \
    "wandb_tags=[final_ablation,$MODEL,$VARIANT,probe$DTYPE_SHORT]" &
done

wait
echo "All tasks for Array ID $SLURM_ARRAY_TASK_ID completed."