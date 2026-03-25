"""
probe/run_probe.py

Checks whether a trained probe checkpoint exists at
  $LOCAL_PROBES_DIR/{probe_id}/probe_head.bin

  - FOUND   → run evaluation via probe/evaluate.py internals and log metrics
              to W&B so the analysis notebook picks them up automatically.
  - NOT FOUND → delegate to probe/train.py via Hydra (full training + eval).

This is the recovery path for SLURM jobs that were cancelled before the W&B
offline sync completed, but after the probe was saved to disk.

Local test example
------------------
  python probe/run_probe.py \\
      --probe-id apertus_seed_ablation_no_lora_no_ln_fp32_layer10_seed42 \\
      --model apertus \\
      --layer 10 \\
      --probe-dtype float32 \\
      --norm none \\
      --lora-layers none \\
      --variant no_lora_no_ln \\
      --model-dtype bfloat16 \\
      --seed 42 \\
      --wandb-tags "final_ablation,apertus,no_lora_no_ln,probefp32"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Eval datasets are the same as in configs/dataset/our_long_form.yaml
EVAL_DATASETS = [
    {
        "dataset_id": "longfact_test_llama",
        "hf_repo": "tymciurymciu/longfact-test-split",
        "subset": "Meta_Llama_3.1_8B_Instruct",
        "split": "test",
        "max_length": 1536,
        "pos_weight": 10.0,
        "neg_weight": 10.0,
        "default_ignore": False,
        "shuffle": False,
    },
    {
        "dataset_id": "longfact_test_apertus",
        "hf_repo": "tymciurymciu/longfact-test-split",
        "subset": "Apertus_8B_Instruct_2509",
        "split": "test",
        "max_length": 1536,
        "pos_weight": 10.0,
        "neg_weight": 10.0,
        "default_ignore": False,
        "shuffle": False,
    },
]

MODEL_NAMES = {
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a saved probe or fall back to training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe-id", required=True, help="Probe ID (used as run name and directory name)")
    p.add_argument("--model", required=True, choices=list(MODEL_NAMES), help="Model family key")
    p.add_argument("--layer", type=int, required=True, help="Transformer layer index")
    p.add_argument("--probe-dtype", required=True, help="Probe head dtype: float32 | bfloat16")
    p.add_argument("--norm", required=True, help="Pre-head normalisation: none | layernorm")
    p.add_argument("--lora-layers", required=True, help="LoRA target: none | all")
    p.add_argument("--variant", required=True, help="Variant tag (for wandb tags, e.g. no_lora_no_ln)")
    p.add_argument("--model-dtype", default="bfloat16", help="Base LLM dtype")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", default="3e-4", help="probe_head_lr for the training fallback")
    p.add_argument("--wandb-project", default="hallucination-probes")
    p.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation path
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    """Load the saved probe from disk, evaluate on both test sets, log to W&B."""
    from dotenv import load_dotenv
    load_dotenv()

    import wandb
    from torch.utils.data import DataLoader

    from feature_probes.config import ProbeConfig
    from feature_probes.data.dataset import (
        TokenizedProbingDatasetConfig,
        create_probing_dataset,
        tokenized_probing_collate_fn,
    )
    from feature_probes.evaluation.evaluate import evaluate_probe
    from feature_probes.probes.value_head_probe import setup_probe
    from transformers import PreTrainedModel
    from feature_probes.utils.model_utils import load_model_and_tokenizer, resolve_torch_dtype

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    # W&B run — same name as the original training run so the notebook's
    # _extract_selected_id() picks it up via run.name.
    # The finished eval run will supersede the crashed/running training run
    # thanks to the dedup logic in Cell 5 (finished > running > queued > ...).
    wandb.init(
        project=args.wandb_project,
        name=args.probe_id,
        tags=tags or None,
        config={
            # Keep probe_config nested so _collect_text_candidates finds probe_id
            "probe_config": {"probe_id": args.probe_id},
            "eval_only": True,
        },
    )

    try:
        probe_config = ProbeConfig(
            probe_id=args.probe_id,
            model_name=MODEL_NAMES[args.model],
            layer=args.layer,
            probe_dtype=args.probe_dtype,
            normalize_before_head=args.norm,
            lora_layers=args.lora_layers,
            load_from="disk",
        )

        print(f"[run_probe] Loading model: {probe_config.model_name}")
        model, tokenizer = load_model_and_tokenizer(
            probe_config.model_name,
            torch_dtype=resolve_torch_dtype(args.model_dtype),
        )

        print(f"[run_probe] Loading probe from disk: {probe_config.probe_path}")
        model, probe = setup_probe(cast(PreTrainedModel, model), probe_config, seed=args.seed)

        all_metrics: dict = {}
        for ds_cfg_dict in EVAL_DATASETS:
            ds_cfg = TokenizedProbingDatasetConfig(**ds_cfg_dict)
            print(f"\n[run_probe] Evaluating on {ds_cfg.dataset_id} ...")
            dataset = create_probing_dataset(ds_cfg, tokenizer)
            print(f"  Dataset size: {len(dataset)} samples")

            dataloader = DataLoader(
                dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=tokenized_probing_collate_fn,
                shuffle=False,
            )

            metrics = evaluate_probe(
                probe,
                dataloader,
                threshold=probe_config.threshold,
                metric_key_prefix=ds_cfg.dataset_id,
                verbose=True,
            )
            all_metrics.update(metrics)

        # Log with "train/" prefix to match what HF Trainer's WandbCallback emits
        # (rewrite_logs() prepends "train/" to non-eval keys).
        # That lets the notebook's summary.get("train/longfact_test_*/all_auc") work
        # without any changes.
        wandb.log({f"train/{k}": v for k, v in all_metrics.items()})
        print("[run_probe] Evaluation complete.")
    finally:
        wandb.finish()


# ---------------------------------------------------------------------------
# Training fallback path
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    """Delegate to probe/train.py via Hydra subprocess."""
    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "probe" / "train.py"),
        f"model={args.model}",
        "training=no_lora",
        "dataset=our_long_form",
        f"probe_config.layer={args.layer}",
        f"model_dtype={args.model_dtype}",
        f"probe_config.probe_dtype={args.probe_dtype}",
        f"probe_config.normalize_before_head={args.norm}",
        f"probe_config.lora_layers={args.lora_layers}",
        f"probe_head_lr={args.lr}",
        f"seed={args.seed}",
        f"probe_config.probe_id={args.probe_id}",
        f"wandb_tags=[{args.wandb_tags}]",
    ]
    print("[run_probe] Checkpoint not found — launching training:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from feature_probes.utils.probe_loader import LOCAL_PROBES_DIR
    checkpoint = LOCAL_PROBES_DIR / args.probe_id / "probe_head.bin"
    adapter_cfg = LOCAL_PROBES_DIR / args.probe_id / "adapter_config.json"

    needs_lora_adapter = str(args.lora_layers).strip().lower() != "none"
    has_minimal_checkpoint = checkpoint.exists() and (adapter_cfg.exists() or not needs_lora_adapter)

    if has_minimal_checkpoint:
        print(f"[run_probe] Checkpoint found at {checkpoint} → running evaluation.")
        run_eval(args)
    else:
        if not checkpoint.exists():
            reason = f"missing {checkpoint.name}"
        elif needs_lora_adapter and not adapter_cfg.exists():
            reason = f"missing {adapter_cfg.name} for LoRA run"
        else:
            reason = "incomplete checkpoint"
        print(f"[run_probe] Checkpoint incomplete ({reason}) → running training.")
        run_train(args)


if __name__ == "__main__":
    main()
