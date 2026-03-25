"""Training script for hallucination detection probes."""
import hydra
from omegaconf import DictConfig, OmegaConf

import os
import json
import atexit
import sys
from pathlib import Path
from typing import List
from dataclasses import asdict
import argparse

# When running as `python probe/train.py`, prefer local repo modules over any
# previously installed `probe` package from site-packages.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import wandb
from torch.utils.data import Subset
from transformers import TrainingArguments
from dotenv import load_dotenv

from feature_probes.utils.file_utils import save_jsonl, save_json, load_yaml
from feature_probes.utils.model_utils import load_model_and_tokenizer, print_trainable_parameters, resolve_torch_dtype
from feature_probes.utils.probe_loader import upload_probe_to_hf

from feature_probes.data.dataset import TokenizedProbingDataset, create_probing_dataset, tokenized_probing_collate_fn
from feature_probes.config import TrainingConfig
from feature_probes.probes.value_head_probe import setup_probe
from feature_probes.training.trainer import ProbeTrainer


def main(training_config: TrainingConfig):
    """Main training function."""

    # Load environment variables from .env if present
    load_dotenv()

    if training_config.upload_to_hf:
        assert os.environ.get("HF_WRITE_TOKEN", None) is not None
    
    wandb.init(project=training_config.wandb_project, 
               name=training_config.probe_config.probe_id,
               tags=training_config.wandb_tags or None,
        )

    print("Training config:")
    for key, value in asdict(training_config).items():
        print(f"\t{key}: {value}")

    # Load model and tokenizer
    print(f"Loading model: {training_config.probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        training_config.probe_config.model_name,
        torch_dtype=resolve_torch_dtype(training_config.model_dtype),
    )

    if hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if training_config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    
    print(f"Setting up probe: {training_config.probe_config.probe_id}")
    model, probe = setup_probe(model, training_config.probe_config, seed=training_config.seed)

    print_trainable_parameters(probe)

    # Load datasets
    print("Loading datasets:")
    train_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.train_dataset_configs
    ]
    eval_datasets: List[TokenizedProbingDataset] = [
        create_probing_dataset(config, tokenizer)
        for config in training_config.eval_dataset_configs
    ]
    
    # Concatenate training datasets
    train_dataset = train_datasets[0]
    for dataset in train_datasets[1:]:
        train_dataset += dataset

    # If requested, shuffle and shave down the training dataset to a fixed number of samples
    if training_config.num_train_samples is not None:
        total = len(train_dataset)
        num = max(0, min(int(training_config.num_train_samples), total))
        if num < total:
            g = torch.Generator()
            g.manual_seed(training_config.seed)
            perm = torch.randperm(total, generator=g).tolist()
            selected_indices = perm[:num]
            train_dataset = Subset(train_dataset, selected_indices)
            print(f"Using a subset of the training dataset: {num}/{total} samples")

    training_args = TrainingArguments(
        output_dir=str(training_config.probe_config.probe_path),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        max_steps=training_config.max_steps,
        num_train_epochs=training_config.num_train_epochs,
        logging_steps=training_config.logging_steps,
        eval_steps=training_config.eval_steps,
        remove_unused_columns=False,
        label_names=["classification_labels", "lm_labels"],
        report_to="wandb",
        run_name=training_config.probe_config.probe_id,
        eval_strategy="steps" if training_config.eval_steps else "no",
        logging_first_step=True,
        logging_strategy="steps",
        max_grad_norm=training_config.max_grad_norm,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        seed=training_config.seed,
    )
    
    # Add separate learning rates to training_args
    training_args.probe_head_lr = training_config.probe_head_lr
    training_args.lora_lr = training_config.lora_lr

    # Disable checkpoint saving
    # (there's a weird bug that occurs when trying to save during training)
    training_args.set_save(strategy="no")

    trainer = ProbeTrainer(
        probe=probe,
        eval_datasets=eval_datasets,
        cfg=training_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None, # this is a dummy argument is for the HF base Trainer class
        data_collator=tokenized_probing_collate_fn,
        eval_steps=training_config.eval_steps,
        tokenizer=tokenizer,
    )

    def save_model_callback():
        """Save probe weigths, tokenizer and training config to disk."""
        probe.save(training_config.probe_config.probe_path)
        tokenizer.save_pretrained(training_config.probe_config.probe_path)
        save_json(
            training_config,
            training_config.probe_config.probe_path / "training_config.json"
        )

    # Register save callback for unexpected exits
    atexit.register(save_model_callback)
    
    print("Training...")
    trainer.train()

    # Save the model
    print(f"Saving model to {training_config.probe_config.probe_path}")
    save_model_callback()

    # Final evaluation
    eval_metrics = trainer.evaluate(
        save_roc_curves=training_config.save_roc_curves,
        dump_raw_eval_results=training_config.dump_raw_eval_results,
        verbose=True,
    )

    if training_config.save_evaluation_metrics:
        save_json(
            eval_metrics,
            training_config.probe_config.probe_path / "evaluation_results.json"
        )

    wandb.finish()

    if training_config.upload_to_hf:
        print(f"Uploading probe to HuggingFace Hub...")
        try:
            upload_probe_to_hf(
                repo_id=training_config.probe_config.hf_repo_id,
                probe_id=training_config.probe_config.probe_id,
                token=os.environ.get("HF_WRITE_TOKEN"),
            )
        except Exception as exc:
            print(f"WARNING: Upload to HuggingFace failed: {exc}")
            print("WARNING: Training completed and artifacts remain available locally.")

@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def hydra_entry(cfg: DictConfig):
    """
    Hydra parses the yaml files and overrides, then passes a DictConfig here.
    We convert it to a standard Python dictionary, then unpack it into your
    existing TrainingConfig dataclass.
    """
    # 1. Convert DictConfig to a standard dictionary recursively
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # remove hydra artifacts
    config_dict.pop("model", None)
    config_dict.pop("training", None)
    config_dict.pop("dataset", None)
    
    # 2. Instantiate your existing dataclass
    training_config = TrainingConfig(**config_dict)
    
    # 3. Call your original main function
    main(training_config)

if __name__ == "__main__":
    # Remove the argparse block entirely

    hydra_entry()
