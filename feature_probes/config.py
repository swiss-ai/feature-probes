"""Configuration classes for probe training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Literal

from feature_probes.utils.probe_loader import LOCAL_PROBES_DIR
from feature_probes.utils.model_utils import get_num_layers
from feature_probes.data.dataset import TokenizedProbingDatasetConfig

@dataclass
class ProbeConfig:
    """Configuration for a probe model."""
    probe_id: str = "probe"  
    
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    layer: Optional[int] = None  # Which layer to attach the probe to
    
    # LoRA configuration
    lora_layers: Optional[Union[List[int], str]] = "all"  # Which layers to apply LoRA to
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.05  # LoRA dropout

    # Loading configuration
    load_from: Optional[Literal['disk', 'hf']] = None  # "disk", "hf", or None
    probe_path: Optional[Path] = None  # Local path for disk loading
    hf_repo_id: Optional[str] = "tkwiecinski/hallucination-probes"  # HuggingFace repository ID
    
    threshold: float = 0.5  # Classification threshold
    context_window_size: int = 1  # Size of context window for the probe
    probe_dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    normalize_before_head: str = "none"  # none | layernorm | rmsnorm | l2
    attention_probe_n_heads: int = 4  # Number of attention heads for the probe (only used when probe_head_type="attention")
    probe_head_type: str = "linear"  # linear | attention

    def __post_init__(self):
        """Validate configuration."""
        self.probe_path = LOCAL_PROBES_DIR / self.probe_id

        if self.load_from == "hf" and not self.hf_repo_id:
            raise ValueError("hf_repo_id must be specified when load_from='hf'")

        if self.load_from in ['disk'] and not self.probe_path.exists():
            raise ValueError(f"Probe with ID {self.probe_id} not found in disk at path {self.probe_path}")

        if self.layer is None:
            # default to hooking the value head at the last layer of the underlying LM
            self.layer = get_num_layers(self.model_name) - 1
        
        if (not isinstance(self.context_window_size, int) or self.context_window_size < 0):
            raise ValueError("context_window must be a positive integer.")

        allowed_probe_dtypes = {"auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}
        probe_dtype_key = str(self.probe_dtype).strip().lower()
        if probe_dtype_key not in allowed_probe_dtypes:
            raise ValueError(
                f"Unsupported probe_dtype={self.probe_dtype!r}. "
                "Use one of: auto, float32, float16, bfloat16"
            )

        allowed_norms = {"none", "layernorm", "rmsnorm", "l2"}
        norm_key = str(self.normalize_before_head).strip().lower()
        if norm_key not in allowed_norms:
            raise ValueError(
                f"Unsupported normalize_before_head={self.normalize_before_head!r}. "
                "Use one of: none, layernorm, rmsnorm, l2"
            )
        if str(self.probe_head_type).strip().lower() not in {"linear", "attention"}:
            raise ValueError(
                f"Unsupported probe_head_type={self.probe_head_type!r}. Use 'linear' or 'attention'."
            )

        if isinstance(self.lora_layers, str):
            if self.lora_layers == 'all':
                # default to training LoRA adaptors on all layers
                # (up to the layer where we hook the value head)
                self.lora_layers = list(range(0, self.layer + 1))
            elif self.lora_layers.lower() == 'none':
                self.lora_layers = []
            else:
                self.lora_layers = [int(layer) for layer in self.lora_layers.strip('[').strip(']').split(",")]
                assert len(self.lora_layers) > 0
        elif self.lora_layers is None:
            self.lora_layers = []
        assert all(isinstance(l, int) for l in self.lora_layers)


@dataclass
class TrainingConfig:
    """Configuration for probe training."""
    
    wandb_project: str = "hallucination-probes"
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list) 
    
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)
    
    upload_to_hf: bool = False
    save_evaluation_metrics: bool = True
    save_roc_curves: bool = False
    dump_raw_eval_results: bool = False
    
    # Training hyperparameters
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    high_loss_threshold: Optional[float] = None  # Threshold for masking high-loss tokens
    lambda_lm: float = 0.0  # Weight for language modeling loss regularization
    lambda_kl: float = 0.0  # Weight for KL divergence regularization
    anneal_max_aggr: bool = True  # Whether to anneal span-level max aggregation loss
    anneal_warmup: float = 1.0  # Fraction of training for span loss warmup
    learning_rate: float = 5e-5  # Overall learning rate (deprecated)
    probe_head_lr: Optional[float] = 5e-3 # Separate LR for probe head
    lora_lr: Optional[float] = 5e-5  # Separate LR for LoRA parameters
    sparsity_penalty_weight: Optional[float] = None
    num_train_samples: Optional[int] = None  # Limit training samples
    max_steps: int = -1  # Override num_epochs if set
    num_train_epochs: int = 1
    enable_gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: Optional[int] = -1  # Only manually evaluate at the end
    evaluation_strategy: str = "no"  # "steps", "epoch", or "no"
    logging_steps: int = 10
    seed: int = 42
    model_dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    
    # Dataset configuration
    train_datasets: List[dict] = field(default_factory=list)
    eval_datasets: List[dict] = field(default_factory=list)
    
    # These will be populated in __post_init__
    train_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)
    eval_dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.probe_config, dict):
            self.probe_config = ProbeConfig(**self.probe_config)
        
        # Handle special values
        if self.eval_steps == -1:
            self.eval_steps = None
        
        # Handle learning rates
        if self.probe_head_lr is None:
            self.probe_head_lr = self.learning_rate
        if self.lora_lr is None:
            self.lora_lr = self.learning_rate
        
        # Parse dataset configurations
        self.train_dataset_configs = [
            TokenizedProbingDatasetConfig(**config)
            for config in self.train_datasets
        ]
        
        self.eval_dataset_configs = [
            TokenizedProbingDatasetConfig(**config)
            for config in self.eval_datasets
        ]

        # Convert scientific notation strings to floats
        # otherwise yaml parses e.g. '1e-6' as a string instead of a float
        float_fields = [
            'learning_rate', 'probe_head_lr', 'lora_lr', 'max_grad_norm', 
            'anneal_warmup', 'lambda_lm', 'lambda_kl', 'sparsity_penalty_weight'
        ]
        for field_name in float_fields:
            value = getattr(self, field_name)
            if value is not None and isinstance(value, str):
                setattr(self, field_name, float(value))

        allowed_model_dtypes = {"auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}
        model_dtype_key = str(self.model_dtype).strip().lower()
        if model_dtype_key not in allowed_model_dtypes:
            raise ValueError(
                f"Unsupported model_dtype={self.model_dtype!r}. "
                "Use one of: auto, float32, float16, bfloat16"
            )


@dataclass
class EvaluationConfig:
    """Configuration for probe evaluation."""
    
    probe_config: ProbeConfig = field(default_factory=ProbeConfig)
    
    datasets: List[dict] = field(default_factory=list)
    per_device_eval_batch_size: int = 8
    
    output_dir: Optional[Path] = None
    save_predictions: bool = True
    save_roc_curves: bool = True
    save_raw_results: bool = False  # Save all predictions and labels
    
    # This will be populated in __post_init__
    dataset_configs: List[TokenizedProbingDatasetConfig] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.probe_config, dict):
            self.probe_config = ProbeConfig(**self.probe_config)
        
        self.dataset_configs = [
            TokenizedProbingDatasetConfig(**dataset_config)
            for dataset_config in self.datasets
        ]

        if self.output_dir is None:
            self.output_dir = self.probe_config.probe_path / "evaluation_results"

# %%