"""ValueHeadProbe: A probe that attaches to a specific layer and predicts hallucinations."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from jaxtyping import Float, Int
from torch import Tensor

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel

from feature_probes.utils.hooks import add_hooks
from feature_probes.utils.model_utils import (
    get_model_layers,
    get_model_hidden_size,
    setup_lora_for_layers,
    resolve_torch_dtype,
)
from feature_probes.utils.probe_loader import download_probe_from_hf
from feature_probes.probes.attention_probe import PerTokenAttentionProbe, AttentionProbeHead


class L2Norm(nn.Module):
    """Per-token L2 normalization for hidden features before probe head."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        return x / denom

class ValueHeadProbe(nn.Module):
    """
    A probe that hooks into a specific layer of a language model and uses a linear
    head to predict per-token binary classification (e.g., hallucination detection).
    
    This probe:
    1. Attaches a forward hook to capture hidden states from a specified layer
    2. Applies a linear transformation to predict hallucination scores
    3. Can be used with or without LoRA adapters
    """
    
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, PeftModel],
        layer_idx: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        context_window_size: Optional[int] = 1,
        attention_probe_n_heads: int = 4,
        probe_dtype: Optional[str] = None,
        normalize_before_head: Optional[str] = None,
        probe_head_type: str = "linear",
        seed: int = 42,
    ):
        """
        Initialize the ValueHeadProbe.
        
        Args:
            model: The language model (with or without LoRA adapters)
            layer_idx: Optional index of the layer to attach the probe to (if not specified we'll infer it from the previously saved probe configuration file)
            path: Optional path to load pre-trained probe weights from
        """
        super().__init__()

        assert layer_idx or path, "Either path or layer index must be provided, otherwise we can't infer the layer where we should hook the value head to."
        
        saved_probe_dtype = None
        saved_norm = None
        if path:
            saved_config = json.load(open(path / "probe_config.json"))

            if layer_idx is None:
                layer_idx = saved_config["layer_idx"]
            else:
                # We check that if a layer_idx is provided, it matches the one given in the previously saved config
                assert layer_idx == saved_config["layer_idx"]
            saved_probe_dtype = saved_config.get("probe_dtype")
            saved_norm = saved_config.get("normalize_before_head")
            probe_head_type = saved_config.get("probe_head_type", probe_head_type)

        hidden_size = get_model_hidden_size(model)
        model_layers = get_model_layers(model)

        self.model = model
        self.layer_idx = layer_idx
        self.target_module = model_layers[layer_idx]
        self.target_layer_name = self.target_module.__class__.__name__
        self.context_window_size = context_window_size
        self.attention_probe_n_heads = attention_probe_n_heads
        self.probe_head_type = str(probe_head_type).strip().lower()

        self.probe_dtype = probe_dtype if probe_dtype is not None else (saved_probe_dtype or "auto")
        self.normalize_before_head = (
            normalize_before_head if normalize_before_head is not None else (saved_norm or "none")
        ).strip().lower()
        head_dtype = resolve_torch_dtype(self.probe_dtype, default=model.dtype)

        norm_hidden_size = hidden_size * context_window_size
        self.pre_head_norm = self._build_pre_head_norm(
            self.normalize_before_head,
            hidden_size=norm_hidden_size,
            device=model.device,
            dtype=head_dtype,
        )

        if not isinstance(model, PeftModel):
            print("WARNING: Model is not a PeftModel. Remember to add LoRA adapters if needed.")

        # Initialize the value head
        input_size = hidden_size * context_window_size
        if path:
            # Load pre-trained weights if path is provided
            self.value_head, _ = ValueHeadProbe.load_head(
                path,
                device=model.device,
                dtype=head_dtype,
            )
        elif self.probe_head_type == "attention":
            torch.manual_seed(seed)
            self.value_head = PerTokenAttentionProbe(
                input_size,
                n_heads=self.attention_probe_n_heads,
                n_outputs=1,
                device=model.device,
                dtype=head_dtype,
            )
        else:  # "linear"
            torch.manual_seed(seed)
            self.value_head = nn.Linear(input_size, 1, device=model.device, dtype=head_dtype)
            self._initialize_weights()
        
        # Initialize hook state
        self._hooked_hidden_states: Optional[torch.Tensor] = None
        self._hook_fn = self._get_hook_fn()

    @staticmethod
    def _build_pre_head_norm(
        norm_kind: str,
        *,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        key = (norm_kind or "none").strip().lower()
        if key == "none":
            return nn.Identity()
        if key == "layernorm":
            return nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        if key == "rmsnorm":
            return nn.RMSNorm(hidden_size, device=device, dtype=dtype)
        if key == "l2":
            return L2Norm()
        raise ValueError(f"Unsupported normalize_before_head={norm_kind!r}")
    
    def _initialize_weights(self):
        """Initialize the value head weights with small random values."""
        with torch.no_grad():
            self.value_head.weight.data.normal_(mean=0.0, std=0.01)
            if self.value_head.bias is not None:
                self.value_head.bias.data.zero_()
    
    def _get_hook_fn(self):
        """
        Forward hook to capture hidden states from target layer.
        `module_output` is typically [batch_size, seq_len, hidden_dim].
        We do NOT detach it, so gradients can flow back.
        """
        def hook_fn(module, module_input, module_output):
            if isinstance(module_input, tuple) and module_output[0].ndim == 3:
                self._hooked_hidden_states = module_output[0]
            else:
                self._hooked_hidden_states = module_output
        return hook_fn
    
    def forward(
        self,
        input_ids: Int[Tensor, 'batch_size seq_len'],
        attention_mask: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        labels: Optional[Int[Tensor, 'batch_size seq_len']] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model with probe attached.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for language modeling loss
            **kwargs: Additional arguments passed to the model
        
        Returns:
            Dictionary containing:
                - lm_logits: Language model output logits [batch_size, seq_len, vocab_size]
                - probe_logits: Probe output logits [batch_size, seq_len, 1]
                - lm_loss: Language modeling loss (if labels provided)
        """
        # Reset hooked hidden states
        self._hooked_hidden_states = None
        
        # Set up hooks
        fwd_hooks = [(self.target_module, self._hook_fn)]
        
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            # Forward pass through the model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                **kwargs
            )
        
        # Check that hidden states were captured
        if self._hooked_hidden_states is None:
            raise RuntimeError("Failed to capture hidden states from target layer")
        
        shifted = [self._hooked_hidden_states]
        for k in range(1, self.context_window_size):
            s = self._hooked_hidden_states.roll(k, dims=1)
            s[:, :k, :] = 0   # zero-pad wrapped positions
            shifted.append(s)
        
        # Concatenate context window hidden states
        context_hidden_states = torch.cat(shifted, dim=-1)

        # Keep norm input dtype aligned with norm params/buffers (e.g., fp32 LayerNorm).
        norm_dtype = None
        for tensor in self.pre_head_norm.parameters():
            norm_dtype = tensor.dtype
            break
        if norm_dtype is None:
            for tensor in self.pre_head_norm.buffers():
                norm_dtype = tensor.dtype
                break
        if norm_dtype is not None and context_hidden_states.dtype != norm_dtype:
            context_hidden_states = context_hidden_states.to(norm_dtype)

        context_hidden_states = self.pre_head_norm(context_hidden_states)

        # Ensure head/input dtypes match when model and probe run at different precision.
        try:
            head_param = next(self.value_head.parameters())
            if context_hidden_states.dtype != head_param.dtype:
                context_hidden_states = context_hidden_states.to(head_param.dtype)
        except StopIteration:
            pass

        probe_logits: Float[Tensor, 'batch_size seq_len 1'] = self.value_head(context_hidden_states)

        return {
            "lm_logits": outputs.logits,
            "probe_logits": probe_logits,
            "lm_loss": outputs.loss if hasattr(outputs, 'loss') else None
        }
    
    def save(self, path: Union[str, Path]):
        """
        Save the probe to disk.
        
        Args:
            path: Directory to save the probe to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapters if present
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        
        # Save value head weights
        torch.save(
            self.value_head.state_dict(),
            path / "probe_head.bin"
        )
        
        # Save configuration
        if isinstance(self.value_head, nn.Linear):
            hidden_size = self.value_head.in_features
        elif hasattr(self.value_head, "query_proj"):  # PerTokenAttentionProbe
            hidden_size = self.value_head.query_proj.in_features
        else:
            hidden_size = None

        probe_config = {
            "target_layer_name": self.target_module.__class__.__name__,
            "layer_idx": self.layer_idx,
            "attention_probe_n_heads": self.attention_probe_n_heads,
            "probe_dtype": self.probe_dtype,
            "normalize_before_head": self.normalize_before_head,
            "hidden_size": hidden_size,
            "probe_head_type": self.probe_head_type,
            "attention_probe_n_heads": self.attention_probe_n_heads,
        }
        with open(path / "probe_config.json", 'w') as f:
            json.dump(probe_config, f, indent=4)
        
        print(f"Probe saved to {path}")
    
    @property
    def device(self):
        """Get the device of the model."""
        return self.model.device
    
    @classmethod
    def load_head(
        cls,
        path: Path,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[nn.Module, int]:
        """
        Loads the linear value head from the given path.
        
        Args:
            path: Path to the probe directory
            device: Device to load the probe head on
            dtype: Data type for the probe head
            
        Returns:
            Tuple of (probe_head, layer_idx)
        """
        with open(path / "probe_config.json") as f:
            probe_config = json.load(f)

        hidden_size = probe_config['hidden_size']
        probe_layer_idx = probe_config['layer_idx']
        head_type = probe_config.get("probe_head_type", "attention")  # old saves default to attention
        n_heads = probe_config.get("attention_probe_n_heads", 4)

        if head_type == "attention":
            probe_head = PerTokenAttentionProbe(hidden_size, n_heads=n_heads, n_outputs=1, device=device, dtype=dtype)
        else:
            probe_head = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
        
        state_dict = torch.load(
            path / "probe_head.bin",
            map_location=device,
            weights_only=True
        )
        probe_head.load_state_dict(state_dict)
        
        return probe_head, probe_layer_idx

    def _probe_input_size(self) -> int:
        if hasattr(self.value_head, "query_proj"):
            return int(self.value_head.query_proj.in_features)
        if hasattr(self.value_head, "in_features"):
            return int(self.value_head.in_features)
        raise ValueError("Unable to infer probe input size from value head")


def setup_probe(
    model: PreTrainedModel,
    probe_config: 'ProbeConfig',
    seed: int = 42,
) -> Tuple[PreTrainedModel, ValueHeadProbe]:
    """
    Set up a probe with the given configuration.
    
    This handles downloading from HF if needed, setting up LoRA adapters,
    and creating the ValueHeadProbe instance.
    
    Args:
        model: The base language model
        probe_config: Probe configuration
        
    Returns:
        Tuple of (model with LoRA if applicable, probe instance)
    """
    
    # We freeze all parameters of the base model
    # otherwise the optimizer will do a full-finetuning
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Download from HF if needed
    if probe_config.load_from == 'hf' and not probe_config.probe_path.exists():
        download_probe_from_hf(
            repo_id=probe_config.hf_repo_id,
            probe_id=probe_config.probe_id
        )

    if probe_config.load_from in ['hf', 'disk']:
        # Set up the LoRAs (if applicable)
        if (probe_config.probe_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, probe_config.probe_path)

        # Load existing probe (layer_idx will be inferred from previously saved config)
        probe = ValueHeadProbe(
            model,
            path=probe_config.probe_path,
            attention_probe_n_heads=probe_config.attention_probe_n_heads,
            probe_dtype=probe_config.probe_dtype,
            normalize_before_head=probe_config.normalize_before_head,
        )

    else:
        # Initialize the probe from scratch
        if probe_config.lora_layers:
            print(f"Initializing new LoRA adapters for layers {probe_config.lora_layers}")
            model = setup_lora_for_layers(
                model,
                probe_config.lora_layers,
                lora_r=probe_config.lora_r,
                lora_alpha=probe_config.lora_alpha,
                lora_dropout=probe_config.lora_dropout,
            )
        
        # Initialize new probe with specified layer

        probe = ValueHeadProbe(
            model,
            layer_idx=probe_config.layer,
            context_window_size=probe_config.context_window_size,
            attention_probe_n_heads=probe_config.attention_probe_n_heads,
            probe_dtype=probe_config.probe_dtype,
            normalize_before_head=probe_config.normalize_before_head,
            probe_head_type=probe_config.probe_head_type,
            seed=seed,
        )
  
    return model, probe
