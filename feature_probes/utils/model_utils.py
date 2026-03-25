"""Model loading and setup utilities."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel


def resolve_torch_dtype(
    dtype_name: Optional[str],
    *,
    default: Optional[torch.dtype] = None,
) -> Optional[torch.dtype]:
    """Resolve config dtype name to a torch dtype.

    Supported values: auto, fp32/float32, fp16/float16, bf16/bfloat16.
    """
    if dtype_name is None:
        return default

    key = str(dtype_name).strip().lower()
    if key in {"", "auto", "default", "model"}:
        return default
    if key in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if key in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16

    raise ValueError(
        f"Unsupported dtype '{dtype_name}'. Use one of: auto, float32, float16, bfloat16."
    )


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str,
    device_map: Optional[Union[str, dict]] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name or path of the model
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        device_map: Device mapping for model parallelism
        torch_dtype: Data type for model weights
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set default dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_model_with_lora(
    model: AutoModelForCausalLM,
    lora_config: dict,
    lora_weights_path: Optional[str] = None,
) -> PeftModel:
    """
    Setup a model with LoRA adapters.
    
    Args:
        model: Base model
        lora_config: LoRA configuration dictionary
        lora_weights_path: Optional path to pre-trained LoRA weights
        
    Returns:
        Model with LoRA adapters
    """
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
    )
    
    # Apply LoRA to model
    if lora_weights_path:
        # Load pre-trained LoRA weights
        model = PeftModel.from_pretrained(model, lora_weights_path)
    else:
        # Initialize new LoRA adapters
        model = get_peft_model(model, peft_config)
    
    return model


def get_model_layers(model: PreTrainedModel) -> List[nn.Module]:
    """
    Get the list of transformer layers from a model.
    
    Args:
        model: The transformer model
    
    Returns:
        List of layer modules
    """
    # Handle PeftModel by getting the base model
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    
    # Common patterns for accessing layers in different model architectures
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        # LLaMA, Mistral, etc.
        return list(base_model.model.layers)
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        # GPT-2, GPT-J, etc.
        return list(base_model.transformer.h)
    elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        # BERT, RoBERTa, etc.
        return list(base_model.encoder.layer)
    elif hasattr(base_model, 'gpt_neox') and hasattr(base_model.gpt_neox, 'layers'):
        # GPT-NeoX
        return list(base_model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")


def get_num_layers(model_or_name: Union[str, PreTrainedModel]) -> int:
    """
    Get the number of transformer layers in a model.
    
    Args:
        model_or_name: Either a model name string or a transformer model
    
    Returns:
        Number of layers
    """
    # If it's a string (model name), use the predefined mapping
    if isinstance(model_or_name, str):
        model_layers_map = {
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 32,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 80,
            "meta-llama/Meta-Llama-3.1-405B-Instruct": 126,
            "google/gemma-2-2b-it": 26,
            "google/gemma-2-9b-it": 42,
            "google/gemma-2-27b-it": 46,
            "Qwen/Qwen2.5-0.5B-Instruct": 24,
            "Qwen/Qwen2.5-1.5B-Instruct": 28,
            "Qwen/Qwen2.5-3B-Instruct": 36,
            "Qwen/Qwen2.5-7B-Instruct": 28,
            "Qwen/Qwen2.5-14B-Instruct": 48,
            "Qwen/Qwen2.5-32B-Instruct": 64,
            "meta-llama/Llama-3.3-70B-Instruct": 80,
            "mistralai/Mistral-Small-24B-Instruct-2501": 40,
        }
        if model_or_name in model_layers_map:
            return model_layers_map[model_or_name]
        else:
            raise ValueError(f"Model {model_or_name} not supported. Please add it to the model_layers_map.")
    
    # If it's a model instance, count the layers
    return len(get_model_layers(model_or_name))


def get_model_layers_prefix(model: PreTrainedModel) -> str:
    """
    Get the prefix path to the model layers.
    
    Args:
        model: The transformer model
    
    Returns:
        String prefix for accessing layers (e.g., "model.layers")
    """
    # Handle PeftModel
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        return "model.layers"
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        return "transformer.h"
    elif hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        return "encoder.layer"
    elif hasattr(base_model, 'gpt_neox') and hasattr(base_model.gpt_neox, 'layers'):
        return "gpt_neox.layers"
    else:
        raise ValueError(f"Unknown model architecture: {type(base_model)}")


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """
    Get the hidden size of a transformer model.
    
    Args:
        model: The transformer model
        
    Returns:
        Hidden size of the model
    """
    # Handle PeftModel
    if isinstance(model, PeftModel):
        base_model = model.get_base_model()
    else:
        base_model = model
    
    if hasattr(base_model, 'config'):
        config = base_model.config
        # Try common attribute names
        for attr in ['hidden_size', 'd_model', 'n_embd', 'embed_dim']:
            if hasattr(config, attr):
                return getattr(config, attr)
    
    # If we can't find it in config, try to infer from the model structure
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
        return base_model.model.embed_tokens.weight.shape[1]
    
    raise ValueError(f"Could not determine hidden size for model type {type(base_model)}")


def setup_lora_for_layers(
    model: PreTrainedModel,
    layer_indices: List[int],
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> Union[PeftModel, PreTrainedModel]:
    """
    Setup LoRA adapters for specific layers in a model.
    
    Args:
        model: Base model to apply LoRA to
        layer_indices: List of layer indices to apply LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        bias: Bias configuration for LoRA
        
    Returns:
        Model with LoRA adapters applied (or original model if no layers specified)
    """
    if not layer_indices:
        print("No LoRA layers specified, returning base model")
        return model
    
    # Get the layer prefix for this model architecture
    layer_prefix = get_model_layers_prefix(model)
    
    # Build target modules list for the specified layers
    target_modules = []
    module_suffixes = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    ]
    
    for layer_idx in layer_indices:
        for module_suffix in module_suffixes:
            target_modules.append(f"{layer_prefix}.{layer_idx}.{module_suffix}")
    
    print(f"Creating LoRA adapters for layers {layer_indices}...")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    return get_peft_model(model, lora_config)


def print_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Print information about trainable parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = 0
    total_params = 0
    
    print("Parameters that will be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  - {name}: shape {param.shape}, device {param.device}")
        total_params += param.numel()
    
    trainable_params_percentage = 100 * trainable_params / total_params
    print(f"\nTotal trainable parameters: {trainable_params:,} ({trainable_params_percentage:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    return trainable_params, total_params