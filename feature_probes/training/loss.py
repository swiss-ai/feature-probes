"""Loss functions for probe training."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from peft import PeftModel
from transformers import AutoModelForCausalLM

def compute_probe_bce_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len'],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    classification_weights: Float[Tensor, 'batch_size seq_len'],
    max_clipped_logits: float = 100.0,
    ignore_label: float = -100.0,
):
    # Clip logits to prevent extreme values
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )
    try:
        bce_loss = F.binary_cross_entropy_with_logits(
            probe_logits_clipped,
            classification_labels,
            weight=classification_weights,
            reduction='none'
        )
        bce_loss = bce_loss[(classification_labels != ignore_label)].mean()
            
        # Check for NaN in bce_loss
        if torch.isnan(bce_loss):
            print(f"WARNING: NaN detected in bce_loss")
            bce_loss = torch.tensor(0.0, device=probe_logits.device)
        return bce_loss
    except Exception as e:
        print(f"Error in compute_probe_bce_loss: {e}")
        return torch.tensor(0.0, device=probe_logits.device)


def compute_probe_max_aggregation_loss(
    probe_logits: Float[Tensor, 'batch_size seq_len'],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    classification_weights: Float[Tensor, 'batch_size seq_len'],
    positive_spans: List[List[Tuple[int, int]]],
    negative_spans: List[List[Tuple[int, int]]],
    max_clipped_logits: float = 100.0,
    sparsity_penalty_weight: Optional[float] = None,
):
    """
    Computes the span-level max-aggregation loss.
    For positive spans, loss is BCE(max(logits_in_span), 1.0).
    For negative spans, loss is BCE(max(logits_in_span), 0.0).
    The final loss is the mean over all spans in the batch.
    """

    span_losses = []
    device = probe_logits.device
    dtype = probe_logits.dtype

    # Clip logits to prevent extreme values
    probe_logits_clipped = torch.clamp(
        probe_logits,
        min=-max_clipped_logits,
        max=max_clipped_logits
    )

    for i in range(probe_logits_clipped.shape[0]): # Iterate over batch items
        # Positive spans
        for start, end in positive_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]

            max_logit = torch.max(span_logits)
            target = torch.tensor(1.0, device=device, dtype=dtype)
            loss = F.binary_cross_entropy_with_logits(max_logit, target, reduction='none')
            span_losses.append(loss)

        # Negative spans
        for start, end in negative_spans[i]:
            should_ignore = (classification_labels[i, start:end+1] == -100.0).any()

            if should_ignore:
                continue

            span_logits = probe_logits_clipped[i, start:end+1]

            max_logit = torch.max(span_logits)
            target = torch.tensor(0.0, device=device, dtype=dtype)
            loss = F.binary_cross_entropy_with_logits(max_logit, target, reduction='none')
            span_losses.append(loss)

    if not span_losses:
        # No valid spans found in the batch, return zero loss
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Compute the mean loss over all spans in the batch
    final_loss = torch.mean(torch.stack(span_losses))

    if sparsity_penalty_weight is not None:
        sparsity_loss = compute_sparsity_loss(
            probe_logits=probe_logits,
            classification_labels=classification_labels,
        )

        final_loss = final_loss + sparsity_penalty_weight * sparsity_loss

    # Check for NaN
    if torch.isnan(final_loss):
        print(f"WARNING: NaN detected in compute_probe_max_aggregation_loss. Returning 0.0")
        final_loss = torch.tensor(0.0, device=device, dtype=dtype)

    return final_loss


def compute_sparsity_loss(
    probe_logits: Float[Tensor, "batch seq_len 1"],
    attention_mask: Int[Tensor, "batch seq_len"],
) -> Float[Tensor, ""]:
    """
    Compute sparsity loss to encourage probe to be selective.
    
    This loss encourages the probe to have low average activation,
    preventing it from flagging everything as a hallucination.
    
    Args:
        probe_logits: Probe output logits
        attention_mask: Mask for valid tokens
    
    Returns:
        Scalar sparsity loss
    """
    # Get probabilities
    probe_probs = torch.sigmoid(probe_logits.squeeze(-1))  # [batch, seq_len]
    
    # Apply attention mask
    masked_probs = probe_probs * attention_mask
    
    # Compute average activation
    num_valid_tokens = attention_mask.sum()
    if num_valid_tokens == 0:
        return torch.tensor(0.0, device=probe_logits.device)
    
    avg_activation = masked_probs.sum() / num_valid_tokens
    
    # Sparsity loss is just the average activation
    return avg_activation


def mask_high_loss_spans(
    lm_model: Union[AutoModelForCausalLM, PeftModel],
    input_ids: Float[Tensor, 'batch_size seq_len'],
    attention_mask: Float[Tensor, 'batch_size seq_len'],
    classification_labels: Float[Tensor, 'batch_size seq_len'],
    spans: List[List[Tuple[int, int]]],
    threshold: float = 1.0, # threshold for the loss to be considered high
):
    """
    Computes the span-level max-aggregation loss.
    For positive spans, loss is BCE(max(logits_in_span), 1.0).
    For negative spans, loss is BCE(max(logits_in_span), 0.0).
    The final loss is the mean over all spans in the batch.
    """

    if isinstance(lm_model, PeftModel):
        with lm_model.disable_adapter():
            lm_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            ).logits
    else:
        lm_logits: Float[Tensor, 'batch_size seq_len vocab_size'] = lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        ).logits

    log_probs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
    log_probs: Float[Tensor, 'batch_size seq_len'] = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    for i in range(log_probs.shape[0]): # Iterate over batch items
        # Do it only for the supported entity spans
        for start, end in spans[i]:
            if start > end: # Invalid span, skip
                continue

            span_neg_log_probs = -log_probs[i, start:end+1]
            max_neg_log_prob = span_neg_log_probs.max()

            if max_neg_log_prob > threshold:
                classification_labels[i, start:end+1] = -100.0

    return classification_labels


def compute_kl_divergence_loss(
    model: 'ValueHeadProbe',
    lm_logits: Float[Tensor, "batch seq_len vocab_size"],
    input_ids: Int[Tensor, "batch seq_len"],
    attention_mask: Int[Tensor, "batch seq_len"],
    lm_labels: Int[Tensor, "batch seq_len"],
) -> Float[Tensor, ""]:
    """
    Compute KL divergence between model with LoRA and base model.
    
    Args:
        model: The ValueHeadProbe model
        lm_logits: Logits from model with LoRA adapters
        input_ids: Input token IDs
        attention_mask: Attention mask
        lm_labels: Language modeling labels
        
    Returns:
        Scalar KL divergence loss
    """
    
    # Check if model has LoRA adapters
    if not isinstance(model.model, PeftModel) or not model.model.active_adapters:
        return torch.tensor(0.0, device=lm_logits.device)
    
    # Get base model outputs without LoRA
    with model.model.disable_adapter():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        base_logits = base_outputs["lm_logits"].detach()
    
    # Compute KL divergence
    with torch.autocast(device_type=lm_logits.device.type, enabled=False):
        log_q = torch.log_softmax(lm_logits.float(), -1)  # model with LoRA
        p_ref = torch.softmax(base_logits.float(), -1)    # reference model
        kl = F.kl_div(log_q, p_ref, reduction='none', log_target=False).sum(-1)
        
        # Only compute loss on valid tokens
        active_mask = (lm_labels != -100)
        if not active_mask.any():
            return torch.tensor(0.0, device=lm_logits.device)
            
        kl_loss = kl[active_mask].mean()
    
    return kl_loss