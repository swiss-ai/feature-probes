"""Probe module for hallucination detection."""

from .value_head_probe import ValueHeadProbe
from .config import ProbeConfig, TrainingConfig, EvaluationConfig
from .loss import (
    compute_probe_bce_loss,
    compute_probe_max_aggregation_loss,
    compute_sparsity_loss,
    compute_kl_divergence_loss,
    mask_high_loss_spans
)
from .types import (
    ProbingItem,
    AnnotatedSpan
)
from .dataset import (
    TokenizedProbingDataset,
    tokenized_probing_collate_fn,
    create_probing_dataset
)

__all__ = [
    "ValueHeadProbe",
    "ProbeConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "compute_probe_bce_loss",
    "compute_probe_max_aggregation_loss",
    "compute_sparsity_loss",
    "compute_kl_divergence_loss",
    "mask_high_loss_spans",
    "setup_probe",
    "ProbingItem",
    "AnnotatedSpan",
    "TokenizedProbingDataset",
    "tokenized_probing_collate_fn",
    "create_probing_dataset"
]