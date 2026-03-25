"""Probe module for feature detection."""

__version__ = "0.1.0"

# Only lightweight, always-needed types at the top level
from .types import ProbingItem, AnnotatedSpan
from .config import ProbeConfig, TrainingConfig, EvaluationConfig
