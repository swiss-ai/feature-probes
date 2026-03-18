"""Data types for probe training."""

from dataclasses import dataclass
from typing import List


@dataclass
class AnnotatedSpan:
    """A text span with its hallucination label."""
    
    span: str  # The span text
    label: float  # 1.0 for hallucination, 0.0 for supported, -100.0 for ignored
    index: int  # Start index in the completion


@dataclass
class ProbingItem:
    """A single item containing prompt, completion and annotated spans."""
    
    prompt: str
    completion: str
    spans: List[AnnotatedSpan]