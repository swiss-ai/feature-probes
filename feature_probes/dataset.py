"""Tokenized dataset classes with token-level labels for probe training."""

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import datasets
from jaxtyping import Float, Int
from termcolor import colored
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.tokenization import find_assistant_tokens_slice, find_string_in_tokens, slice_to_list
from probe.types import AnnotatedSpan, ProbingItem
from probe.dataset_converters import get_prepare_function

@dataclass
class TokenizedProbingDatasetConfig:
    """Configuration for tokenizing and labeling a probing dataset at token level."""
    
    dataset_id: str
    hf_repo: str
    subset: Optional[str] = None
    split: str = "train"
    max_length: int = 2048
    ignore_buffer: int = 0  # Buffer around spans to ignore
    default_ignore: bool = False  # If true, ignore tokens not in any span
    last_span_token: bool = False  # If true, only label the last token of each span
    pos_weight: float = 1.0  # Weight for positive (hallucination) tokens
    neg_weight: float = 1.0  # Weight for negative (supported) tokens
    shuffle: bool = True
    seed: int = 42
    process_on_the_fly: bool = False
    max_num_samples: Optional[int] = None


class TokenizedProbingDataset(Dataset):
    """Dataset for probing model activations with annotated spans."""
    
    def __init__(
        self,
        items: List[ProbingItem],
        config: TokenizedProbingDatasetConfig,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.items = deepcopy(items)
        self.processed_items = [None] * len(items)
        self.debug_mode = False
        self.print_first_example = False
        
        self._num_skipped_spans: int = 0
        self._num_added_spans: int = 0
        
        if self.config.shuffle:
            self._shuffle_items()
        
        # Limit samples if specified (do this after shuffling)
        if self.config.max_num_samples:
            self.items = self.items[:self.config.max_num_samples]
            self.processed_items = self.processed_items[:self.config.max_num_samples]
        
        if not self.config.process_on_the_fly:
            self._process_items()
    
    def _process_items(self):
        """Pre-process all items in the dataset."""
        for i, item in tqdm(enumerate(self.items), desc=f"Processing items ({self.config.dataset_id})", total=len(self.items)):
            if i == 0 and self.print_first_example:
                self.debug_mode = True
            else:
                self.debug_mode = False
            
            processed_item = self._process_item(item)
            if processed_item:
                self.processed_items[i] = processed_item
        
        print(f"Dataset {self.config.dataset_id} stats:")
        print(f"\t- Number of added spans: {self._num_added_spans}")
        print(f"\t- Number of skipped spans: {self._num_skipped_spans} / {self._num_added_spans + self._num_skipped_spans}")
        print(f"\t- Total number of items: {len(self.items)}")
    
    def _process_item(self, item: ProbingItem) -> Dict:
        """Process a single example into tokenized format with labels."""
        conversation = [
            {'role': 'user', 'content': item.prompt},
            {'role': 'assistant', 'content': item.completion}
        ]
        full_text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        
        if self.tokenizer.bos_token and self.tokenizer.bos_token in full_text:
            full_text = full_text.replace(self.tokenizer.bos_token, '')
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding='max_length',
            return_tensors='pt',
            padding_side='right'
        )
        
        input_ids: Int[Tensor, "seq_len"] = encoding["input_ids"][0]
        attention_mask: Int[Tensor, "seq_len"] = encoding["attention_mask"][0]
        
        labels, weights, pos_spans, neg_spans = self._compute_positional_labels(
            input_ids=input_ids,
            item=item
        )
        
        input_str: str = self.tokenizer.decode(input_ids)
        assistant_tokens_slice = find_assistant_tokens_slice(input_ids, input_str, self.tokenizer)
        completion_start_idx = assistant_tokens_slice.stop
        
        lm_labels = input_ids.clone()
        lm_labels[:completion_start_idx] = -100  # ignore all tokens in the prompt
        lm_labels[attention_mask == 0] = -100  # ignore padding tokens
        
        return {
            "input_ids": input_ids,  # Int[Tensor, "seq_len"]
            "attention_mask": attention_mask,  # Int[Tensor, "seq_len"]
            "classification_labels": labels,  # Float[Tensor, "seq_len"]
            "classification_weights": weights,  # Float[Tensor, "seq_len"]
            "pos_spans": pos_spans,  # List[List[int]]
            "neg_spans": neg_spans,  # List[List[int]]
            "lm_labels": lm_labels,  # Int[Tensor, "seq_len"]
        }
    
    def print_token_labels(
        self,
        input_ids: torch.Tensor,
        positive_indices: List[int],
        negative_indices: List[int],
        ignore_indices: List[int],
        spans: List[AnnotatedSpan]
    ):
        """Debug method to print how tokens have been labeled."""

        tokens = [self.tokenizer.decode(tok) for tok in input_ids]

        print(f"================================================")
        print(f"Number of spans: {len(spans)}")
        print(f"Number of non-factual (hallucinated) spans: {len([f for f in spans if f.label == 1.0])}")
        print(f"Number of N/A spans: {len([f for f in spans if f.label == -100])}")
        print(f"Number of factual spans: {len([f for f in spans if f.label == 0.0])}")
        print(f"Legend: red - positive, green - negative, blue - ignored")

        for i, token in enumerate(tokens):
            if token == self.tokenizer.eos_token:
                continue

            if i in positive_indices:
                print(colored(token, 'red'), end='')
            elif i in negative_indices:
                print(colored(token, 'green'), end='')
            elif i in ignore_indices:
                print(colored(token, 'blue'), end='')
            else:
                print(token, end='')

        print(f"================================================")
    
    def _compute_positional_labels(
        self,
        input_ids: torch.Tensor,
        item: ProbingItem
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], List[List[int]]]:
        """
        Computes positional labels for a sequence of tokens based on annotated spans.
        
        For each span in the example:
        - If it's a hallucination (label 1.0): Sets span tokens to 1.0 and nearby tokens within
          ignore_buffer to -100.0
        - If it's a supported fact (label 0.0): Sets span tokens to 0.0 and nearby tokens within
          ignore_buffer to -100.0
        - If it's unlabeled/undecided (label -100.0): Sets span tokens to -100.0
        
        Args:
            input_ids: Input token IDs
            item: ProbingItem containing the spans and their labels
            
        Returns:
            Tuple of (labels, weights, pos_spans, neg_spans)
        """
        input_str: str = self.tokenizer.decode(input_ids)
        completion: str = item.completion
        
        positive_indices: List[int] = []    # indices of hallucinated spans
        negative_indices: List[int] = []    # indices of supported spans
        ignore_indices: List[int] = []      # indices to ignore in training
        
        positive_spans: List[List[int]] = []
        negative_spans: List[List[int]] = []
        
        def get_nearby_indices(span_indices: List[int]) -> List[int]:
            left_window = list(range(max(0, span_indices[0] - self.config.ignore_buffer), span_indices[0]))
            right_window = list(range(span_indices[-1] + 1, min(len(input_ids), span_indices[-1] + 1 + self.config.ignore_buffer)))
            return left_window + right_window
        
        # Find assistant tokens slice to know where to start looking for spans
        assistant_tokens_slice = find_assistant_tokens_slice(
            input_ids,
            input_str,
            self.tokenizer
        )
        completion_start_idx = assistant_tokens_slice.stop
        cur_idx = assistant_tokens_slice.stop
        
        # Sort spans by their index in the text
        spans = sorted(item.spans, key=lambda x: x.index)
        
        for span in spans:
            if span.span not in input_str:
                self._num_skipped_spans += 1
                continue
            
            try:
                # First try to find the span after the assistant tokens
                positions_slice = find_string_in_tokens(span.span, input_ids[cur_idx:], self.tokenizer)
                positions_slice = slice(positions_slice.start + cur_idx, positions_slice.stop + cur_idx)
            except (AssertionError, ValueError):
                try:
                    # If not found, try the whole input_ids
                    print(f"Repeating position_slice search on all tokens after failing to find span {repr(span.span)} in input_ids[cur_idx:]: {repr(self.tokenizer.decode(input_ids[cur_idx:]))[:50]}...")
                    positions_slice = find_string_in_tokens(span.span, input_ids, self.tokenizer)
                except (AssertionError, ValueError) as e:
                    print(f"Span {repr(span.span)} not found in input_ids, skipping entity")
                    self._num_skipped_spans += 1
                    continue
            
            if positions_slice is None:
                continue
            
            span_indices = slice_to_list(positions_slice, len(input_ids))
            if not span_indices:
                continue
            
            cur_idx = positions_slice.start
            
            if self.config.last_span_token:
                # If last_span_token is true, only use the last token of the span
                span_indices = [span_indices[-1]]
            
            # Get indices of tokens to ignore around this span
            nearby_indices = get_nearby_indices(span_indices)
            
            if span.label == 1.0:  # Hallucination
                positive_indices.extend(span_indices)
                ignore_indices.extend(nearby_indices)
                positive_spans.append([span_indices[0], span_indices[-1]])
            elif span.label == 0.0:  # Supported
                negative_indices.extend(span_indices)
                negative_spans.append([span_indices[0], span_indices[-1]])
            else:  # -100.0 (ignored)
                ignore_indices.extend(span_indices)
            
            self._num_added_spans += 1
        
        # Remove duplicates and sort
        positive_indices = sorted(list(set(positive_indices)))
        negative_indices = sorted(list(set(negative_indices) - set(positive_indices)))
        ignore_indices = sorted(list(set(ignore_indices) - set(positive_indices) - set(negative_indices)))
        
        # Initialize labels and weights
        default_label = -100.0 if self.config.default_ignore else 0.0
        labels = torch.full((len(input_ids),), default_label, dtype=torch.float32)
        
        # Set labels and weights
        labels[input_ids == self.tokenizer.pad_token_id] = -100.0
        labels[:completion_start_idx] = -100.0
        labels[ignore_indices] = -100.0
        labels[positive_indices] = 1.0
        labels[negative_indices] = 0.0
        
        weights = torch.full((len(input_ids),), 1.0, dtype=torch.float32)
        weights[ignore_indices] = 0.0  # N/A weight
        weights[positive_indices] = self.config.pos_weight
        weights[negative_indices] = self.config.neg_weight
        
        if self.debug_mode:
            self.print_token_labels(input_ids, positive_indices, negative_indices, ignore_indices, spans)
        
        return labels, weights, positive_spans, negative_spans
    
    def _shuffle_items(self):
        """Shuffle the items using the configured seed."""
        random.seed(self.config.seed)
        random.shuffle(self.items)
        random.seed(self.config.seed)
        random.shuffle(self.processed_items)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        if self.config.process_on_the_fly and self.processed_items[idx] is None:
            self.processed_items[idx] = self._process_item(self.items[idx])
        
        return self.processed_items[idx]
    
    def __add__(self, other):
        """
        Concatenate two TokenizedProbingDataset instances.
        
        Args:
            other: Another TokenizedProbingDataset instance to concatenate
            
        Returns:
            TokenizedProbingDataset: A new dataset containing items from both
        """
        if not isinstance(other, TokenizedProbingDataset):
            raise TypeError(f"Can only concatenate with another TokenizedProbingDataset, got {type(other)}")
        
        if self.config.max_length != other.config.max_length:
            raise ValueError("Can't concatenate datasets of different token lengths")
        
        if self.config.shuffle != other.config.shuffle:
            raise ValueError("Can't concatenate datasets if one of them (but not the other) are shuffled")
        
        # Create a new dataset with combined items
        combined_items = self.items + other.items
        combined_processed_items = self.processed_items + other.processed_items
        
        # Use the configuration from the first dataset
        new_dataset = TokenizedProbingDataset(
            items=[],  # we don't want to recompute everything again
            tokenizer=self.tokenizer,
            config=self.config,
        )
        
        new_dataset.items = combined_items
        new_dataset.processed_items = combined_processed_items
        
        if self.config.shuffle:
            new_dataset._shuffle_items()
        
        return new_dataset
    
    def __radd__(self, other):
        return self.__add__(other)


def tokenized_probing_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader that handles variable-length tokenized sequences.
    
    Args:
        batch: List of tokenized dataset items
    
    Returns:
        Batched dictionary with padded sequences
    """
    # Find max length in batch
    max_len = max(len(item["input_ids"]) for item in batch)
    
    # Initialize batched tensors
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    classification_labels = torch.full((batch_size, max_len), -100.0, dtype=torch.float32)
    classification_weights = torch.zeros((batch_size, max_len), dtype=torch.float32)
    lm_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    # Lists for spans
    pos_spans = []
    neg_spans = []
    
    # Fill in the batch
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        classification_labels[i, :seq_len] = item["classification_labels"]
        classification_weights[i, :seq_len] = item["classification_weights"]
        lm_labels[i, :seq_len] = item["lm_labels"]
        pos_spans.append(item["pos_spans"])
        neg_spans.append(item["neg_spans"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "classification_labels": classification_labels,
        "classification_weights": classification_weights,
        "lm_labels": lm_labels,
        "pos_spans": pos_spans,
        "neg_spans": neg_spans,
    }


def create_probing_dataset(
    cfg: TokenizedProbingDatasetConfig,
    tokenizer: AutoTokenizer
) -> TokenizedProbingDataset:
    """
    Create a probing dataset from configuration.
    
    This loads the dataset from HuggingFace and processes it using the
    appropriate dataset-specific preparation function.
    """
    # Lazy import to avoid circular dependency
    
    # Load dataset from HuggingFace
    if cfg.subset:
        raw_hf_dataset = datasets.load_dataset(cfg.hf_repo, cfg.subset, split=cfg.split)
    else:
        raw_hf_dataset = datasets.load_dataset(cfg.hf_repo, split=cfg.split)

    # Handle max_num_samples
    if cfg.max_num_samples is not None:
        if cfg.shuffle:
            print(f"Shuffling and truncating dataset to {cfg.max_num_samples} / {len(raw_hf_dataset)} samples")
            assert cfg.seed is not None, "Seed must be provided if shuffle is True"
            raw_hf_dataset = raw_hf_dataset.shuffle(seed=cfg.seed)
        else:
            print(f"Truncating dataset to first {cfg.max_num_samples} / {len(raw_hf_dataset)} samples")
        raw_hf_dataset = raw_hf_dataset.select(range(min(cfg.max_num_samples, len(raw_hf_dataset))))

    print(f"Loading dataset: {cfg.hf_repo} | {cfg.subset} | {cfg.split}")

    # Get appropriate preparation function
    prepare_function = get_prepare_function(cfg.hf_repo, cfg.subset)

    # Convert HF dataset to list of probing items
    probing_items: List[ProbingItem] = prepare_function(raw_hf_dataset)

    tokenized_probing_dataset = TokenizedProbingDataset(
        items=probing_items,
        config=cfg,
        tokenizer=tokenizer
    )

    return tokenized_probing_dataset