"""Dataset converters for different HuggingFace dataset formats to probing format."""

from typing import Callable, Dict, List, Optional

from datasets import Dataset

from probe.types import AnnotatedSpan, ProbingItem


# Mapping from text labels to numeric values for probe training
_MAP_LABEL_TO_SCALAR = {
    'Not Supported': 1.0,
    'NS': 1.0,  # the probe should output 1.0 on text containing unsupported claims
    'Insufficient Information': 1.0,  # the probe should also output 1.0 if the label is 'Insufficient Information'
    'Supported': 0.0,
    'S': 0.0,
    'N/A': -100.0,
    None: -100.0
}

def prepare_longform_dataset(dataset: Dataset) -> List[ProbingItem]:
    """Prepare dataset from the one-shot pipeline labeling format."""
    probing_items: List[ProbingItem] = []

    for hf_item in dataset:
        prompt = hf_item['conversation'][-2]['content']
        completion = hf_item['conversation'][-1]['content']
        annotations: List[dict] = hf_item['annotations']

        annotated_spans: List[AnnotatedSpan] = []

        for entity in annotations:
            if entity is None or 'index' not in entity or not isinstance(entity['index'], int):
                continue

            entity_text = entity['span']
            label = entity['label']
            idx = entity['index']
            
            if idx is None:
                print(f"Entity {repr(entity_text)}'s idx set to None, discarding entity")
                continue
            elif not entity_text or entity_text not in completion:
                print(f"Entity {repr(entity_text)} not found in completion, discarding entity")
                continue

            annotated_spans.append(
                AnnotatedSpan(
                    span=entity_text,
                    label=_MAP_LABEL_TO_SCALAR[label],
                    index=idx
                )
            )

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_longform_dataset_old_format(dataset: Dataset) -> List[ProbingItem]:
    """Prepare dataset from the one-shot pipeline labeling format."""
    probing_items: List[ProbingItem] = []

    for hf_item in dataset:
        prompt = hf_item['conversation'][0]['content']
        completion = hf_item['completion'] if 'completion' in hf_item else hf_item['conversation'][-1]['content']
        annotations: List[dict] = hf_item['verified_entities']

        annotated_spans: List[AnnotatedSpan] = []

        for entity in annotations:
            if entity is None or 'idx' not in entity or not isinstance(entity['idx'], int):
                continue

            entity_text = entity['text']
            label = entity['label']
            idx = entity['idx']
            
            if idx is None:
                print(f"Entity {repr(entity_text)}'s idx set to None, discarding entity")
                continue
            elif not entity_text or entity_text not in completion:
                print(f"Entity {repr(entity_text)} not found in completion, discarding entity")
                continue

            annotated_spans.append(
                AnnotatedSpan(
                    span=entity_text,
                    label=_MAP_LABEL_TO_SCALAR[label],
                    index=idx
                )
            )

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_triviaqa(dataset: Dataset) -> List[ProbingItem]:
    """
    Pre-processes TriviaQA dataset.
    The greedy completion (labeled by an LLM) is at `gt_completion`
    The label is at `llm_judge_label` and it's a string containing `S`, `NS`, `N/A` or some undefined string
    The annotated spans will be the *whole completion*
    """
    assert 'question' in dataset[0] or 'conversation' in dataset[0]

    LABEL_FIELD: str = 'llm_judge_label' if 'llm_judge_label' in dataset.features else 'label'
    COMPLETION_FIELD: str = 'gt_completion'
    VALID_LABELS: List[str] = ['S', 'NS', 'N/A']
    EXACT_ANSWER_FIELD: str = 'exact_answer'

    probing_items = []
    for item in dataset:
        if item[LABEL_FIELD] not in VALID_LABELS:
            print(f"Invalid label {item[LABEL_FIELD]} for item, skipping")
            continue

        prompt = item['question'] if 'question' in item else item['conversation'][0]['content']
        completion = item[COMPLETION_FIELD]
        exact_answer = item[EXACT_ANSWER_FIELD] if EXACT_ANSWER_FIELD in item else ""

        if exact_answer is None or exact_answer not in completion:
            print(f"Exact answer {repr(exact_answer)} not found in completion {repr(completion)}")
            return None

        exact_answer_start_idx = completion.find(exact_answer)
        
        # The whole completion is labeled with the given label
        label_value: float = _MAP_LABEL_TO_SCALAR[item[LABEL_FIELD]]
        
        annotated_spans = [
            AnnotatedSpan(
                span=exact_answer,
                label=label_value,
                index=exact_answer_start_idx
            )
        ]

        probing_items.append(
            ProbingItem(
                prompt=prompt,
                completion=completion,
                spans=annotated_spans
            )
        )

    return probing_items


def prepare_synthetic(dataset: Dataset) -> List[ProbingItem]:
    """Loads the synthetic dataset from the hub."""
    FIELD = 'probing_item_with_hallucinations'

    probing_items = []
    for i, item in enumerate(dataset):
        probing_item = item[FIELD]
        annotated_spans = [
            AnnotatedSpan(
                span=span['text'],
                label=span['label'],
                index=span['start_idx']
            )
            for span in probing_item['spans']
        ]

        # Sort spans by their index in the text
        completion = probing_item['completion']

        if len(completion) <= 500:
            print(f"For item {i} completion is too short ({len(completion)} characters): {repr(completion)}")
            continue

        annotated_spans = sorted(annotated_spans, key=lambda x: x.index)

        if not all(completion[span.index:span.index+len(span.span)] == span.span for span in annotated_spans):
            print(f"For item {i} spans are not aligned with the completion")
            for span in annotated_spans:
                if completion[span.index:span.index+len(span.span)] != span.span:
                    print(f"- Span: {span.span} | {span.index} | {span.label}")
            continue

        probing_items.append(ProbingItem(
            prompt=probing_item['prompt'],
            completion=probing_item['completion'],
            spans=annotated_spans
        ))
    return probing_items


def get_prepare_function(
    hf_repo: str,
    subset: Optional[str] = None
) -> Callable[[Dataset], List[ProbingItem]]:
    """Get the appropriate preparation function based on dataset name."""
    if 'one_shot_pipeline' in str(subset) or 'hallucination-heads' in hf_repo:
        return prepare_longform_dataset_old_format
    elif 'modified' in str(subset) and 'synthetic-hallucinations' in hf_repo:
        return prepare_synthetic
    elif 'trivia_qa' in str(subset) or 'triviaqa' in hf_repo:
        return prepare_triviaqa
    else:
        # Default to one-shot pipeline format
        return prepare_longform_dataset


