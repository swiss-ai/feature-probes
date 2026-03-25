import re
import math
from typing import Any, Dict, Type, Union, Optional, Tuple
from pydantic import BaseModel, parse_obj_as
from pydantic_core import from_json
from rouge_score import rouge_scorer

ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def normalize_text(text: str) -> str:
    """
    Normalize text by removing control characters and standardizing Unicode.
    
    Args:
        text: Raw text that may contain control characters or Unicode variants
        
    Returns:
        Normalized text with standard characters
    """
    # Remove control characters except newline, tab, and carriage return
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Replace Unicode spaces with regular spaces
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
    
    # Replace Unicode quotes with standard quotes
    text = re.sub(r'[\u201C\u201D\u2018\u2019\u201E\u201F\u2039\u203A\u00AB\u00BB]', '"', text)
    
    # Replace Unicode dashes with standard dash
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text

def normalize_for_matching(text: str) -> str:
    """Normalize text for matching - quotes, whitespace, punctuation."""
    # One-liner version:
    # return re.sub(r'\s+', ' ', re.sub(r'[\'"`''""‛„:;()\[\]\-–—]|[.,](?!(?<=\d[.,])\d)', ' ', re.sub(r'(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text))).strip().lower()

    # More readable step-by-step version:
    # Step 1: Add spaces between letters and numbers (except after M/m/X/x for Roman numerals)
    text = re.sub(
        r'(?<=[^\W\d_])(?<![MmXx])(?=\d)|(?<=\d)(?=[^\W\d_])', ' ', text)

    # Step 2: Remove quotes and punctuation (but preserve decimals)
    # This combines quote removal with punctuation removal in one regex
    text = re.sub(r'[\'"`''""‛„:;()\[\]\-–—]|[.,](?!(?<=\d[.,])\d)', ' ', text)

    # Step 3: Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text

def trim_match_edges(query: str, match: str, normalize_text: bool = False) -> str:
    """
    Trim unnecessary characters from the beginning and end of a match
    while maintaining or improving recall.
    """
    if not match or match == query:
        return match
        
    # Get initial recall score
    if normalize_text:
        initial_recall = ROUGE_SCORER.score(normalize_for_matching(query), normalize_for_matching(match))['rougeL'].recall
    else:
        initial_recall = ROUGE_SCORER.score(query, match)['rougeL'].recall

    best_match = match
    best_recall = initial_recall
    
    # Try trimming from the beginning
    for i in range(1, min(len(match) // 2, 20)):  # Don't trim more than half or 20 chars
        trimmed = match[i:]
        if normalize_text:
            scores = ROUGE_SCORER.score(normalize_for_matching(query), normalize_for_matching(trimmed))
        else:
            scores = ROUGE_SCORER.score(query, trimmed)
        
        if scores['rougeL'].recall >= best_recall:
            best_match = trimmed
            best_recall = scores['rougeL'].recall
        else:
            break  # Stop if recall drops
    
    # Try trimming from the end
    match = best_match  # Start with best so far
    for i in range(1, min(len(match) // 2, 20)):
        trimmed = match[:-i]
        if normalize_text:
            scores = ROUGE_SCORER.score(normalize_for_matching(query), normalize_for_matching(trimmed))
        else:
            scores = ROUGE_SCORER.score(query, trimmed)
        
        if scores['rougeL'].recall >= best_recall:
            best_match = trimmed
            best_recall = scores['rougeL'].recall
        else:
            break  # Stop if recall drops
    
    return best_match

def find_closest_match(query: str, text: str, window_margin: int = 10, min_similarity: float = 0.9, normalize_text: bool = False) -> Optional[str]:
    """
    Finds the substring of `text` that best matches `query` using ROUGE-L similarity.
    Returns the exact substring from `text`.

    :param query: The query string (e.g., the extracted factual claim).
    :param text: The larger text in which we want to find the best matching substring.
    :param window_margin: The +/- margin around the query length for the sliding window.
    :return: The substring from `text` that best matches `query`.
    """
    if query in text:
        return query

    # We'll attempt to match substrings in `text` that are around len(query).
    query_len = len(query)

    # Precompute the best match info
    best_substring = ""
    best_score = -math.inf

    if normalize_text:
        query = normalize_for_matching(query)

    # Because we want to allow some variation (extra words, punctuation, etc.),
    # we use a window size = query_len +/- window_margin
    min_len = max(1, query_len - window_margin)
    max_len = min(len(text), query_len + window_margin)

    # Slide over the text and compare substrings
    for start_idx in range(len(text)):
        # We'll try multiple substring lengths in [min_len, max_len]
        for length in range(min_len, max_len + 1):
            end_idx = start_idx + length
            if end_idx > len(text):
                break  # no need to go further if we exceed text length

            candidate_substring = text[start_idx:end_idx].strip()

            if normalize_text:
                normalized_candidate_substring = normalize_for_matching(candidate_substring)
                scores = ROUGE_SCORER.score(query, normalized_candidate_substring)
            else:
                scores = ROUGE_SCORER.score(query, candidate_substring)

            rouge_l_score = scores['rougeL'].fmeasure  # f-measure of ROUGE-L

            if rouge_l_score > best_score:
                best_score = rouge_l_score
                best_substring = candidate_substring

            if best_score > 0.95:
                # Trim edges if not perfect
                best_substring = trim_match_edges(query, best_substring, normalize_text)
                assert best_substring in text, f"Best substring {best_substring} not in text {text}"
                return best_substring

    if best_score >= min_similarity:
        # Trim edges before returning
        best_substring = trim_match_edges(query, best_substring, normalize_text)
        assert best_substring in text
        return best_substring
    else:
        if not normalize_text:
            return find_closest_match(query, text, window_margin, min_similarity, normalize_text=True)
        return None


def try_matching_span_in_text(span: str, text: str, cur_idx: int = 0, min_similarity: float = 0.8) -> Tuple[Optional[str], Optional[int]]:
    """
    Try to match a span to a substring in the text.
    
    Args:
        span: The span text to match
        text: The full text to search in
        cur_idx: Current index to start searching from
        min_similarity: Minimum similarity threshold
        
    Returns:
        Tuple of (matched_text, index) or (None, None) if no match found
    """
    closest_match: Optional[str] = find_closest_match(span, text[cur_idx:])

    if closest_match is not None:
        span_idx = text[cur_idx:].index(closest_match)
        assert span_idx != -1, f"Span {repr(span)} not found in text[cur_idx:]: {repr(text[cur_idx:])}"
        return closest_match, cur_idx + span_idx

    # try searching for the span in the previous part of the text
    if cur_idx > 0:
        return try_matching_span_in_text(span, text, cur_idx=0, min_similarity=min_similarity)

    return None, None