"""Utilities for parsing and validating JSON from LLM responses."""

import re
from typing import Any, List, Type, TypeVar, Union

from pydantic import BaseModel, parse_obj_as
from pydantic_core import from_json

from utils.string_utils import normalize_text


T = TypeVar('T', bound=BaseModel)


def parse_and_validate_json(
    llm_response: str, 
    schema: Union[Type[BaseModel], Any], 
    allow_partial: bool = False
) -> Any:
    """
    Parse and validate JSON from an LLM response.
    
    This function:
    1. Normalizes the text to handle encoding issues
    2. Removes markdown code fences
    3. Extracts the JSON structure
    4. Parses and validates against the provided schema
    
    Args:
        llm_response: Raw LLM response that may contain JSON with surrounding text
        schema: Pydantic model or type hint for validation (e.g., MyModel or List[MyModel])
        allow_partial: Whether to allow partial JSON parsing (useful for truncated responses)
        
    Returns:
        Parsed and validated object(s) according to the schema
        
    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    # Normalize text to handle control characters
    cleaned_response = normalize_text(llm_response)
    
    # Remove markdown code fences (```json or ```)
    cleaned_response = re.sub(
        r"```(?:json)?", "", 
        cleaned_response, 
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    # Find JSON structure (object or array)
    json_match = re.search(r'(\{.*\}|\[.*\])', cleaned_response, flags=re.DOTALL)
    if not json_match:
        raise ValueError(
            f"No valid JSON object or array found in response: {llm_response[:200]}..."
        )

    json_str = json_match.group(0).strip()

    # remove markdown links
        
    pattern = r"\(\[(.*?)\]\((.*?)\)\)"
    replacement = r"Source: \2"

    json_str_clean = re.sub(pattern, replacement, json_str)

    
    try:
        # Parse JSON
        parsed = from_json(json_str_clean, allow_partial=allow_partial)
        
        # Validate against schema
        validated_data = parse_obj_as(schema, parsed)
        
        return validated_data
        
    except Exception as e:
        raise ValueError(
            f"Error parsing/validating JSON: {e}\n"
            f"JSON string: {json_str_clean[:200]}..."
        ) from e


def validate_dicts_to_pydantic(
    dicts: List[dict],
    model: Type[T],
    skip_invalid: bool = False
) -> List[T]:
    """
    Validate a list of dictionaries against a Pydantic model.
    
    Args:
        dicts: List of dictionaries to validate
        model: Pydantic model class to validate against
        skip_invalid: If True, skip invalid items instead of raising an error
        
    Returns:
        List of validated Pydantic model instances
        
    Raises:
        ValueError: If skip_invalid is False and validation fails for any item
    """
    validated = []
    
    for i, item_dict in enumerate(dicts):
        try:
            validated_item = model.model_validate(item_dict)
            validated.append(validated_item)
        except Exception as e:
            if skip_invalid:
                # Silently skip invalid items
                continue
            else:
                raise ValueError(
                    f"Validation failed for item {i}: {e}"
                ) from e
    
    return validated