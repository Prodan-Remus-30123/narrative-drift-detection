"""
Text preprocessing utilities for narrative drift analysis.

This module handles:
- Basic text cleaning
- Optional normalization
- Sentence segmentation 
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Perform basic cleaning on raw article text.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_corpus(texts: List[str]) -> List[str]:
    """
    Apply preprocessing to a list of documents.

    Args:
        texts (List[str]): List of raw article texts.

    Returns:
        List[str]: List of cleaned texts.
    """
    return [clean_text(t) for t in texts]