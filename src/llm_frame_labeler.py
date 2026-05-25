"""
llm_frame_labeler.py

Ollama-based semantic labeling for latent narrative frames.
"""

import json
import requests
import hashlib
import numpy as np
from embeddings import EmbeddingModel
from frame_label_cache import (get_cached_label, store_cached_label)
from frame_cache_db import (
    initialize_frame_cache,
    get_cached_frame,
    store_cached_frame
)
from embedding_model_registry import get_embedding_model


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def compute_cluster_hash(
    verbs
):
    """
    Compute semantic centroid hash.
    """

    model = get_embedding_model()

    embeddings = np.array(
        model.encode_documents(verbs)
    )

    centroid = np.mean(
        embeddings,
        axis=0
    )

    rounded = np.round(
        centroid,
        2
    )

    hash_string = rounded.tobytes()

    return hashlib.md5(
        hash_string
    ).hexdigest()

def label_latent_frame(verbs,entity=None, source=None, transition=None, model=DEFAULT_MODEL):
    initialize_frame_cache()
    frame_hash = compute_cluster_hash(verbs)
    cached = get_cached_frame(frame_hash)

    if cached is not None:
        return cached
    
    prompt = build_frame_label_prompt(
        verbs=verbs,
        entity=entity,
        source=source,
        transition=transition
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=120
    )

    response.raise_for_status()

    raw = response.json()["response"]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "label": "unlabeled frame",
            "description": raw,
            "confidence": "low"
        }

    result = {
        "label": parsed.get("label", "unlabeled frame"),
        "description": parsed.get("description", ""),
        "confidence": parsed.get("confidence", "medium")
    }
    store_cached_frame(frame_hash, result)
    return result


def build_frame_label_prompt(
    verbs,
    entity=None,
    source=None,
    transition=None
):
    return f"""
You are labeling a latent narrative frame discovered from news coverage.

Entity: {entity}
Source: {source}
Transition: {transition}

Verbs in this latent frame:
{verbs}

Task:
Give a short, general, topic-independent narrative frame label.

Rules:
- Do not simply repeat the verbs.
- Do not mention "cluster".
- Prefer labels like:
  "public health crisis framing",
  "geopolitical conflict framing",
  "institutional response framing",
  "economic disruption framing",
  "political blame framing",
  "recovery framing",
  "risk escalation framing".
- Keep the label under 6 words.
- Give a one-sentence description.
- Return valid JSON only.

JSON format:
{{
  "label": "...",
  "description": "...",
  "confidence": "low|medium|high"
}}
"""