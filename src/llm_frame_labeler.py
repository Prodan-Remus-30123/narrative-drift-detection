"""
llm_frame_labeler.py

LLM-based semantic labeling for latent narrative frames, keyed by an
entity/transition-conditioned verb cluster. Uses whichever backend
llm_backend.py is configured for (local Ollama by default, or the
Hugging Face Inference API when LLM_BACKEND=hf).
"""

import hashlib
import numpy as np
from frame_cache_db import (
    initialize_frame_cache,
    get_cached_frame,
    store_cached_frame
)
from embedding_model_registry import get_embedding_model
from llm_backend import generate_text, extract_json


DEFAULT_MODEL = None


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

def label_latent_frame(verbs, entity=None, source=None, transition=None, model=DEFAULT_MODEL):
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

    raw = generate_text(prompt, model=model, json_mode=True)
    parsed = extract_json(raw)

    if parsed is None:
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