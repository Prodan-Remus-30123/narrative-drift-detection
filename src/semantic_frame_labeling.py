"""
semantic_frame_labeling.py

LLM-based semantic labeling for latent narrative frames. Uses
whichever backend llm_backend.py is configured for (local Ollama by
default, or the Hugging Face Inference API when LLM_BACKEND=hf).
"""

from frame_cache_db import (
    get_cached_verb_frame_label,
    store_cached_verb_frame_label,
    make_frame_key
)
from llm_backend import generate_text, extract_json

MODEL_NAME = None



def build_frame_label_prompt(frame_verbs):
    verbs_text = ", ".join(frame_verbs)

    return f"""
You are analyzing latent narrative frames in news media.

Given the following action verbs extracted from clustered news narratives:

{verbs_text}

Generate:
1. A short semantic frame label (2-5 words)
2. A one-sentence interpretation

Return ONLY valid JSON in this format:

{{
    "frame_label": "...",
    "frame_description": "..."
}}
"""


def label_latent_frame(frame_verbs, model=MODEL_NAME):
    prompt = build_frame_label_prompt(frame_verbs)
    cache_key = make_frame_key(frame_verbs)
    cached = get_cached_verb_frame_label(cache_key)

    if cached is not None:
        return cached

    try:
        raw = generate_text(prompt, model=model)
        parsed = extract_json(raw)

        if parsed is None:
            return {
                "frame_label": "unlabeled_frame",
                "frame_description": f"Could not parse LLM response: {raw!r}"
            }

        result = {
            "frame_label": parsed.get("frame_label", "unknown_frame"),
            "frame_description": parsed.get("frame_description", "")
        }
        store_cached_verb_frame_label(cache_key, result)
        return result

    except Exception as e:

        return {
            "frame_label":
                "unlabeled_frame",

            "frame_description":
                f"LLM labeling failed: {str(e)}"
        }


def label_all_latent_frames(
    latent_result,
    model=MODEL_NAME
):
    labeled_frames = {}

    clusters = latent_result[
        "clusters"
    ]

    for frame_id, frame_data in clusters.items():

        verbs = frame_data.get(
            "verbs",
            []
        )

        label_result = label_latent_frame(
            verbs,
            model=model
        )

        labeled_frames[frame_id] = {

            "verbs": verbs,

            "frame_label":
                label_result[
                    "frame_label"
                ],

            "frame_description":
                label_result[
                    "frame_description"
                ]
        }

    return labeled_frames


def print_labeled_frames(
    labeled_frames,
    top_n=10
):
    for frame_id, frame_data in list(labeled_frames.items())[:top_n]:

        print(f"\nFrame {frame_id}")

        print(
            f"Label: "
            f"{frame_data['frame_label']}"
        )

        print(
            f"Description: "
            f"{frame_data['frame_description']}"
        )

        print(
            f"Top verbs: "
            f"{frame_data['verbs'][:10]}"
        )