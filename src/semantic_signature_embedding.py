"""
semantic_signature_embedding.py

Adds semantic frame meaning
to narrative signatures.
"""

import numpy as np

from embedding_model_registry import (
    get_embedding_model
)


def build_frame_semantic_embedding(signature):
    """
    Builds one embedding
    from the signature's
    top semantic frames.
    """

    frame_labels = signature.get(
        "top_frame_labels",
        []
    )

    frame_descriptions = signature.get(
        "top_frame_descriptions",
        []
    )

    semantic_document = "\n".join(
        frame_labels + frame_descriptions
    )

    if not frame_labels:
        return None

    model = get_embedding_model()

    embeddings = model.encode_documents([semantic_document])

    if len(embeddings) == 0:
        return None

    return embeddings[0]


def build_semantic_signature_embeddings(
    narrative_signatures
):
    """
    source -> embedding
    """

    results = {}

    for signature in narrative_signatures:

        source = signature["source"]

        results[source] = (
            build_frame_semantic_embedding(
                signature
            )
        )

    return results