"""
embedding_model_registry.py

Global singleton embedding model cache.
"""

from embeddings import EmbeddingModel


_MODEL = None


def get_embedding_model():

    global _MODEL

    if _MODEL is None:

        print(
            "\n[Embedding Registry] "
            "Loading global embedding model..."
        )

        _MODEL = EmbeddingModel()

    return _MODEL