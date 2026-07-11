"""
Embedding generation using Sentence-BERT.

Responsible for:
- Loading embedding model
- Encoding documents into vector representations
- Aggregating embeddings per time window
"""

import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper for Sentence-BERT embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        # EMBEDDING_DEVICE lets the Hugging Face Space demo force "cpu"
        # explicitly: that Space is provisioned with ZeroGPU hardware,
        # whose CUDA-interception (see src/llm_backend.py's sibling
        # `spaces` import in app.py) only works inside a
        # `@spaces.GPU`-decorated call, and otherwise raises rather
        # than silently falling back. Local dissertation runs are
        # unaffected (device stays None -> sentence-transformers'
        # normal auto-detection).
        device = device or os.environ.get("EMBEDDING_DEVICE")
        self.model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts (List[str]): List of documents.

        Returns:
            np.ndarray: Document embeddings.
        """
        return self.model.encode(texts)

    @staticmethod
    def aggregate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Aggregate embeddings for a time window (mean pooling).

        Args:
            embeddings (np.ndarray): Document embeddings.

        Returns:
            np.ndarray: Aggregated vector.
        """
        return np.mean(embeddings, axis=0)