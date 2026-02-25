"""
Embedding generation using Sentence-BERT.

Responsible for:
- Loading embedding model
- Encoding documents into vector representations
- Aggregating embeddings per time window
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper for Sentence-BERT embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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