"""
Simple embedding wrapper with retries and batch support.
"""

import time
from typing import List, Optional, Sequence

from openai import OpenAI

from config.settings import OPENAI_API_KEY

DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_BATCH_SIZE = 16


class EmbeddingFunction:
    """
    Minimal embedding helper that:
    - wraps the OpenAI embeddings API
    - retries transient failures
    - handles batch processing with chunking
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        client: Optional[OpenAI] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(api_key=OPENAI_API_KEY)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size

    def embed_one(self, text: str) -> List[float]:
        """Embed a single string (returns a 1D list of floats)."""
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a batch of strings. Splits into smaller chunks to avoid payload limits.
        Returns a list of embeddings in the same order as input.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_with_retry(chunk))
        return embeddings

    def _embed_with_retry(self, texts: Sequence[str]) -> List[List[float]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=list(texts))
                return [list(item.embedding) for item in resp.data]
            except Exception as exc:  # broad catch to handle transient API/network errors
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay)
        raise RuntimeError(f"Failed to embed texts after {self.max_retries} attempts") from last_error
