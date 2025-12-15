from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from config.settings import OPENAI_API_KEY


class ChatModel(Protocol):
    """Narrow chat interface to support dependency inversion and testing."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - protocol
        ...


class OpenAIChatModel:
    """Thin wrapper around the OpenAI chat API."""

    def __init__(self, model: str, client: OpenAI | None = None) -> None:
        self._model = model
        self._client = client or OpenAI(api_key=OPENAI_API_KEY)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()


__all__ = ["ChatModel", "OpenAIChatModel"]