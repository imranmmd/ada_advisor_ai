import re
from dataclasses import dataclass
from typing import List, Optional

from telegram import Message, Update

MAX_TEXT_LENGTH = 3000


@dataclass
class ParsedMessage:
    text: str
    session_id: str
    tokens: List[str]
    raw_text: str


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_commands(message: Message, text: str) -> str:
    """Remove command prefixes like /start from the incoming text."""
    if not message.entities:
        return text
    cleaned = text
    for entity in message.entities:
        if entity.type != "bot_command":
            continue
        try:
            command = message.parse_entity(entity)
        except Exception:
            command = None
        if command:
            cleaned = cleaned.replace(command, "").strip()
    return cleaned


def _strip_mentions(text: str) -> str:
    """Remove leading @mentions so the prompt is clean for the model."""
    return re.sub(r"^@[\w_]+\s*", "", text)


def parse_message(update: Update) -> Optional[ParsedMessage]:
    """Extract, clean, and tokenize the text payload from a Telegram update."""
    message = update.effective_message
    if not message:
        return None

    raw_text = (message.text or message.caption or "").strip()
    if not raw_text:
        return None

    cleaned = _strip_mentions(_strip_commands(message, raw_text))
    cleaned = _normalize_whitespace(cleaned)
    if not cleaned:
        return None

    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH].rstrip()

    tokens = [tok for tok in cleaned.split(" ") if tok]
    if not tokens:
        return None

    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    session_id = str(chat_id or user_id or "anonymous")

    return ParsedMessage(
        text=cleaned,
        session_id=session_id,
        tokens=tokens,
        raw_text=raw_text,
    )


__all__ = ["ParsedMessage", "parse_message"]
