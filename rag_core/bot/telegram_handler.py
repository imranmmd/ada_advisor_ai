import asyncio
import logging
from typing import List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag_core.bot import error_handler
from rag_core.bot.message_parser import ParsedMessage, parse_message
from rag_core.bot.rag_interface import run_rag
from rag_core.bot.response_formatter import FormattedResponse, format_result

LOGGER = logging.getLogger(__name__)

ALLOWED_UPDATES = ["message", "edited_message"]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return
    await message.reply_text(
        "Hi! Send me a question and I'll search the knowledge base for you."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return
    await message.reply_text("Just send any text question and I will do the rest.")


async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    parsed: ParsedMessage | None = parse_message(update)
    if not parsed:
        await update.effective_message.reply_text("Please send a text question.")
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        rag_result = await asyncio.to_thread(run_rag, parsed.text, parsed.session_id)
    except Exception as exc:  # pragma: no cover - runtime guard
        LOGGER.exception("RAG pipeline failed")
        await error_handler.send_user_fallback(update)
        await error_handler.notify_admin(context, exc, update)
        return

    responses: List[FormattedResponse] = format_result(rag_result)
    for resp in responses:
        await update.effective_message.reply_text(
            resp.text,
            parse_mode=resp.parse_mode,
            disable_web_page_preview=True,
        )


def build_application(token: str) -> Application:
    """Create a Telegram Application with handlers attached."""
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat))
    app.add_error_handler(error_handler.handle_error)

    return app


__all__ = ["ALLOWED_UPDATES", "build_application"]
