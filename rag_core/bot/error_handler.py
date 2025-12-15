import logging
import os
from typing import Optional

from telegram import Update
from telegram.ext import ContextTypes

LOGGER = logging.getLogger(__name__)

USER_FALLBACK_MESSAGE = "Sorry, something went wrong. Please try again."


async def send_user_fallback(update: Update) -> None:
    message = update.effective_message
    if message:
        await message.reply_text(USER_FALLBACK_MESSAGE)


async def notify_admin(
    context: ContextTypes.DEFAULT_TYPE,
    error: Exception,
    update: Optional[Update] = None,
) -> None:
    admin_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
    if not admin_chat_id:
        return

    details = [f"Telegram bot error: {error}"]
    if update and update.effective_user:
        details.append(f"User: {update.effective_user.id}")
    if update and update.effective_chat:
        details.append(f"Chat: {update.effective_chat.id}")

    try:
        await context.bot.send_message(
            chat_id=int(admin_chat_id),
            text="\n".join(details)[:3500],
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        LOGGER.debug("Could not notify admin: %s", exc)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler hooked into the Application."""
    LOGGER.exception("Unhandled Telegram bot error", exc_info=context.error)
    if isinstance(update, Update):
        await send_user_fallback(update)
        if context.error:
            await notify_admin(context, context.error, update)


__all__ = ["handle_error", "notify_admin", "send_user_fallback"]
