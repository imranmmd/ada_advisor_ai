import logging
import os

from rag_core.bot.telegram_handler import ALLOWED_UPDATES, build_application

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _compose_webhook_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path.lstrip("/")
    return f"{base}/{suffix}" if suffix else base


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required to run the bot.")

    webhook_base = os.getenv("TELEGRAM_WEBHOOK_URL")
    webhook_path = os.getenv("TELEGRAM_WEBHOOK_PATH", "/telegram")
    listen_addr = os.getenv("TELEGRAM_WEBHOOK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("TELEGRAM_WEBHOOK_PORT", "8443")))

    app = build_application(token)

    if webhook_base:
        webhook_url = _compose_webhook_url(webhook_base, webhook_path)
        LOGGER.info("Starting Telegram bot via webhook at %s", webhook_url)
        app.run_webhook(
            listen=listen_addr,
            port=port,
            url_path=webhook_path.lstrip("/"),
            webhook_url=webhook_url,
            allowed_updates=ALLOWED_UPDATES,
        )
    else:
        LOGGER.info("Starting Telegram bot in polling mode")
        app.run_polling(allowed_updates=ALLOWED_UPDATES)


if __name__ == "__main__":
    main()
