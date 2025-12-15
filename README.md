# ada_advisor_ai

## Telegram bot
- Set environment: `OPENAI_API_KEY` (existing RAG flow) and `TELEGRAM_BOT_TOKEN`.
- Optional: `TELEGRAM_WEBHOOK_URL`/`TELEGRAM_WEBHOOK_PATH` to enable webhooks (defaults to polling), `TELEGRAM_ADMIN_CHAT_ID` for error alerts, `TELEGRAM_RAG_TOP_K` to override retrieval depth.
- Run locally with polling: `python -m rag_core.bot.main`.
- For webhooks: set the webhook URL envs, expose the chosen port, and start the bot (`PORT`/`TELEGRAM_WEBHOOK_PORT` default to 8443).
