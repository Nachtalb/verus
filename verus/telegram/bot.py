import asyncio
import logging
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import uvicorn
from quart import Quart
from telegram.ext import ApplicationBuilder, ContextTypes, PicklePersistence

from verus.const import TG_BASE_URL
from verus.db import setup_db
from verus.telegram._verusbot import VerusBot
from verus.telegram._veruscontext import VerusContext
from verus.telegram._webapp import Webapp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def amain() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="Directory all images / videos are stored in.")
    parser.add_argument("--default-tag", type=str, default="misc", help="Default tag for imported images / videos.")
    parser.add_argument("--import-folder", type=Path, help="Folder where to import images / videos from.")

    parser.add_argument("--telegram-token", required=True, help="Telegram bot token.")

    parser.add_argument(
        "--listen",
        default="0.0.0.0",
        help="Host to listen on for telegram and the webapp (default: 0.0.0.0)",
    )
    parser.add_argument("--port", type=int, default=8433, help="Port to listen on (default: 8433)")

    parser.add_argument(
        "--telegram-webhook-url",
        required=True,
        help="Webhook URL reported to Telegram, eg. https://example.com/bot",
    )
    parser.add_argument(
        "--telegram-path",
        default="/bot",
        help="Path for the Telegram webhook (default: /bot)",
    )
    parser.add_argument(
        "--webapp-path",
        default="/webapp",
        help="Path for the webapp (default: /webapp)",
    )

    # Typical local path: "http://localhost:8081/bot"
    parser.add_argument(
        "--telegram-api-server",
        default=TG_BASE_URL,
        help=f"Telegram BOT API server URL (default: {TG_BASE_URL})",
    )
    parser.add_argument(
        "--local-mode",
        default=False,
        action="store_true",
        help="Run the bot in local mode. Use when the Telegram Server provided with --telegram-api-server is started with --local-mode flag.",
    )
    args = parser.parse_args()

    # Initialize the database
    setup_db()

    # Initialize the bot
    verus_bot = VerusBot(
        dir=args.dir,
        default_tag=args.default_tag,
        import_folder=args.import_folder,
        local_mode=args.local_mode,
    )

    # Build the application
    context_types = ContextTypes(context=VerusContext)
    persistence = PicklePersistence(filepath="verus_bot.dat", context_types=context_types)
    app = (
        ApplicationBuilder()
        .token(args.telegram_token)
        .updater(None)
        .context_types(context_types)
        .persistence(persistence)
        .arbitrary_callback_data(True)
        .post_init(verus_bot.post_init)
        .post_stop(verus_bot.post_stop)
        .base_url(args.telegram_api_server)
        .local_mode(args.local_mode)
        .build()
    )
    secret_token = uuid4().hex
    await app.bot.set_webhook(args.telegram_webhook_url, secret_token=secret_token)

    # Initialize the Webapp
    quart_app = Quart("Verus")
    webapp = Webapp(
        quart_app=quart_app,
        verus_bot=verus_bot,
        bot=app.bot,
        app=app,
        bot_path=args.telegram_path,
        webapp_path=args.webapp_path,
        secret_token=secret_token,
    )
    webapp.setup_routes()

    # Setup all handlers
    verus_bot.setup_hooks(app, web_prefix=args.webapp_path)

    # Configure the webserver
    webserver = uvicorn.Server(
        config=uvicorn.Config(
            app=quart_app,
            host=args.listen,
            port=args.port,
        )
    )

    # Start the application and the webserver
    try:
        async with app:
            try:
                if app.post_init:
                    await app.post_init(app)
                await app.start()
                await webserver.serve()
            finally:
                try:
                    await app.stop()
                finally:
                    if app.post_stop:
                        await app.post_stop(app)
    finally:
        if app.post_shutdown:
            await app.post_shutdown(app)


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
