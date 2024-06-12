import logging
from http import HTTPStatus

from quart import Quart, Response, abort, make_response, request
from telegram import Update
from telegram.ext import Application, ExtBot

from verus.db import User
from verus.telegram._verusbot import VerusBot
from verus.telegram._webhookupdate import WebhookUpdate

__all__ = ["Webapp"]


class Webapp:
    def __init__(
        self,
        quart_app: Quart,
        verus_bot: VerusBot,
        bot: ExtBot,  # type: ignore[type-arg]
        app: Application,  # type: ignore[type-arg]
        bot_path: str,
        webapp_path: str,
        secret_token: str | None = None,
    ) -> None:
        self.quart_app = quart_app
        self.verus_bot = verus_bot

        self.bot = bot
        self.app = app

        self.bot_path = bot_path
        self.webapp_path = webapp_path
        self.secret_token = secret_token

        self.logger = logging.getLogger(__name__)

    def setup_routes(self) -> None:
        self.quart_app.post(self.bot_path)(self.telegram)
        self.quart_app.route(self.webapp_path, methods=["GET", "POST", "PUT", "DELETE"])(self.custom_updates)
        self.quart_app.get("/health")(self.health)

    async def _validate_post(self) -> None:
        if request.content_type != "application/json":
            abort(HTTPStatus.FORBIDDEN)

        if self.secret_token:
            token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if not token:
                self.logger.debug("Request did not include the secret token")
                abort(HTTPStatus.FORBIDDEN, "Request did not include the secret token")
            elif token != self.secret_token:
                self.logger.debug("Request had the wrong secret token")
                abort(HTTPStatus.FORBIDDEN, "Request had the wrong secret token")

    async def telegram(self) -> Response:
        await self._validate_post()

        try:
            update = Update.de_json(await request.get_json(), self.bot)
        except Exception as exc:
            self.logger.critical(
                "Something went wrong processing the data received from Telegram. "
                "Received data was *not* processed!",
                exc_info=exc,
            )
            abort(HTTPStatus.BAD_REQUEST, "Update could not be processed")

        update = Update.de_json(data=await request.get_json(), bot=self.app.bot)

        if not update:
            abort(Response(status=HTTPStatus.BAD_REQUEST, response="Update could not be processed"))

        self.logger.debug("Received Update with ID %d on Webhook", update.update_id)
        if isinstance(self.bot, ExtBot):
            self.bot.insert_callback_data(update)

        await self.app.update_queue.put(update)
        return Response(status=HTTPStatus.OK)

    async def custom_updates(self) -> Response:
        try:
            api_key = request.headers["X-API-Key"]
            user = User.get_or_none((User.api_key == api_key) & (User.telegram_id.is_null(False)))

            if not user:
                abort(HTTPStatus.UNAUTHORIZED, {"error": "Invalid API key."})

            values = await request.values
        except KeyError:
            abort(HTTPStatus.BAD_REQUEST, {"error": "API key missing."})

        await self.app.update_queue.put(
            WebhookUpdate(
                user=user,
                values=values,
                request=request,
            )
        )
        return Response(status=HTTPStatus.OK)

    async def health(self) -> Response:
        response = await make_response("The bot is still running fine :)", HTTPStatus.OK)
        response.mime_type = "text/plain"  # type: ignore[union-attr]
        return response  # type: ignore[return-value]
