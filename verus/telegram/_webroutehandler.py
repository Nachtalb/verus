from telegram.ext import TypeHandler
from telegram.ext._utils.types import HandlerCallback

from verus.telegram._veruscontext import VerusContext
from verus.telegram._webhookupdate import WebhookUpdate

__all__ = ["WebRouteHandler"]


class WebRouteHandler(TypeHandler[WebhookUpdate, VerusContext]):
    def __init__(
        self,
        callback: HandlerCallback[WebhookUpdate, VerusContext, None],
        method: str = "GET",
        path: str = "/",
        prefix: str = "",
    ):
        super().__init__(WebhookUpdate, callback)
        self.method = method
        self.path = prefix + path

    def check_update(self, update: object) -> bool:
        if not isinstance(update, WebhookUpdate):
            return False

        if self.method != update.request.method:
            return False

        if self.path != update.request.path:
            return False

        return True
