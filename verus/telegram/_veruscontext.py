from typing import Type

from telegram.ext import Application, CallbackContext, ExtBot

from verus.telegram._webhookupdate import WebhookUpdate

__all__ = ["VerusContext"]


class VerusContext(CallbackContext[ExtBot, dict, dict, dict]):  # type: ignore[type-arg]
    """Custom context for the Verus bot."""

    @classmethod
    def from_update(cls: "Type[VerusContext]", update: object, application: Application) -> "VerusContext":  # type: ignore[type-arg]
        if isinstance(update, WebhookUpdate):
            return cls(application=application, user_id=update.user.telegram_id)

        return super().from_update(update, application)
