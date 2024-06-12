from dataclasses import dataclass
from typing import Any

from quart import Request

from verus.db import User

__all__ = ["WebhookUpdate"]


@dataclass
class WebhookUpdate:
    user: User
    values: dict[str, Any]

    request: Request

    @property
    def user_id(self) -> int:
        return self.user.telegram_id  # type: ignore[no-any-return]
