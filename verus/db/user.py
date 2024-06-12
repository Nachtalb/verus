from __future__ import annotations

from uuid import uuid4

from peewee import CharField, IntegerField

from verus.db._base import BaseModel


def _new_api_key() -> str:
    return uuid4().hex


class User(BaseModel):
    username = CharField(unique=True)
    api_key = CharField(unique=True, default=_new_api_key)
    role = CharField(default="user")
    telegram_id = IntegerField(null=True)

    def __str__(self) -> str:
        return f"{self.username} ({self.partial_api_key()})"

    def partial_api_key(self) -> str:
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def recreate_api_key(self) -> str:
        self.api_key = _new_api_key()
        self.save()
        return self.api_key
