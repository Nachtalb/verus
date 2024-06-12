from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from peewee import CharField, DateTimeField, ForeignKeyField
from playhouse.sqlite_ext import JSONField

from verus.db._base import BaseModel
from verus.db.media import Media, Tag


class History(BaseModel):
    media = ForeignKeyField(Media, backref="history")
    action = CharField()
    timestamp = DateTimeField(default=datetime.now)
    before = JSONField(null=True)
    after = JSONField(null=True)
    data = JSONField(null=True)

    def undo(self) -> None:
        media = self.media

        media.path = self.before["path"]
        media.sha256 = self.before["sha256"]
        media.tags.clear()
        tags = [row["tag"]["name"] for row in self.before["mediatagthrough_set"]]
        media.tags.add([Tag.get_or_create(tag) for tag in tags])
        media._processed = self.before["_processed"]
        media._processed_at = self.before["_processed_at"]
        media.save()
        self.delete_instance()

    @staticmethod
    def latest_action(exclude: Media | None = None) -> History | None:
        history = History.select().order_by(History.timestamp.desc())
        if exclude:
            history = history.where(History.media != exclude)
        return history.first()  # type: ignore[no-any-return]


@contextmanager
def history_action(media: Media, action: str, data: dict[str, Any] = {}) -> Generator[None, None, None]:
    before = json.loads(media.to_json(exceptions=["history"]))
    yield
    after = json.loads(media.to_json(exceptions=["history"]))
    History.create(media=media, action=action, before=before, after=after, data=data)
