from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from peewee import BooleanField, CharField, DateTimeField, IntegerField, ManyToManyField, Query
from playhouse.sqlite_ext import JSONField
from telegram import Bot, PhotoSize, Video

from verus.db._base import BaseModel

if TYPE_CHECKING:
    from verus.db.history import History


class Tag(BaseModel):
    name = CharField(unique=True)

    def __str__(self) -> str:
        return self.name  # type: ignore[no-any-return]

    @staticmethod
    def get_or_create(name: str) -> Tag:
        if tag := Tag.get_or_none(Tag.name == name):
            return tag  # type: ignore[no-any-return]
        return Tag.create(name=name)  # type: ignore[no-any-return]


class Media(BaseModel):
    name = CharField()
    path = CharField(unique=True)
    sha256 = CharField(unique=True)
    tags = ManyToManyField(Tag, backref="media")
    stale: bool = BooleanField(default=False)

    tg_file_info = JSONField(null=True)
    group_id = IntegerField(null=True)

    _processed = BooleanField(default=False)
    _processed_at = DateTimeField(null=True)

    def __str__(self) -> str:
        return self.path  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"<Media id={self.id}, name={self.name}>"

    def get_tg_file_obj(self, bot: Bot) -> Video | PhotoSize | None:
        if not self.tg_file_info:
            return None

        if self.tg_file_info.get("mime_type", "").startswith("video"):
            return Video.de_json(self.tg_file_info, bot)
        else:
            return PhotoSize.de_json(self.tg_file_info, bot)

    def get_group(self, non_processed_only: bool = False) -> list[Media]:
        if not self.group_id:
            if non_processed_only and not self._processed:
                return [self]
            return []
        medias = Media.select().where(Media.group_id == self.group_id).order_by(Media.name)
        if non_processed_only:
            return list(medias.where(Media._processed == False))  # noqa: E712
        return list(medias)

    @staticmethod
    def unprocessed() -> Query:
        """Return all unprocessed media sorted by name."""
        return Media.select().where(Media._processed == False)  # noqa: E712

    @staticmethod
    def get_or_create(
        path: str,
        sha256: str,
        tags: list[Tag | str] = [],
        processed: bool = False,
        processed_at: datetime | None = None,
    ) -> Media:
        if media := Media.get_or_none(Media.path == path):
            return media  # type: ignore[no-any-return]

        media = Media.create(path=path, sha256=sha256)

        if tags:
            media.tags.clear()
            media.tags.add([tag if isinstance(tag, Tag) else Tag.get_or_create(tag) for tag in tags])

        if processed:
            media._processed = True
            media._processed_at = processed_at
        return media  # type: ignore[no-any-return]

    @property
    def processed(self) -> bool:
        return self._processed  # type: ignore[no-any-return]

    @processed.setter
    def processed(self, value: bool) -> None:
        if value:
            self._processed = True
            self._processed_at = datetime.now()
        else:
            self._processed = False
            self._processed_at = None

    @property
    def last_action(self) -> History | None:
        from .history import History

        return self.history.order_by(History.timestamp.desc()).first()  # type: ignore[no-any-return]


MediaTag = Media.tags.get_through_model()
