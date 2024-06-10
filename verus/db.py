import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator
from uuid import uuid4

from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    ManyToManyField,
    Model,
    Query,
    SqliteDatabase,
    fn,
)
from playhouse.shortcuts import dict_to_model, model_to_dict
from playhouse.sqlite_ext import JSONField
from telegram import Bot, PhotoSize, Video

DATABASE: SqliteDatabase = SqliteDatabase("verus.db")


class BaseModel(Model):  # type: ignore[misc]
    class Meta:
        database = DATABASE

    def to_dict(self, exceptions: list[str] = []) -> dict[str, Any]:
        data = model_to_dict(self, backrefs=True, recurse=True)
        for key in exceptions:
            data.pop(key, None)
        return data  # type: ignore[no-any-return]

    def to_json(self, exceptions: list[str] = []) -> str:
        data = self.to_dict(exceptions=exceptions)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any], force_insert: bool = False) -> "BaseModel":
        instance = dict_to_model(cls, data)
        instance.save(force_insert=force_insert)
        return instance  # type: ignore[no-any-return]

    @classmethod
    def from_json(cls, data: str, force_insert: bool = False) -> "BaseModel":
        json_data = json.loads(data)
        for key, value in json_data.items():
            if value is not None and cls._meta.fields.get(key) == DateTimeField:
                json_data[key] = datetime.fromisoformat(value)
        return cls.from_dict(json_data, force_insert=force_insert)


class Tag(BaseModel):
    name = CharField(unique=True)

    def __str__(self) -> str:
        return self.name  # type: ignore[no-any-return]

    @staticmethod
    def get_or_create(name: str) -> "Tag":
        if tag := Tag.get_or_none(Tag.name == name):
            return tag  # type: ignore[no-any-return]
        return Tag.create(name=name)  # type: ignore[no-any-return]


class Media(BaseModel):
    name = CharField()
    path = CharField(unique=True)
    sha256 = CharField()
    tags = ManyToManyField(Tag, backref="media")

    tg_file_info = JSONField(null=True)
    group_id = IntegerField(null=True)

    _processed = BooleanField(default=False)
    _processed_at = DateTimeField(null=True)

    def __str__(self) -> str:
        return self.path  # type: ignore[no-any-return]

    def get_tg_file_obj(self, bot: Bot) -> Video | PhotoSize | None:
        if not self.tg_file_info:
            return None

        if self.tg_file_info.get("mime_type", "").startswith("video"):
            return Video.de_json(self.tg_file_info, bot)
        else:
            return PhotoSize.de_json(self.tg_file_info, bot)

    def get_group(self, non_processed_only: bool = False) -> "list[Media]":
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
        return (
            Media.select()
            .where(Media._processed == False)  # noqa: E712
            .order_by(fn.substr(Media.path, fn.length(Media.path) - fn.length(fn.replace(Media.path, "/", "")) + 1))
        )

    @staticmethod
    def get_or_create(
        path: str,
        sha256: str,
        tags: list[Tag | str] = [],
        processed: bool = False,
        processed_at: datetime | None = None,
    ) -> "Media":
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
    def last_action(self) -> "History | None":
        return self.history.order_by(History.timestamp.desc()).first()  # type: ignore[no-any-return]


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
    def latest_action(exclude: Media | None = None) -> "History | None":
        history = History.select().order_by(History.timestamp.desc())
        if exclude:
            history = history.where(History.media != exclude)
        return history.first()  # type: ignore[no-any-return]


MediaTag = Media.tags.get_through_model()


@contextmanager
def history_action(media: Media, action: str, data: dict[str, Any] = {}) -> Generator[None, None, None]:
    before = json.loads(media.to_json(exceptions=["history"]))
    yield
    after = json.loads(media.to_json(exceptions=["history"]))
    History.create(media=media, action=action, before=before, after=after, data=data)


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


def setup_db() -> SqliteDatabase:
    global DATABASE
    DATABASE.connect()
    DATABASE.create_tables([Tag, Media, MediaTag, History, User])
    return DATABASE
