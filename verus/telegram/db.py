import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    ForeignKeyField,
    ManyToManyField,
    Model,
    Query,
    SqliteDatabase,
)
from playhouse.shortcuts import dict_to_model, model_to_dict
from playhouse.sqlite_ext import JSONField

DATABASE: SqliteDatabase = SqliteDatabase("verus.db")


class BaseModel(Model):  # type: ignore[misc]
    class Meta:
        database = DATABASE

    def to_dict(self) -> dict[str, Any]:
        return model_to_dict(self, backrefs=True, recurse=True)  # type: ignore[no-any-return]

    def to_json(self) -> str:
        data = self.to_dict()
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
    path = CharField(unique=True)
    sha256 = CharField()
    tags = ManyToManyField(Tag, backref="media")

    _processed = BooleanField(default=False)
    _processed_at = DateTimeField(null=True)

    def __str__(self) -> str:
        return self.path  # type: ignore[no-any-return]

    @staticmethod
    def unprocessed() -> Query:
        return Media.select().where(Media._processed == False)  # noqa: E712

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
        media.from_dict(self.before, force_insert=True)
        self.delete_instance()

    @staticmethod
    def latest_action() -> "History | None":
        return History.select().order_by(History.timestamp.desc()).first()  # type: ignore[no-any-return]


MediaTag = Media.tags.get_through_model()


@contextmanager
def history_action(media: Media, action: str, data: dict[str, Any] = {}) -> Generator[None, None, None]:
    before = json.loads(media.to_json())
    yield
    after = json.loads(media.to_json())
    History.create(media=media, action=action, before=before, after=after, data=data)


def setup_db(db_path: str) -> SqliteDatabase:
    global DATABASE
    #  DATABASE = SqliteDatabase(db_path)
    DATABASE.connect()
    DATABASE.create_tables([Tag, Media, MediaTag, History])
    return DATABASE
