from contextlib import contextmanager
from typing import Generator

from peewee import SqliteDatabase

from ._base import BaseModel
from .history import History
from .media import Media, MediaTag, Tag
from .user import User

__all__ = ["DATABASE", "setup_db", "BaseModel", "Tag", "Media", "MediaTag", "History", "User"]


_DATABASE: SqliteDatabase = None


@contextmanager
def atomic() -> Generator[None, None, None]:
    global _DATABASE
    with _DATABASE.atomic():
        yield


def setup_db(database: str = "verus.db") -> SqliteDatabase:
    global _DATABASE

    _DATABASE = SqliteDatabase(database)

    for model in BaseModel.__subclasses__():
        model._meta.database = _DATABASE

    MediaTag._meta.database = _DATABASE

    _DATABASE.connect()
    _DATABASE.create_tables([Tag, Media, MediaTag, History, User])
    return _DATABASE
