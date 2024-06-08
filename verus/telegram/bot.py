import asyncio
import hashlib
import logging
import re
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import cast

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, PicklePersistence
from tqdm import tqdm

from verus.image import create_tg_thumbnail, create_tg_thumbnail_from_video
from verus.telegram.db import DATABASE, History, Media, Tag, history_action, setup_db
from verus.utils import bool_emoji, chunk_iterable, tqdm_logging_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MAX_IMAGE_SIDE_LENGTH = 10_000
MAX_IMAGE_BYTES = 10_000_000  # 10 MB


class Indexer:
    def __init__(self, image_dir: Path, extensions: list[str] = ["jpg", "jpeg", "png", "mp4"]):
        self.image_dir = image_dir
        self.extensions = extensions
        self.logger = logging.getLogger(__name__)

    def index(self) -> list[Media]:
        self.logger.info("Indexing images in %s", self.image_dir)

        with DATABASE.atomic():
            tags = {
                path.name: Tag.get_or_create(path.name)
                for path in self.image_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            }

            known_images = Media.select()
            last_id = 0
            if known_images:
                last_id = Media.select().first().id

            images = list(chain(*[self.image_dir.rglob(f"*.{ext}") for ext in self.extensions]))
            images = [image for image in images if "thumb" not in image.name]
            images.sort()
            new_images = set(images) - {Path(image.path) for image in known_images}

            self.logger.info("Found %d new images", len(new_images))
            hashes = self._load_image_hashes(list(new_images))

            self.logger.info("Inserting new images into the database")
            Media.insert_many([{"path": str(image), "sha256": hash} for image, hash in hashes.items()]).execute()
            inserted_images = Media.select().where(Media.id > last_id)

            for path, image in tqdm(zip(new_images, inserted_images), desc="Adding tags"):
                tag = tags[path.parent.name]
                if tag not in image.tags:
                    image.tags.add(tag)
                    image.save()

        return Media.select()  # type: ignore[no-any-return]

    def _load_image_hashes(self, images: list[Path]) -> dict[Path, str]:
        with tqdm_logging_context():
            with ProcessPoolExecutor() as executor:
                return dict(zip(images, tqdm(executor.map(self._get_image_hash, images), total=len(images))))

    def _get_image_hash(self, image_path: Path) -> str:
        hasher = hashlib.sha256(image_path.read_bytes())

        return hasher.hexdigest()


class Bot:
    _intermediate_group_message: tuple[Message, ...] = ()

    def __init__(self, authorized_user_id: int):
        self.logger = logging.getLogger(__name__)
        self.authorized_user_id = authorized_user_id

        self.toggle_mode: bool = False
        self.group_ask: bool = True
        self.tags = Tag.select()

    def _undo(self, media: Media) -> None:
        self.logger.info("Undo: %s", media.path)
        if last_action := media.last_action:
            last_action.undo()

    def _move(self, media: Media, to_tag: str | Tag, fill_history: bool = True) -> None:
        self.logger.info("Move: %s to %s", media.path, to_tag)

        with history_action(media, action="move") if fill_history else nullcontext():
            media.tags.clear()
            media.tags.add(Tag.get_or_create(to_tag))
            media.processed = True
            media.save()

    def _group_set_tags(self, group: list[Media], tags: list[Tag], fill_history: bool = True) -> None:
        id = self.extract_id(group[0].path)
        self.logger.info("Group move: %s to %s", id, ", ".join([tag.name for tag in tags]))

        if not tags:
            raise ValueError("No tags provided")
            return

        for item in group:
            with history_action(item, action="group_move") if fill_history else nullcontext():
                item.tags.clear()
                for tag in tags:
                    item.tags.add(tag)
                item.processed = True
                item.save()

    def _toggle(self, media: Media, to_tag: str | Tag, fill_history: bool = True) -> None:
        self.logger.info("Toggle: %s, %s", media.path, to_tag)

        with history_action(media, action="toggle") if fill_history else nullcontext():
            tag = Tag.get_or_create(to_tag)
            if tag in media.tags:
                media.tags.remove(tag)
            else:
                media.tags.add(Tag.get_or_create(to_tag))

    def _continue(self, media: Media) -> None:
        self.logger.info("Continue: %s", media.path)
        with history_action(media, action="continue"):
            media.processed = True
            media.save()

    def extract_id(self, filename: str) -> str | None:
        match = re.search(r"_(\d+)_p\d+\.", filename)
        return match.group(1) if match else None

    def get_group(self, media: Media, non_processed_only: bool = True) -> list[Media]:
        if not media:
            return []

        id = self.extract_id(media.path)

        if not id:
            return []

        group = Media.select().where(Media.path.contains(id))

        if non_processed_only:
            group = group.where(Media._processed == False)  # noqa: E712

        group = sorted(group, key=lambda item: Path(item.path).name)
        return group  # type: ignore[no-any-return]

    async def send_media_group(self, group: list[Media], message: Message) -> None:
        id = self.extract_id(group[0].path)
        pixiv_url = f"https://www.pixiv.net/artworks/{id}"

        media_group = []
        for index, item in enumerate(group):
            if index == 0:
                input_ = InputMediaPhoto(
                    media=self._get_or_create_thumbnail(item.path).read_bytes(), caption=f"ID: {id}\nPixiv: {pixiv_url}"
                )
            else:
                input_ = InputMediaPhoto(media=self._get_or_create_thumbnail(item.path).read_bytes())
            media_group.append(input_)

        messages: list[Message] = []
        for index, chunk in enumerate(chunk_iterable(media_group, 10)):
            if index > 1:
                await asyncio.sleep(1)
            messages.extend(await message.reply_media_group(list(chunk)))

        self._intermediate_group_message = tuple(messages)

    def _get_or_create_thumbnail(self, image: Path | str) -> Path:
        image = Path(image)
        thumb_path = image.with_name(f"{image.stem}.thumb.jpg")
        if not thumb_path.is_file():
            if image.suffix in [".mp4", ".webm"]:
                thumb = create_tg_thumbnail_from_video(image, 1024)
            else:
                thumb = create_tg_thumbnail(image, 1024)

            thumb.save(thumb_path, format="JPEG", quality=70)
            thumb.close()

        return thumb_path

    async def _next_image(self, update: Update, force_new_message: bool = False) -> None:
        is_update = update.callback_query is not None

        message = update.callback_query.message if is_update else update.message  # type: ignore[union-attr]

        if message is None:
            return

        media = Media.unprocessed().first()
        while media and not Path(media.path).exists():
            media.delete_instance()
            media = Media.unprocessed().first()

        if media is None:
            if is_update:
                await message.delete()  # type: ignore[union-attr]
                await message.chat.send_message("No more images to process.")
            else:
                await message.reply_text("No more images to process.")  # type: ignore[union-attr]
            return

        is_video = media.path.endswith((".mp4", ".webm"))
        if is_video:
            raw_image = BytesIO(Path(media.path).read_bytes())
        else:
            raw_image = BytesIO(self._get_or_create_thumbnail(Path(media.path)).read_bytes())

        self.logger.info("Current image: %s %s", media.path, media.sha256)

        reply_markup = self._buttons(media)

        categories = ", ".join([tag.name for tag in media.tags])
        processed_images = Media.select().where(Media._processed == True).count()  # noqa: E712
        total_images = Media.select().count()
        caption = (
            f"<b>Category: {categories}</b>\n"
            f"Mode: {'toggle' if self.toggle_mode else 'move'}\n"
            f"Name: <code>{Path(media.path).name}</code>\n"
            f"SHA256: <code>{media.sha256}</code>\n"
            f"Progress: {processed_images}/{total_images} {processed_images/total_images*100:.2f}%"
        )

        try:
            if is_update and not force_new_message:
                media_type = InputMediaVideo if is_video else InputMediaPhoto

                await message.edit_media(  # type: ignore[union-attr]
                    media=media_type(media=raw_image, caption=caption, parse_mode=ParseMode.HTML),
                    reply_markup=reply_markup,
                )
            else:
                if is_video:
                    await message.reply_video(  # type: ignore[union-attr]
                        video=raw_image, caption=caption, reply_markup=reply_markup, parse_mode=ParseMode.HTML
                    )
                else:
                    await message.reply_photo(  # type: ignore[union-attr]
                        photo=raw_image, caption=caption, reply_markup=reply_markup, parse_mode=ParseMode.HTML
                    )
        except BadRequest as e:
            if "Image_process_failed" in str(e):
                self.logger.error("Image processing failed for %s", media.path)
                media.delete_instance()
                await self._next_image(update)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        if update.effective_user.id != int(self.authorized_user_id):
            await update.message.reply_text("Unauthorized access.")
            return

        await self._next_image(update)

    def _buttons(self, media: Media) -> InlineKeyboardMarkup:
        categories = chunk_iterable(
            [
                InlineKeyboardButton(
                    ("✅" if cat in media.tags else "") + cat.name,
                    callback_data=("toggle" if self.toggle_mode else "move", media.path, cat.name),
                )
                for cat in self.tags
            ],
            3,
        )

        buttons = [
            [InlineKeyboardButton("Continue", callback_data=("continue", media.path, ""))],
            *[list(row) for row in categories],
            [
                InlineKeyboardButton(
                    "Mode: " + ("Toggle" if self.toggle_mode else "Move"), callback_data=("toggle_mode", "", "")
                ),
                InlineKeyboardButton(
                    "Group: " + bool_emoji(self.group_ask), callback_data=("toggle_group", media.path, "")
                ),
                InlineKeyboardButton("More", callback_data=("more", media.path, "")),
            ],
        ]

        if latest_action := History.latest_action():
            buttons.append([InlineKeyboardButton("Undo", callback_data=("undo", latest_action.media.path, ""))])

        return InlineKeyboardMarkup(buttons)

    async def _clear_intermediate_group_message(self) -> None:
        for message in self._intermediate_group_message:
            await message.delete()
        self._intermediate_group_message = ()

    async def ask_for_group_action(self, media: Media, action: str, move_category: str, query: CallbackQuery) -> None:
        group = self.get_group(media)

        if not group or not query.message:
            # something went wrong
            return

        await self.send_media_group(group, query.message)  # type: ignore[arg-type]

        buttons = [
            InlineKeyboardButton("Move", callback_data=(f"group_{action}", media.path, move_category)),
            InlineKeyboardButton(
                "Move & Delete Last", callback_data=(f"group_{action}_except_last", media.path, move_category)
            ),
            InlineKeyboardButton("Cancel", callback_data=(f"simple_{action}", media.path, move_category)),
        ]

        tags = move_category if action == "move" else ", ".join([tag.name for tag in media.tags])

        await query.message.reply_text(  # type: ignore[attr-defined]
            f"Do you want to move the group to these tags?: {tags}",
            reply_markup=InlineKeyboardMarkup([buttons]),
            parse_mode=ParseMode.MARKDOWN,
        )

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return

        try:
            action, path, category = cast(tuple[str, str, str], query.data)
        except TypeError:
            await query.answer()
            await query.message.delete()  # type: ignore[union-attr]
            return

        force_new = False

        initial_action = action
        if action in ["continue", "move"] and not self.group_ask:
            action = "simple_" + action

        if action in ["continue", "move"]:
            media = Media.get_or_none(Media.path == path)

            group = self.get_group(media)

            if not group or len(group) == 1 or not self.group_ask:
                action = "simple_move" if initial_action == "move" else "simple_continue"
            else:
                await self._clear_intermediate_group_message()
                await self.ask_for_group_action(media, action, category, query)
                await query.answer()
                if query.message:
                    await query.message.delete()  # type: ignore[attr-defined]
                context.drop_callback_data(query)
                return

        if action in ["undo", "simple_continue", "simple_move"]:
            media = Media.get_or_none(Media.path == path)

            if initial_action.startswith("simple_") and query.message:
                await query.message.delete()  # type: ignore[attr-defined]
                force_new = True

                self.group_ask = False
                await query.answer("Turned off group ask.")

            with DATABASE.atomic():
                if action == "simple_continue":
                    self._continue(media)
                elif action == "simple_move":
                    self._move(media, category)
                elif action == "undo":
                    self._undo(media)
            await self._clear_intermediate_group_message()
        elif action.startswith("group_"):
            media = Media.get_or_none(Media.path == path)
            tags = media.tags if "continue" in action else [Tag.get_or_create(category)]

            group = self.get_group(media)
            if query.message:
                await query.message.delete()  # type: ignore[attr-defined]
                force_new = True

            if "except_last" in action and len(group) > 1:
                self._move(group[-1], "delete")
                group = group[:-1]

            self._group_set_tags(group, tags)
            await self._clear_intermediate_group_message()
        elif action == "toggle":
            media = Media.get_or_none(Media.path == path)

            with DATABASE.atomic():
                if action == "toggle":
                    self._toggle(media, category)
        elif action == "more":
            media = Media.get_or_none(Media.path == path)
            group = self.get_group(media, non_processed_only=False)
            if not group or not query.message:
                await query.answer("No group found.")
            else:
                await self.send_media_group(group, query.message)  # type: ignore[arg-type]
        elif action == "toggle_group":
            self.group_ask = not self.group_ask
            await query.answer(f"Group Ask: {bool_emoji(self.group_ask)}")
        elif action == "toggle_mode":
            self.toggle_mode = not self.toggle_mode
            await query.answer(f"Current Mode: {'Toggle' if self.toggle_mode else 'Move'}")
        else:
            self.logger.error(f"Unknown action: {action}")
            await query.answer()
            return
        await query.answer()

        context.drop_callback_data(query)
        await self._next_image(update, force_new)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--user", type=int, required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--log", type=Path, default=Path("log.json"))
    parser.add_argument("--db", type=Path, default=Path("verus.db"))

    sub_parsers = parser.add_subparsers()
    webhook_parser = sub_parsers.add_parser("webhook")
    webhook_parser.add_argument("--webhook-url", required=True)
    webhook_parser.add_argument("--webhook-path", default="")
    webhook_parser.add_argument("--listen", default="0.0.0.0")
    webhook_parser.add_argument("--port", type=int, default=8433)

    webhook_parser.set_defaults(webhook=True)

    args = parser.parse_args()

    setup_db(args.db)

    indexer = Indexer(args.dir)
    indexer.index()

    bot = Bot(args.user)

    persistence = PicklePersistence(filepath="verus_bot.dat")
    app = ApplicationBuilder().token(args.token).persistence(persistence).arbitrary_callback_data(True).build()
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CallbackQueryHandler(bot.button))

    if hasattr(args, "webhook"):
        app.run_webhook(
            listen=args.listen,
            port=args.port,
            webhook_url=args.webhook_url,
            url_path=args.webhook_path,
            secret_token="ASecretTokenIHaveChangedByNow",
            #  allowed_updates=Update.ALL_TYPES,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
