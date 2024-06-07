import hashlib
import logging
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import cast

from huggingface_hub.utils import chunk_iterable
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, PicklePersistence
from tqdm import tqdm

from verus.telegram.db import DATABASE, History, Media, Tag, history_action, setup_db
from verus.utils import resize_image_max_side_length, tqdm_logging_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MAX_IMAGE_SIDE_LENGTH = 10_000
MAX_IMAGE_BYTES = 10_000_000  # 10 MB


class Indexer:
    def __init__(self, image_dir: Path, extensions: list[str] = ["jpg", "jpeg", "png"]):
        self.image_dir = image_dir
        self.extensions = extensions
        self.logger = logging.getLogger(__name__)

    def index(self) -> list[Media]:
        self.logger.info("Indexing images in %s", self.image_dir)

        with DATABASE.atomic():
            tags = {path.name: Tag.get_or_create(path.name) for path in self.image_dir.iterdir() if path.is_dir()}

            known_images = Media.select()
            last_id = 0
            if known_images:
                last_id = Media.select().first().id

            images = list(chain(*[self.image_dir.rglob(f"*.{ext}") for ext in self.extensions]))
            images.sort()
            new_images = set(images) - {Path(image.path) for image in known_images}

            self.logger.info("Found %d new images", len(new_images))
            hashes = self._load_image_hashes(list(new_images))

            self.logger.info("Inserting new images into the database")
            Media.insert_many([{"path": str(image), "sha256": hash} for image, hash in hashes.items()]).execute()
            inserted_images = Media.select().where(Media.id > last_id)

            for path, image in tqdm(zip(new_images, inserted_images), desc="Adding tags"):
                image.tags.add(tags[path.parent.name])
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
    def __init__(self, authorized_user_id: int):
        self.logger = logging.getLogger(__name__)
        self.authorized_user_id = authorized_user_id

        self.toggle_mode: bool = False
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

    def _prepare_image(self, image: Path | str) -> BytesIO:
        self.logger.info("Preparing image: %s", image)
        image = Path(image)
        pil_image = Image.open(image)
        changed = False

        # Convert image to JPEG to reduce size
        if image.stat().st_size > MAX_IMAGE_BYTES:
            changed = True

        # Resize image if any side is longer than MAX_IMAGE_SIDE_LENGTH
        if sum(pil_image.size) > MAX_IMAGE_SIDE_LENGTH:
            pil_image = resize_image_max_side_length(pil_image, MAX_IMAGE_SIDE_LENGTH)
            changed = True

        #  Width and height ratio must be at most 20, crop center otherwise
        if pil_image.width / pil_image.height > 20:
            pil_image = pil_image.crop(
                (
                    pil_image.width // 2 - pil_image.height // 2,
                    0,
                    pil_image.width // 2 + pil_image.height // 2,
                    pil_image.height,
                )
            )
            changed = True
        elif pil_image.height / pil_image.width > 20:
            pil_image = pil_image.crop(
                (
                    0,
                    pil_image.height // 2 - pil_image.width // 2,
                    pil_image.width,
                    pil_image.height // 2 + pil_image.width // 2,
                )
            )
            changed = True

        if changed:
            bytes_ = BytesIO()
            if pil_image.mode == "RGBA":
                white = Image.new("RGB", pil_image.size, "WHITE")
                white.paste(pil_image, (0, 0), pil_image)
                pil_image = white
            pil_image.save(bytes_, format="JPEG", quality=70)
            bytes_.seek(0)
            return bytes_

        pil_image.close()
        return BytesIO(image.read_bytes())

    async def _next_image(self, update: Update) -> None:
        is_update = update.callback_query is not None

        message = update.callback_query.message if is_update else update.message  # type: ignore[union-attr]

        if message is None:
            return

        media = Media.unprocessed().first()
        while not Path(media.path).exists():
            media.delete_instance()
            media = Media.unprocessed().first()

        if media is None:
            if is_update:
                await message.delete()  # type: ignore[union-attr]
                await message.chat.send_message("No more images to process.")
            else:
                await message.reply_text("No more images to process.")  # type: ignore[union-attr]
            return

        raw_image = self._prepare_image(media.path)

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
            if is_update:
                await message.edit_media(  # type: ignore[union-attr]
                    media=InputMediaPhoto(media=raw_image, caption=caption, parse_mode=ParseMode.HTML),
                    reply_markup=reply_markup,
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
                )
            ],
        ]

        if latest_action := History.latest_action():
            buttons.append([InlineKeyboardButton("Undo", callback_data=("undo", latest_action.media.path, ""))])

        return InlineKeyboardMarkup(buttons)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return

        await query.answer()

        try:
            action, path, category = cast(tuple[str, str, str], query.data)
        except TypeError:
            await query.message.delete()
            return

        if action in ["continue", "move", "toggle", "undo"]:
            media = Media.get_or_none(Media.path == path)

            with DATABASE.atomic():
                if action == "continue":
                    self._continue(media)
                elif action == "move":
                    self._move(media, category)
                elif action == "toggle":
                    self._toggle(media, category)
                elif action == "undo":
                    self._undo(media)

        elif action == "toggle_mode":
            self.toggle_mode = not self.toggle_mode
        else:
            self.logger.error(f"Unknown action: {action}")
            return

        context.drop_callback_data(query)
        await self._next_image(update)


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
