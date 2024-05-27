import hashlib
import json
import logging
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import cast

from huggingface_hub.utils import chunk_iterable
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, PicklePersistence
from tqdm import tqdm

from verus.utils import resize_image_max_side_length, tqdm_logging_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MAX_IMAGE_SIDE_LENGTH = 10_000
MAX_IMAGE_BYTES = 10_000_000  # 10 MB


class Bot:
    file_types = ["jpg", "jpeg", "png"]  # , "gif"]
    file_hash_cache = Path("hashes.json")

    def __init__(self, authorized_user_id: int, image_dir: Path, log_file: Path):
        self.logger = logging.getLogger(__name__)
        self.authorized_user_id = authorized_user_id
        self.image_dir = image_dir
        self.log_file = log_file
        self.log: list[dict[str, str]] = []

        self.images: dict[str, Path] = {}
        self.categories: list[str] = []

        self.processed_hashes: set[str] = set()

        self._load_log()
        self._load_images()

    def _load_image_hashes(self, images: list[Path]) -> dict[Path, str]:
        cache: dict[Path, str] = {}

        if self.file_hash_cache.exists():
            self.logger.info("Loading image hashes from cache...")
            cache = {Path(path): hash for path, hash in json.loads(self.file_hash_cache.read_text()).items()}

        new_images = [image for image in images if image not in cache]

        with tqdm_logging_context():
            with ProcessPoolExecutor() as executor:
                new_image_hashes = dict(
                    zip(images, tqdm(executor.map(self._get_image_hash, new_images), total=len(new_images)))
                )

        cache.update(new_image_hashes)
        self.file_hash_cache.write_text(json.dumps({str(path): hash for path, hash in cache.items()}))

        return cache

    def _save_cache(self) -> None:
        self.file_hash_cache.write_text(json.dumps({str(path): hash for hash, path in self.images.items()}))

    def _load_images(self) -> None:
        self.logger.info("Loading images...")

        image_paths = list(chain(*[self.image_dir.rglob(f"*.{ext}") for ext in self.file_types]))
        images: dict[Path, str] = self._load_image_hashes(image_paths)

        total_duplicates = len(images) - len(set(images.values()))
        if total_duplicates:
            self.logger.warning(f"Found {total_duplicates} duplicate images.")
            self.logger.warning("Delete duplicates? (y/n)")

            if input().lower() == "y":
                for image, image_hash in images.items():
                    if list(images.values()).count(image_hash) > 1:
                        image.unlink()
            else:
                self.logger.warning("Exiting...")
                sys.exit(1)

        for image, image_hash in images.items():
            self.images[image_hash] = image

        self.categories = [cat.name for cat in self.image_dir.iterdir() if cat.is_dir()]

    def _load_log(self) -> None:
        self.logger.info("Loading log...")
        if not self.log_file.exists():
            self.log_file.write_text("[]")

        self.log = json.loads(self.log_file.read_text())
        self.processed_hashes = {entry["hash"] for entry in self.log}

    def _add_log(self, name: str, sha256: str, from_category: str, to_category: str, action: str) -> None:
        self.log.append(
            {
                "filename": name,
                "hash": sha256,
                "from": from_category,
                "to": to_category,
                "action": action,
            }
        )
        self.processed_hashes.add(sha256)
        self.log_file.write_text(json.dumps(self.log))

    def _save_log(self) -> None:
        self.log_file.write_text(json.dumps(self.log))

    def _log_contains_hash(self, image_hash: str) -> bool:
        return image_hash in self.processed_hashes

    def _get_image_hash(self, image_path: Path) -> str:
        hasher = hashlib.sha256(image_path.read_bytes())

        return hasher.hexdigest()

    def _undo(self, sha256: str) -> None:
        self.logger.info("Undo: %s", sha256)
        last_move = next(
            (entry for entry in reversed(self.log) if entry["hash"] == sha256),
            None,
        )

        if last_move:
            self.log.remove(last_move)
            self.processed_hashes.remove(sha256)
            self._save_log()

            if last_move["action"] == "move":
                self._move(sha256, last_move["to"], last_move["from"], add_log=False)

    def _move(self, sha256: str, from_category: str, to_category: str, add_log: bool = True) -> None:
        self.logger.info("Move: %s", sha256)
        image = self.images[sha256]
        new_path = self.image_dir / to_category / image.name
        image.rename(new_path)

        if add_log:
            self._add_log(image.name, sha256, from_category, to_category, "move")

        self.images[sha256] = new_path
        self._save_cache()

    def _prepare_image(self, image: Path) -> BytesIO:
        pil_image = Image.open(image)

        if sum(pil_image.size) > MAX_IMAGE_SIDE_LENGTH:
            pil_image = resize_image_max_side_length(pil_image, MAX_IMAGE_SIDE_LENGTH)
            bytes_ = BytesIO()
            pil_image.save(bytes_, format="JPEG", quality=70)
            bytes_.seek(0)
            return bytes_
        elif image.stat().st_size > MAX_IMAGE_BYTES:
            bytes_ = BytesIO()
            pil_image.save(bytes_, format="JPEG", quality=70)
            bytes_.seek(0)
            return bytes_

        pil_image.close()
        return BytesIO(image.read_bytes())

    async def _next_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        is_update = update.callback_query is not None

        message = update.callback_query.message if is_update else update.message  # type: ignore[union-attr]

        if message is None:
            return

        sha256, image = next(
            ((sha256, image) for sha256, image in self.images.items() if sha256 not in self.processed_hashes),
            (None, None),
        )

        if not image or not sha256:
            if is_update:
                await message.delete()  # type: ignore[union-attr]
                await message.chat.send_message("No more images to process.")
            else:
                await message.reply_text("No more images to process.")  # type: ignore[union-attr]
            return

        raw_image = self._prepare_image(image)

        self.logger.info("Current image: %s %s", sha256, image)

        category = image.parent.name
        reply_markup = self._buttons(sha256)

        context.chat_data["image_hash"] = sha256  # type: ignore[index]

        caption = (
            f"**{category}**\n"
            f"`{image.name}`\n"
            f"`{sha256}`\n"
            f"{len(self.processed_hashes)}/{len(self.images)} {len(self.processed_hashes) / len(self.images) * 100:.2f}%"
        )

        if is_update:
            await message.edit_media(  # type: ignore[union-attr]
                media=InputMediaPhoto(media=raw_image, caption=caption, parse_mode=ParseMode.MARKDOWN),
                reply_markup=reply_markup,
            )
        else:
            await message.reply_photo(  # type: ignore[union-attr]
                photo=raw_image, caption=caption, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN
            )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        if update.effective_user.id != int(self.authorized_user_id):
            await update.message.reply_text("Unauthorized access.")
            return

        await self._next_image(update, context)

    def _buttons(self, image_hash: str) -> InlineKeyboardMarkup:
        categories = chunk_iterable(
            [
                InlineKeyboardButton(
                    cat,
                    callback_data=("move", image_hash, cat),
                )
                for cat in self.categories
            ],
            3,
        )

        buttons = [
            [InlineKeyboardButton("Continue", callback_data=("continue", image_hash, ""))],
            *[list(row) for row in categories],
        ]

        if last_image := self.log[-1]["hash"]:
            buttons.append([InlineKeyboardButton("Undo", callback_data=("undo", last_image, ""))])

        return InlineKeyboardMarkup(buttons)

    def _continue(self, image_hash: str) -> None:
        self.logger.info("Continue: %s", image_hash)
        self._add_log(
            self.images[image_hash].name,
            image_hash,
            self.images[image_hash].parent.name,
            "",
            "continue",
        )

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return

        await query.answer()

        action, hash, category = cast(tuple[str, str, str], query.data)

        if action == "continue":
            self._continue(hash)
        elif action == "move":
            self._move(hash, self.images[hash].parent.name, category)
        elif action == "undo":
            self._undo(hash)
        else:
            self.logger.error(f"Unknown action: {action}")
            return

        context.drop_callback_data(query)
        await self._next_image(update, context)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--user", type=int, required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--log", type=Path, default=Path("log.json"))
    args = parser.parse_args()

    bot = Bot(args.user, args.dir, args.log)

    persistence = PicklePersistence(filepath="verus_bot")
    app = ApplicationBuilder().token(args.token).persistence(persistence).arbitrary_callback_data(True).build()
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CallbackQueryHandler(bot.button))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
