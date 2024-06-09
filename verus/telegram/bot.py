import asyncio
import logging
import re
import shutil
from argparse import ArgumentParser
from contextlib import nullcontext
from functools import reduce
from io import BytesIO
from pathlib import Path
from typing import cast

from tabulate import tabulate
from telegram import (
    Animation,
    CallbackQuery,
    Document,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
    PhotoSize,
    Update,
    Video,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PicklePersistence,
    filters,
)

from verus.const import TG_BASE_URL, TG_MAX_DOWNLOAD_SIZE
from verus.db import DATABASE, History, Media, MediaTag, Tag, User, history_action, setup_db
from verus.files import get_supported_files
from verus.image import create_and_save_tg_thumbnail, get_dimensions
from verus.indexer import Indexer
from verus.utils import bool_emoji, chunk_iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Bot:
    _intermediate_group_message: tuple[Message, ...] = ()

    def __init__(
        self,
        authorized_users: list[User],
        image_dir: Path,
        upload_folder_name: str,
        import_folder: Path | None = None,
        local_mode: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.authorized_users = authorized_users
        self.image_dir = image_dir
        self.upload_folder = image_dir / upload_folder_name
        self.import_folder = import_folder
        self.local_mode = local_mode

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
            photo = self.photo_or_raw(item)
            if index == 0:
                input_ = InputMediaPhoto(media=photo, caption=f"ID: {id}\nPixiv: {pixiv_url}")
            else:
                input_ = InputMediaPhoto(media=photo)
            media_group.append(input_)

        messages: list[Message] = []
        for index, chunk in enumerate(chunk_iterable(media_group, 10)):
            if index > 1:
                await asyncio.sleep(1)
            messages.extend(await message.reply_media_group(list(chunk)))

        self._intermediate_group_message = tuple(messages)

    async def _next_image(self, update: Update, force_new_message: bool = False) -> None:
        is_update = update.callback_query is not None

        message = update.callback_query.message if is_update else update.message  # type: ignore[union-attr]

        if message is None:
            return

        media: Media | None = Media.unprocessed().first()
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

        self.logger.info("Current image: %s %s", media.path, media.sha256)

        reply_markup = self._buttons(media)

        try:
            send_func = self.send_video if media.path.endswith((".mp4", ".gif")) else self.send_photo
            new_msg = await send_func(
                media,
                message,  # type: ignore[arg-type]
                is_update and not force_new_message,
                buttons=reply_markup,
                caption=self.get_sort_caption(media),
            )

            if not media.tg_file_info and new_msg and (tg_file := new_msg.video or new_msg.photo[-1]):
                media.tg_file_info = tg_file.to_dict()
                media.save()

        except BadRequest as e:
            if "Image_process_failed" in str(e):
                self.logger.error("Image processing failed for %s", media.path)
                media.delete_instance()
                await self._next_image(update)

    def get_sort_caption(self, media: Media) -> str:
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
        return caption

    def photo_or_raw(self, media: Media, as_thumbnail: bool = True) -> PhotoSize | BytesIO:
        photo = media.get_tg_file_obj(media)
        if isinstance(photo, PhotoSize):
            return photo
        if as_thumbnail:
            return BytesIO(create_and_save_tg_thumbnail(media.path).read_bytes())
        return BytesIO(Path(media.path).read_bytes())

    def video_or_raw(self, media: Media, as_thumbnail: bool = True) -> Video | BytesIO:
        video = media.get_tg_file_obj(media)
        if isinstance(video, Video):
            return video
        if as_thumbnail:
            return BytesIO(create_and_save_tg_thumbnail(media.path).read_bytes())
        return BytesIO(Path(media.path).read_bytes())

    async def send_video(
        self,
        media: Media,
        message: Message,
        update_message: bool,
        buttons: InlineKeyboardMarkup | None = None,
        caption: str | None = None,
    ) -> Message:
        video = self.video_or_raw(media, as_thumbnail=False)
        thumbnail = create_and_save_tg_thumbnail(media.path) if isinstance(video, BytesIO) else None
        width, height = get_dimensions(thumbnail) if thumbnail else (None, None)

        if update_message:
            input_media = InputMediaVideo(
                media=video,
                caption=caption,
                parse_mode=ParseMode.HTML,
                thumbnail=thumbnail,
                width=width,
                height=height,
            )
            return await message.edit_media(media=input_media, reply_markup=buttons)  # type: ignore[return-value]
        else:
            return await message.reply_video(
                video=video,
                caption=caption,
                reply_markup=buttons,
                parse_mode=ParseMode.HTML,
                thumbnail=thumbnail,
                width=width,
                height=height,
            )

    async def send_photo(
        self,
        media: Media,
        message: Message,
        update_message: bool,
        buttons: InlineKeyboardMarkup | None = None,
        caption: str | None = None,
    ) -> Message:
        if update_message:
            input_media = InputMediaPhoto(
                media=self.photo_or_raw(media),
                caption=caption,
                parse_mode=ParseMode.HTML,
            )
            return await message.edit_media(media=input_media, reply_markup=buttons)  # type: ignore[return-value]
        else:
            return await message.reply_photo(
                photo=self.photo_or_raw(media),
                caption=caption,
                reply_markup=buttons,
                parse_mode=ParseMode.HTML,
            )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        await self._next_image(update)

    async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        processed_images = Media.select().where(Media._processed == True).count()  # noqa: E712
        unprocessed_images = Media.select().where(Media._processed == False).count()  # noqa: E712
        total_images = Media.select().count()
        media: Media | None = Media.unprocessed().first()

        per_category = {
            tag.name: Media.select()
            .join(MediaTag, on=(Media.id == MediaTag.media_id))
            .where(MediaTag.tag == tag)
            .count()
            for tag in self.tags
        }

        per_category_str = tabulate(per_category.items(), headers=["Category", "Count"], tablefmt="grid")

        await update.message.reply_text(
            f"Current image: `{media and media.path or 'None'}`\n"
            f"Categories: {','.join([f'`{tag.name}`' for tag in self.tags])}\n"
            f"Processed: `{processed_images}`\n"
            f"Unprocessed: `{unprocessed_images}`\n"
            f"Total: `{total_images}`\n"
            f"Per category: \n```{per_category_str}```",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def refresh(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        message_text = "Refreshing...this may take a while."
        message = await update.message.reply_text(message_text)

        if self.import_folder:
            files = get_supported_files(self.import_folder)
            message_text += f"\nImporting {len(files)} files..."
            await message.edit_text(message_text)
            for file in files:
                new_path = self.upload_folder / file.name
                file.rename(new_path)
            message_text += "done."
            await message.edit_text(message_text)

        message_text += "\nIndexing..."
        await message.edit_text(message_text)

        total_images = Media.select().count()
        total_tags = Tag.select().count()

        indexer = Indexer(self.image_dir)
        new, removed = indexer.index()

        new_total_images = Media.select().count()
        new_total_tags = Tag.select().count()

        if new or removed or total_tags != new_total_tags:
            message_text += f"done.\n  Images: `{total_images}` => `{new_total_images}`\n  Tags: `{total_tags}` => `{new_total_tags}`"
            if removed:
                message_text += f"\n  Removed Stale: {removed}"
        else:
            message_text += "done.\nNo new images or tags found."

        message_text += "\nRefresh complete. Use /start to begin."
        await message.edit_text(message_text, parse_mode=ParseMode.MARKDOWN)

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
        if not self._intermediate_group_message:
            msg = self._intermediate_group_message[0]
            bot = msg.get_bot()
            await bot.delete_messages(msg.chat.id, [m.message_id for m in self._intermediate_group_message])
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

    async def post_init(self, application: Application) -> None:  # type: ignore[type-arg]
        self.logger.info("Media Sorting Bot started.")
        await application.bot.set_my_commands(
            [
                ("start", "Start the sorting"),
                ("info", "Show information about the bot"),
                ("refresh", "Refresh the database"),
                ("undo", "Undo the last action"),
            ]
        )

        for user in self.authorized_users:
            if not user.telegram_id:
                continue
            self.logger.info("Sending message to %s", user.telegram_id)
            await application.bot.send_message(user.telegram_id, "Media Sorting Bot started.")

    async def post_stop(self, application: Application) -> None:  # type: ignore[type-arg]
        self.logger.info("Media Sorting Bot stopped.")
        for user in self.authorized_users:
            if not user.telegram_id:
                continue
            self.logger.info("Sending message to %s", user.telegram_id)
            await application.bot.send_message(user.telegram_id, "Media Sorting Bot stopped.")

    async def receive_new_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        obj: PhotoSize | Video | Document | Animation | None
        if update.message.photo:
            obj = update.message.photo[-1]
        else:
            obj = update.message.video or update.message.document or update.message.animation

        if not obj:
            return

        if not obj.file_size:
            await update.message.reply_text("File size is not available.")
            return
        elif not self.local_mode and obj.file_size > TG_MAX_DOWNLOAD_SIZE:
            await update.message.reply_text("File is too large. Max size is 20 MB.")
            return

        media = await obj.get_file()
        if not media.file_path:
            await update.message.reply_text("Can't determine file extension.")
            return

        ext = Path(media.file_path).suffix
        if ext not in [".jpg", ".jpeg", ".png", ".mp4"]:
            await update.message.reply_text("Unsupported file type.")
            return

        media_path = self.upload_folder / f"{media.file_id}{ext}"

        if media_path.exists():
            existing_media: Media | None = Media.get_or_none(Media.path == str(media_path))
            if existing_media and existing_media.processed:
                await update.message.reply_text(
                    f"Media already exists `{media_path.name}` and is processed.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            await update.message.reply_text(
                f"Media already exists `{media_path.name}`. Use /refresh then /start to process it.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        progress_msg = await update.message.reply_text(
            f"Receiving media `{media_path.name}`...", parse_mode=ParseMode.MARKDOWN
        )

        if self.local_mode and Path(media.file_path).exists():
            shutil.copyfile(media.file_path, media_path)
        else:
            out = BytesIO()
            await media.download_to_memory(out)
            media_path.write_bytes(out.getvalue())

        await progress_msg.delete()
        await update.message.reply_text(
            f"Media received `{media_path.name}`. Use /refresh then /start to process it.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def undo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        last_action = History.latest_action()
        if last_action:
            self._undo(last_action.media)
            await update.message.reply_text(
                f"Last action undone. `{last_action.action}` on `{Path(last_action.media.path).name}`",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await update.message.reply_text("No action to undo.")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--upload-folder-name", type=str, default="misc")
    parser.add_argument("--import-folder", type=Path, default=Path())
    parser.add_argument("--token", required=True)
    parser.add_argument("--log", type=Path, default=Path("log.json"))
    # Typical local path: "http://localhost:8081/bot"
    parser.add_argument("--base-url", default=TG_BASE_URL, help="Base URL for the bot API")
    parser.add_argument("--local-mode", action="store_true", help="Run the bot in local mode", default=False)

    sub_parsers = parser.add_subparsers()
    webhook_parser = sub_parsers.add_parser("webhook")
    webhook_parser.add_argument("--webhook-url", required=True)
    webhook_parser.add_argument("--webhook-path", default="")
    webhook_parser.add_argument("--listen", default="0.0.0.0")
    webhook_parser.add_argument("--port", type=int, default=8433)

    webhook_parser.set_defaults(webhook=True)

    args = parser.parse_args()

    setup_db()

    indexer = Indexer(args.dir)
    indexer.index()

    import_folder = args.import_folder if Path() != args.import_folder else None

    admins = list(User.select().where((User.role == "admin") & (User.telegram_id.is_null(False))))

    bot = Bot(
        authorized_users=admins,
        image_dir=args.dir,
        upload_folder_name=args.upload_folder_name,
        import_folder=import_folder,
        local_mode=args.local_mode,
    )

    persistence = PicklePersistence(filepath="verus_bot.dat")
    app = (
        ApplicationBuilder()
        .token(args.token)
        .persistence(persistence)
        .arbitrary_callback_data(True)
        .post_init(bot.post_init)
        .post_stop(bot.post_stop)
        .base_url(args.base_url)
        .local_mode(args.local_mode)
        .build()
    )

    users = [filters.User(user.telegram_id) for user in admins]
    user_filter = users[0]
    if len(users) > 1:
        user_filter = reduce(lambda x, y: x | y, users)  # type: ignore[return-value, arg-type]

    default_filter = filters.ChatType.PRIVATE & user_filter

    app.add_handler(CommandHandler("start", bot.start, filters=default_filter))
    app.add_handler(CommandHandler("info", bot.info, filters=default_filter))
    app.add_handler(CommandHandler("refresh", bot.refresh, filters=default_filter))
    app.add_handler(CommandHandler("undo", bot.undo, filters=default_filter))
    app.add_handler(CallbackQueryHandler(bot.button))
    app.add_handler(
        MessageHandler(
            default_filter
            & (filters.PHOTO | filters.VIDEO | filters.Document.IMAGE | filters.Document.VIDEO | filters.ANIMATION),
            bot.receive_new_media,
        )
    )

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
