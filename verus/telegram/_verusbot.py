import asyncio
import hashlib
import logging
import re
import shutil
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
    File,
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
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from verus.const import TG_MAX_DOWNLOAD_SIZE
from verus.db import History, Media, MediaTag, Tag, User, atomic
from verus.db.history import history_action
from verus.files import get_supported_files, hash_bytes, hash_file, is_supported
from verus.image import create_and_save_tg_thumbnail, get_dimensions
from verus.indexer import Indexer
from verus.telegram._veruscontext import VerusContext
from verus.telegram._webhookupdate import WebhookUpdate
from verus.telegram._webroutehandler import WebRouteHandler
from verus.utils import bool_emoji, chunk_iterable


class VerusBot:
    _intermediate_group_message: tuple[Message, ...] = ()

    def __init__(
        self,
        dir: Path,
        default_tag: str,
        import_folder: Path,
        local_mode: bool,
    ):
        self.logger = logging.getLogger(__name__)
        self.dir = dir
        self.default_tag = Tag.get_or_create(default_tag)
        self.import_folder = import_folder
        self.local_mode = local_mode

        self._tag_mode: bool = False
        self._group_mode: bool = True

        self.tags = Tag.select()
        self.indexer = Indexer(self.dir)

    def setup_hooks(self, application: Application, web_prefix: str) -> None:  # type: ignore[type-arg]
        user_filters = [filters.User(user.telegram_id) for user in User.select().where(User.telegram_id.is_null(False))]
        multi_user_filter = user_filters[0]
        if len(user_filters) > 1:
            multi_user_filter = reduce(lambda x, y: x | y, user_filters)  # type: ignore[arg-type, return-value]

        default_filter = filters.ChatType.PRIVATE & multi_user_filter

        application.add_handler(CommandHandler("start", self.start, filters=default_filter))
        application.add_handler(CommandHandler("info", self.info, filters=default_filter))
        application.add_handler(CommandHandler("refresh", self.refresh, filters=default_filter))
        application.add_handler(CommandHandler("undo", self.undo, filters=default_filter))
        application.add_handler(CallbackQueryHandler(self.button))
        application.add_handler(
            MessageHandler(
                default_filter
                & (filters.PHOTO | filters.VIDEO | filters.Document.IMAGE | filters.Document.VIDEO | filters.ANIMATION),
                self.receive_new_media,
            )
        )

        application.add_handler(WebRouteHandler(self.import_file, method="POST", path="/import", prefix=web_prefix))

    async def import_file(self, update: WebhookUpdate, context: VerusContext) -> None:
        if not update.values:
            return

        if "file" not in update.values:
            return

        __import__("ipdb").set_trace()

        self.logger.info("Importing file %s", update.values["file"])

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
            f"Mode: {'toggle' if self._tag_mode else 'move'}\n"
            f"Name: <code>{Path(media.path).name}</code>\n"
            f"Group ID: <code>{media.group_id or '-'}</code>\n"
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

        import_files = get_supported_files(self.import_folder)
        if import_files:
            message_text += f"\nImporting {len(import_files)} files..."
            await message.edit_text(message_text)
            hash_dict = self.indexer.load_image_hashes(import_files)

            for file, hash in hash_dict.items():
                new_path = self.dir / file.name
                if new_path.exists():
                    new_path = new_path.with_stem(f"{file.stem}_{hash}")
                file.rename(new_path)

            message_text += "done."
        else:
            message_text += "\nNo files to import."

        message_text += "\nIndexing..."
        await message.edit_text(message_text)

        total_images = Media.select().count()
        total_tags = Tag.select().count()

        new, removed = self.indexer.index(self.default_tag)

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
                    callback_data=("toggle" if self._tag_mode else "move", media.path, cat.name),
                )
                for cat in self.tags
            ],
            3,
        )

        buttons = [
            [InlineKeyboardButton("Continue", callback_data=("continue", media.path, ""))],
            [
                InlineKeyboardButton(
                    "As Previous: " + ", ".join(map(str, self._previous_tags(media))),
                    callback_data=("same", media.path, ""),
                )
            ],
            *[list(row) for row in categories],
            [
                InlineKeyboardButton(
                    "Mode: " + ("Toggle" if self._tag_mode else "Move"), callback_data=("toggle_mode", "", "")
                ),
                InlineKeyboardButton(
                    "Group: " + bool_emoji(self._group_mode), callback_data=("toggle_group", media.path, "")
                ),
                InlineKeyboardButton("More", callback_data=("more", media.path, "")),
            ],
        ]

        if latest_action := History.latest_action():
            buttons.append([InlineKeyboardButton("Undo", callback_data=("undo", latest_action.media.path, ""))])

        return InlineKeyboardMarkup(buttons)

    async def _clear_intermediate_group_message(self) -> None:
        if self._intermediate_group_message:
            msg = self._intermediate_group_message[0]
            bot = msg.get_bot()
            await bot.delete_messages(msg.chat.id, [m.message_id for m in self._intermediate_group_message])
        self._intermediate_group_message = ()

    async def ask_for_group_action(self, media: Media, action: str, move_category: str, query: CallbackQuery) -> None:
        group = media.get_group(non_processed_only=True)

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

    def _previous_tags(self, exclude: Media | None = None) -> list[Tag]:
        last_action = History.latest_action(exclude)
        if not last_action:
            return []

        return list(last_action.media.tags)

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
        if action == "same":
            action = "continue"
            media = Media.get_or_none(Media.path == path)
            previous_tags = self._previous_tags(media)
            if not previous_tags:
                await query.answer("No last action found.")
                return

            with history_action(media, action="same"):
                media.tags.clear()
                for tag in previous_tags:
                    media.tags.add(tag)

        if action in ["continue", "move"] and not self._group_mode:
            action = "simple_" + action

        if action in ["continue", "move", "same"]:
            media = Media.get_or_none(Media.path == path)

            group = media.get_group(non_processed_only=True)

            if not group or len(group) == 1 or not self._group_mode:
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

                self._group_mode = False
                await query.answer("Turned off group ask.")

            with atomic():
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

            group = media.get_group(non_processed_only=True)
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

            with atomic():
                if action == "toggle":
                    self._toggle(media, category)
        elif action == "more":
            media = Media.get_or_none(Media.path == path)
            group = media.get_group()
            if not group or not query.message:
                await query.answer("No group found.")
            else:
                await self.send_media_group(group, query.message)  # type: ignore[arg-type]
        elif action == "toggle_group":
            self._group_mode = not self._group_mode
            await query.answer(f"Group Ask: {bool_emoji(self._group_mode)}")
        elif action == "toggle_mode":
            self._tag_mode = not self._tag_mode
            await query.answer(f"Current Mode: {'Toggle' if self._tag_mode else 'Move'}")
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

        for user in User.select().where(User.telegram_id.is_null(False)):
            if not user.telegram_id:
                continue
            self.logger.info("Sending message to %s", user.telegram_id)
            await application.bot.send_message(user.telegram_id, "Media Sorting Bot started.")

    async def post_stop(self, application: Application) -> None:  # type: ignore[type-arg]
        self.logger.info("Media Sorting Bot stopped.")
        for user in User.select().where(User.telegram_id.is_null(False)):
            if not user.telegram_id:
                continue
            self.logger.info("Sending message to %s", user.telegram_id)
            await application.bot.send_message(user.telegram_id, "Media Sorting Bot stopped.")

    def hash_str_to_int(self, s: str) -> int:
        hash_value = int(hashlib.sha256(s.encode()).hexdigest(), 16)
        return (hash_value & 0xFFFFFFFFFF) + 0x80000000  # Masking to 48 bits and adding base value (1b)

    async def get_file_for_download(
        self, message: Message, check_file_ext: bool = False
    ) -> tuple[File, PhotoSize | Video | Document | Animation]:
        obj: PhotoSize | Video | Document | Animation | None
        if message.photo:
            obj = message.photo[-1]
        else:
            obj = message.video or message.document or message.animation

        if not obj:
            raise ValueError("No media found.")

        if not obj.file_size:
            raise ValueError("File size not available.")
        elif not self.local_mode and obj.file_size > TG_MAX_DOWNLOAD_SIZE:
            raise ValueError("File is too large. Max size is 20 MB.")

        file = await obj.get_file()
        if check_file_ext:
            if not file.file_path:
                raise ValueError("Can't determine file extension.")

            if not is_supported(Path(file.file_path)):
                raise ValueError("File type not supported.")

        return file, obj

    async def receive_new_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return

        msg_id = update.message.message_id

        # check validity
        try:
            file, obj = await self.get_file_for_download(update.message, check_file_ext=True)
        except ValueError as e:
            self.logger.warning("Error receiving media: %s", e)
            await update.message.reply_text(str(e), reply_to_message_id=msg_id)
            return

        # get group id from media group
        group_id_str = ""
        if group_id := update.message.media_group_id:
            group_id_str = f"_g{self.hash_str_to_int(group_id)}_p{update.message.message_id}"

        # progress message for large files
        progress_msg = None
        if file.file_size > TG_MAX_DOWNLOAD_SIZE:  # type: ignore[operator]
            progress_msg = await update.message.reply_text("Receiving media ...", reply_to_message_id=msg_id)

        # download / copy file to local storage
        source: BytesIO | Path
        new_file_path = Path(file.file_path)  # type: ignore[arg-type]
        if self.local_mode and new_file_path.exists():
            hash = hash_file(file.file_path)
            source = new_file_path
        else:
            out = BytesIO()
            await file.download_to_memory(out)
            out.seek(0)
            hash = hash_bytes(out.getvalue())
            source = out

        file_path = self.dir / f"{hash}{group_id_str}{new_file_path.suffix}"
        if isinstance(source, BytesIO):
            file_path.write_bytes(source.getvalue())
        else:
            shutil.copyfile(source, file_path)

        # index new file
        with atomic():
            media, new = self.indexer.index_single(file_path, self.default_tag)

            if progress_msg:
                await progress_msg.delete()

            if not media:
                # this case should not happen, but better safe than sorry
                await update.message.reply_text(
                    "Media could not be imported.",
                    reply_to_message_id=msg_id,
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            if media and not media.tg_file_info and not isinstance(obj, (Animation, Document)):
                media.tg_file_info = obj.to_dict()
                media.save()

        if new:
            await update.message.reply_text(
                f"Media `{media.sha256[:8]}` received. Use /refresh then /start to process it.",
                reply_to_message_id=msg_id,
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            already_processed = "" if media.processed else "Use /start to process it."
            await update.message.reply_text(
                f"Media `{media.sha256[:8]}` already exists.{already_processed}",
                reply_to_message_id=msg_id,
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