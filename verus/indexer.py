import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verus.const import SUPPORTED_EXTENSIONS
from verus.db import Media, MediaTag, Tag, atomic
from verus.files import get_supported_files, hash_file
from verus.image import create_and_save_tg_thumbnail
from verus.utils import run_multiprocessed


class Indexer:
    """Index images in a directory.

    Args:
        image_dir (`Path`):
            The directory to index images from.
        extensions (`list[str]`, optional):
            The extensions of the images to index. Defaults to SUPPORTED_EXTENSIONS.

    Attributes:
        logger (`logging.Logger`):
            The logger for the indexer.
        image_dir (`Path`):
            The directory to index images from.
        extensions (`list[str]`):
            The extensions of the images to index.
    """

    def __init__(self, image_dir: Path, extensions: list[str] = SUPPORTED_EXTENSIONS):
        self.logger = logging.getLogger(__name__)

        self.image_dir = image_dir
        self.extensions = extensions

    def index_file(self, path: Path, hash: str, tag: Tag) -> tuple[Media, bool]:
        """Index a single image.

        Args:
            path (`Path`):
                The path to the image.
            hash (`str`):
                The hash of the image.
            tag (`Tag`):
                The default tag to assign to images, if the image is new or does not have any tags.

        Returns:
            `tuple[Media, bool]`:
                The indexed image and whether the image was new.
        """
        self.logger.info("Indexing single image %s", path)

        media = Media.get_or_none(Media.sha256 == hash)

        if media:
            self.logger.info("File %s already indexed", path)
            if not media.tags:
                media.tags.add(tag)
                media.save()
            return media, False

        media = Media.create(
            name=path.name,
            path=str(path),
            sha256=hash,
            group_id=self._extract_id(path.name),
        )
        if tag not in media.tags:
            media.tags.add(tag)
            media.save()
        return media, True

    def index_single(self, path: Path, tag: Tag, check_stale: bool = True) -> tuple[Media | None, bool]:
        """Index a single image.

        Note:
            Stale files will be removed from the DB.

        Args:
            path (`Path`):
                The path to the image.
            tag (`Tag`):
                The default tag to assign to images, if the image is new or does not have any tags.
            check_stale (`bool`, optional):
                Whether to check for stale files and mark them. Defaults to True.

        Returns:
            `tuple[Media | None, bool]`:
                The indexed image and whether the image was new.
        """
        self.logger.info("Indexing single image %s", path)
        file_hash = hash_file(path)

        media = Media.get_or_none(Media.sha256 == file_hash)
        file_exists = os.path.exists(path)
        if not file_exists and not media:
            return None, False
        elif check_stale and not file_exists and media:
            self.check_and_mark_as_stale(media)
            return media, False

        return self.index_file(path, file_hash, tag)

    def index(self, tag: Tag, check_stale: bool = True) -> tuple[list[Media], int]:
        """Index images in the directory.


        Args:
            tag (`Tag`):
                The default tag to assign to images.
            check_stale (`bool`, optional):
                Whether to check for stale files and mark them. Defaults to True.

        Returns:
            `tuple[list[Media], int]`: The inserted images and the number of stale images removed.
        """
        known_media: list[Media] = Media.select()

        if check_stale:
            self.logger.info("Check for stale media files")
            not_stale = Media.select().where(Media.stale == False).count()  # noqa: E712
            self.check_and_mark_all_stale()
            not_stale_after = Media.select().where(Media.stale == False).count()  # noqa: E712
            stale_files = not_stale - not_stale_after
            self.logger.info("Removed %d stale media files", stale_files)

        self.logger.info("Indexing files in %s", self.image_dir)
        with atomic():
            last_media: Media | None = Media.select().order_by(Media.id.desc()).first()
            last_id: int = 0 if not last_media else last_media.id

            files = get_supported_files(self.image_dir)
            files.sort(key=lambda path: path.stem)

            known_hashes: set[str] = {media.sha256 for media in known_media}
            new_files_v1 = set(files) - {Path(media.path) for media in known_media}

            self.logger.info("Found %d new media files", len(new_files_v1))
            hashes = self.load_image_hashes(list(new_files_v1))

            new_files_v2 = {file for file, hash in hashes.items() if hash not in known_hashes}

            if len(new_files_v1) < len(new_files_v2):
                for file in new_files_v1 - new_files_v2:
                    self.logger.warning("File %s already indexed", file)
                    file.unlink()
            if len(new_files_v2) == 0:
                return [], stale_files

            hashes = {file: hash for file, hash in hashes.items() if file in new_files_v2}

            self.logger.info("Inserting new media files into the database")
            Media.insert_many(
                [
                    {"name": file.name, "path": str(file), "sha256": hash, "group_id": self._extract_id(str(file))}
                    for file, hash in hashes.items()
                ]
            ).execute()

            new_media = Media.select(Media.id, Media.path).where(Media.id > last_id)
            MediaTag.insert_many([{"media_id": media.id, "tag_id": tag.id} for media in new_media]).execute()

        self._create_thumbnails([media.path for media in new_media])

        return list(new_media), stale_files

    def _extract_id(self, filename: str) -> str | None:
        match = re.search(r"_g?(\d+)_p\d+\.", filename)
        return match.group(1) if match else None

    def check_and_mark_as_stale(self, media: Media) -> bool:
        """Check and mark media as stale.

        Args:
            media_or_hash (`Media`):
                The media or hash to check for staleness.

        Returns:
            `bool`: Whether the media is stale.
        """
        if media.stale and os.path.exists(media.path):
            media.stale = False
            media.save()
            return False
        elif not media.stale and not os.path.exists(media.path):
            self.logger.error("Stale file %s marked", media.path)
            media.stale = True
            media.save()
            return True
        return media.stale

    def check_and_mark_all_stale(self) -> None:
        """Check and mark all applicable media as stale."""
        with atomic():
            media: list[Media] = list(Media.select())
            with ThreadPoolExecutor() as executor:
                stale_info = list(
                    executor.map(
                        os.path.exists,
                        [media.path for media in media],
                    ),
                )

                for media, exists in zip(media, stale_info):
                    if not exists and not media.stale:
                        media.stale = True
                        media.save()
                    elif exists and media.stale:
                        media.stale = False
                        media.save()

    def load_image_hashes(self, images: list[Path]) -> dict[Path, str]:
        """Load the hashes of images.

        Args:
            images (`list[Path]`):
                The images to load the hashes of.

        Returns:
            `dict[Path, str]`: The hashes of the images.
        """
        self.logger.info("Hashing %d images", len(images))
        return dict(zip(images, run_multiprocessed(hash_file, images, desc="Hashing images", ordered=True)))

    def _create_thumbnails(self, files: list[Path]) -> None:
        """Create thumbnails for media files.

        Args:
            files (`list[Path]`):
                The media files to create thumbnails for.
        """
        self.logger.info("Creating thumbnails for %d media files", len(files))
        run_multiprocessed(create_and_save_tg_thumbnail, files, desc="Creating thumbnails")
