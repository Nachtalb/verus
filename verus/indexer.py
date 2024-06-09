import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from verus.const import SUPPORTED_EXTENSIONS
from verus.db import DATABASE, Media, MediaTag, Tag
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

    def index_single(self, path: Path) -> Media | None:
        """Index a single image.

        Args:
            path (`Path`):
                The path to the image.

        Returns:
            `Media | None`: The indexed image or None if the image does not exist.
        """
        self.logger.info("Indexing single image %s", path)
        if not path.exists() or not path.is_file():
            self.logger.error("File %s does not exist", path)
            if media := Media.get_or_none(Media.path == str(path)):
                self.delete_media(media)
            return None

        with DATABASE.atomic():
            tag = Tag.get_or_create(path.parent.name)
            if media := Media.get_or_none(Media.path == str(path)):
                self.logger.info("Image %s already indexed", path)
                if tag not in media.tags:
                    media.tags.add(tag)
                    media.save()
                return media  # type: ignore[no-any-return]

            hash = hash_file(path)
            media = Media.create(name=path.name, path=str(path), sha256=hash, group_id=self._extract_id(str(path)))
            media.tags.add(tag)
            media.save()

        return media  # type: ignore[no-any-return]

    def index(self) -> tuple[list[Media], int]:
        """Index images in the directory.

        Returns:
            `tuple[list[Media], int]`: The inserted images and the number of stale images removed.
        """
        self.logger.info("Indexing images in %s", self.image_dir)

        with DATABASE.atomic():
            tags = {
                path.name: Tag.get_or_create(path.name)
                for path in self.image_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            }

            last_media: Media | None = Media.select().order_by(Media.id.desc()).first()
            last_id: int = 0 if not last_media else last_media.id

            images = get_supported_files(self.image_dir)
            images.sort(key=lambda path: path.stem)

            known_images = Media.select()
            new_images = set(images) - {Path(image.path) for image in known_images}

            self.logger.info("Found %d new images", len(new_images))
            hashes = self._load_image_hashes(list(new_images))

            self.logger.info("Inserting new images into the database")
            Media.insert_many(
                [
                    {"name": image.name, "path": str(image), "sha256": hash, "group_id": self._extract_id(str(image))}
                    for image, hash in hashes.items()
                ]
            ).execute()
            inserted_images = Media.select().where(Media.id > last_id)

            for path, image in tqdm(zip(new_images, inserted_images), desc="Adding tags"):
                tag = tags[path.parent.name]
                if tag not in image.tags:
                    image.tags.add(tag)
                    image.save()

        self.logger.info("Check for stale images")
        counter = 0
        for image, exists in self._check_for_stales(known_images):
            if exists:
                continue

            self.delete_media(image)
            counter += 1

        self._create_thumbnails([image.path for image in inserted_images])

        return list(inserted_images), counter

    def delete_media(self, media: Media) -> None:
        """Delete a media including its dependencies.

        Args:
            media (`Media`):
                The media to delete.
        """
        self.logger.info("Removing stale image %s", media.path)
        MediaTag.delete().where(MediaTag.media_id == media.id).execute()
        media.delete_instance(recursive=True)
        thumb = Path(f"{media.path}.thumb.jpg")
        if thumb.exists():
            thumb.unlink()

    def _extract_id(self, filename: str) -> str | None:
        match = re.search(r"_g?(\d+)_p\d+\.", filename)
        return match.group(1) if match else None

    def _check_for_stales(self, medias: list[Media]) -> list[tuple[Media, bool]]:
        """Check for stale files.

        Args:
            medias (`list[Media]`):
                The images to check for staleness.

        Returns:
            `list[Media]`: The stale medias.
        """
        self.logger.info("Checking for stale images")
        with ThreadPoolExecutor() as executor:
            return list(zip(medias, executor.map(self._check_file_exists, [media.path for media in medias])))

    def _check_file_exists(self, path: str) -> bool:
        """Check if a file exists.

        Args:
            path (`str`):
                The path to check.

        Returns:
            `bool`: Whether the file exists.
        """
        return os.path.exists(path)

    def _load_image_hashes(self, images: list[Path]) -> dict[Path, str]:
        """Load the hashes of images.

        Args:
            images (`list[Path]`):
                The images to load the hashes of.

        Returns:
            `dict[Path, str]`: The hashes of the images.
        """
        self.logger.info("Hashing %d images", len(images))
        return dict(zip(images, run_multiprocessed(hash_file, images, desc="Hashing images", ordered=True)))

    def _create_thumbnails(self, images: list[Path]) -> None:
        """Create thumbnails for images.

        Args:
            images (`list[Path]`):
                The images to create thumbnails for.
        """
        self.logger.info("Creating thumbnails for %d images", len(images))
        run_multiprocessed(create_and_save_tg_thumbnail, images, desc="Creating thumbnails")
