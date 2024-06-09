import logging
from pathlib import Path

from tqdm import tqdm

from verus.const import SUPPORTED_EXTENSIONS
from verus.db import DATABASE, Media, Tag
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

    def index(self) -> list[Media]:
        """Index images in the directory.

        Returns:
            `list[Media]`: The images that were indexed.
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
            Media.insert_many([{"path": str(image), "sha256": hash} for image, hash in hashes.items()]).execute()
            inserted_images = Media.select().where(Media.id > last_id)

            for path, image in tqdm(zip(new_images, inserted_images), desc="Adding tags"):
                tag = tags[path.parent.name]
                if tag not in image.tags:
                    image.tags.add(tag)
                    image.save()

        self._create_thumbnails([image.path for image in inserted_images])

        return list(inserted_images)

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
