import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from peewee import Query, fn
from tabulate import tabulate
from tqdm import tqdm

from verus.cli.filter import Node, parse_node
from verus.db import History, Media, Tag, User, atomic, setup_db
from verus.files import get_supported_files, hash_file, is_supported
from verus.image import create_and_save_tg_thumbnail
from verus.indexer import Indexer
from verus.ml.client import PredictClient
from verus.utils import run_multiprocessed, with_tqdm_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Verus:
    """Verus - Image Tag Prediction and Organisation Tool.

    Args:
        tags_path (`Path`):
            Path to the tags configuration file.
        host (`str`, optional):
            Host the daemon runs on. Defaults to "localhost".
        port (`int`, optional):
            Port the daemon runs on. Defaults to 65432.

    Attributes:
        client (`PredictClient`):
            The client to communicate with the prediction daemon.
        tags (`dict[str, Node]`):
            The tags configuration.
        logger (`logging.Logger`):
            The logger for the Verus instance.
    """

    def __init__(self, tags_path: Path, host: str = "localhost", port: int = 65432):
        self.client = PredictClient(host, port)
        self.tags = self.load_tags(tags_path)
        self.logger = logging.getLogger("Verus")

    def load_tags(self, tags_path: Path) -> dict[str, Node]:
        """Load the tags configuration from a file.

        Args:
            tags_path (`Path`):
                Path to the tags configuration file.

        Returns:
            `dict[str, Node]`: The tags configuration.
        """
        return {tag: parse_node(data) for tag, data in json.loads(tags_path.read_text()).items()}

    def identify_main_tag(self, tags: dict[str, float], nodes: dict[str, Node]) -> str:
        """Identify the main tag for a set of tags.

        Args:
            tags (`dict[str, float]`):
                The tags to identify.
            nodes (`dict[str, Node]`):
                The tag nodes to evaluate.

        Returns:
            `str`: The main tag.
        """
        for tag, node in nodes.items():
            if node.evaluate(tags):
                return tag
        return ""

    def identify(self, args: Namespace) -> None:
        """Identify tags for a single file.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Identifying image %s", args.file)
        if not args.file.is_file():
            raise ValueError("File does not exist")
        elif not is_supported(args.file):
            raise ValueError("File type not supported")

        prediction = self.client.predict(args.file, args.threshold)
        main_tag = self.identify_main_tag(prediction.filtered_tags, self.tags)

        if args.save_json:
            json_path = args.image.with_suffix(".json")
            json_path.write_text(prediction.to_json())

        print(f"Image: {prediction.file_path}")
        print(f"SHA256: {prediction.original_sha256}")
        print(f"Main tag: {main_tag}")
        print(tabulate(prediction.filtered_tags.items(), headers=["Tag", "Probability"]))

    def _identify_main_tag(self, tags: dict[str, float], nodes: dict[str, Node]) -> str | None:
        """Identify the main tag for a set of tags.

        Args:
            tags (`dict[str, float]`):
                The tags to identify.
            nodes (`dict[str, Node]`):
                The tag nodes to evaluate.

        Returns:
            `str | None`: The main tag or None if no tag is identified.
        """
        for tag, node in nodes.items():
            if node.evaluate(tags):
                return tag
        return None

    @with_tqdm_logging
    def move(self, args: Namespace) -> None:
        """Move images based on their predicted tags.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Moving images from %s to %s", args.path, args.output)
        setup_db()

        indexer = Indexer(args.path)
        files = get_supported_files(args.path)

        for file in tqdm(files):
            hash = hash_file(file)

            if Media.get_or_none(Media.sha256 == hash):
                self.logger.info("Image %s already indexed", file.name)
                continue

            prediction = self.client.predict(file, args.threshold)
            main_tag = self._identify_main_tag(prediction.filtered_tags, self.tags) or args.default_tag

            new_path = args.output / file.name
            if new_path.is_file():
                new_path = new_path.with_name(f"{file.stem}_{prediction.original_sha256[:8]}{file.suffix}")

            self.logger.info(f"{main_tag} \t{file.name}")
            if not args.dry_run:
                file.rename(new_path)

                if args.save_json:
                    json_path = new_path.with_suffix(".json")
                    data = prediction.to_json() if args.full_json else prediction.filtered_tags
                    json_path.write_text(json.dumps(data, indent=2))

                indexer.index_file(new_path, hash, Tag.get_or_create(main_tag))

    @with_tqdm_logging
    def thumbs(self, args: Namespace) -> None:
        """Generate thumbnails for images.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Generating thumbnails for images in %s", args.path)
        files = set(get_supported_files(args.path))
        run_multiprocessed(create_and_save_tg_thumbnail, files)

    def setup(self, args: Namespace) -> None:
        """Setup the database.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Setting up the database")
        setup_db()

        system_user = User.get_or_none(User.username == "system")
        if not system_user:
            User.create(username="system", role="system")
            self.logger.info("System user created")

    def user_add(self, args: Namespace) -> None:
        """Create a new user.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Creating user %s", args.username)
        setup_db()
        if User.get_or_none(User.username == args.username):
            self.logger.warning("User %s already exists", args.username)
            exit(1)

        user = User.create(username=args.username, role=args.role, telegram_id=args.telegram_id)
        self.logger.info("User %s created with API key %s", user.username, user.api_key)

    def user_edit(self, args: Namespace) -> None:
        """Edit a user.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Editing user %s", args.username)
        setup_db()
        user = User.get_or_none(User.username == args.username)
        if not user:
            self.logger.warning("User %s does not exist", args.username)
            exit(1)

        if args.role:
            user.role = args.role
        if args.telegram_id:
            user.telegram_id = args.telegram_id

        user.save()
        self.logger.info("User %s edited", user.username)

    def user_del(self, args: Namespace) -> None:
        """Delete a user.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Deleting user %s", args.username)
        setup_db()
        user = User.get_or_none(User.username == args.username)
        if not user:
            self.logger.warning("User %s does not exist", args.username)
            exit(1)

        user.delete_instance()
        self.logger.info("User %s deleted", args.username)

    def _list_users(self, users: list[User]) -> None:
        print(
            tabulate(
                ((user.username, user.partial_api_key(), user.role, user.telegram_id) for user in users),
                headers=["Username", "API Key", "Role", "Telegram ID"],
            )
        )

    def user_list(self, args: Namespace) -> None:
        """List all users.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Listing all users")
        setup_db()
        users = User.select()
        self._list_users(users)

    def user_reset(self, args: Namespace) -> None:
        """Reset a user's API key.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Resetting API key for user %s", args.username)
        setup_db()
        user = User.get_or_none(User.username == args.username)
        if not user:
            self.logger.warning("User %s does not exist", args.username)
            exit(1)

        new_key = user.recreate_api_key()
        self.logger.info("API key for user %s reset to %s", user.username, new_key)

    def user_show(self, args: Namespace) -> None:
        """Show a user's API key.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Showing API key for user %s", args.username)
        setup_db()
        user = User.get_or_none(User.username == args.username)
        if not user:
            self.logger.warning("User %s does not exist", args.username)
            exit(1)

        self._list_users([user])

    def migrate(self, args: Namespace) -> None:
        """Migrate the database.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Migrating the database to %s", args.new_path)
        setup_db()

        with atomic():
            medias = Media.update(path=fn.REPLACE(Media.path, args.prev_path, args.new_path)).execute()
            for entry in tqdm(History.select(), desc="Migrating histories"):
                entry.before = json.loads(json.dumps(entry.before).replace(args.prev_path, args.new_path))
                entry.after = json.loads(json.dumps(entry.after).replace(args.prev_path, args.new_path))
                entry.save()

        self.logger.info("Migrated %d medias and %d histories", medias, History.select().count())

    def list_media(self, args: Namespace) -> None:
        """List all media in the database.

        Args:
            args (`Namespace`):
                The parsed arguments.
        """
        self.logger.info("Listing all media")
        setup_db()

        media: Query = Media.select()
        if args.limit:
            media = media.limit(args.limit)

        if args.stale:
            media.where(Media.stale == True)  # noqa: E712

        if args.desc:
            media = media.order_by(Media.id.desc())

        items: list[Media] = media

        dict_items = []
        for item in items:
            dikt = item.to_dict()
            if not args.path:
                dikt.pop("path")
            dikt["tg_file_info"] = "..." if item.tg_file_info else None
            dikt["tags"] = ", ".join(tag["tag"]["name"] for tag in dikt.pop("mediatagthrough_set"))
            dict_items.append(dikt)

            dikt["last_action"] = next(iter(dikt.pop("history")), {}).get("action", None)  # type: ignore[call-overload]
            if len(item.name) > 30:
                dikt["name"] = item.name[:20] + "..." + item.name[-7:]

        print(tabulate(dict_items, headers="keys"))


def main() -> None:
    parser = ArgumentParser(description="Verus - Image Tag Prediction and Organisation Tool")
    parser.add_argument("--tags", type=Path, default=Path("tags.json"), help="Path to the tags configuration file.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the daemon.")
    parser.add_argument("--port", type=int, default=65432, help="Port to run the daemon.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True

    db_parser = sub_parsers.add_parser("db", help="Manage the database.")
    db_sub_parsers = db_parser.add_subparsers()
    db_sub_parsers.required = True

    db_setup_parser = db_sub_parsers.add_parser("setup", help="Setup the database.")
    db_setup_parser.set_defaults(func="setup")

    db_media_parser = db_sub_parsers.add_parser("media", help="Manage media.")
    db_media_sub_parsers = db_media_parser.add_subparsers()
    db_media_sub_parsers.required = True

    db_media_list_parser = db_media_sub_parsers.add_parser("list", help="List all media in the database.")
    db_media_list_parser.add_argument("--limit", type=int, default=10, help="Limit the number of results.")
    db_media_list_parser.add_argument("--desc", action="store_true", help="Sort results in descending order.")
    db_media_list_parser.add_argument("--path", action="store_true", help="Show the full path of the media.")
    db_media_list_parser.add_argument("--stale", action="store_true", help="Show only stale media.")
    db_media_list_parser.set_defaults(func="list_media")

    db_migrate_parser = db_sub_parsers.add_parser("migrate", help="Migrate the database.")
    db_migrate_parser.add_argument("prev_path", type=str, help="Path to the old files")
    db_migrate_parser.add_argument("new_path", type=str, help="Path to the new files")
    db_migrate_parser.set_defaults(func="migrate")

    db_user_parser = db_sub_parsers.add_parser("user", help="Manage users.")
    db_user_sub_parsers = db_user_parser.add_subparsers()
    db_user_sub_parsers.required = True

    db_user_add_parser = db_user_sub_parsers.add_parser("add", help="Create a new user.")
    db_user_add_parser.add_argument("username", type=str, help="Username of the new user.")
    db_user_add_parser.add_argument("--role", type=str, default="user", help="Role of the new user.")
    db_user_add_parser.add_argument("--telegram-id", type=int, default=None, help="Telegram ID of the new user.")
    db_user_add_parser.set_defaults(func="user_add")

    db_user_edit_parser = db_user_sub_parsers.add_parser("edit", help="Edit a user.")
    db_user_edit_parser.add_argument("username", type=str, help="Username of the user to edit.")
    db_user_edit_parser.add_argument("--role", type=str, default=None, help="Role of the user.")
    db_user_edit_parser.add_argument("--telegram-id", type=int, default=None, help="Telegram ID of the user.")
    db_user_edit_parser.set_defaults(func="user_edit")

    db_user_del_parser = db_user_sub_parsers.add_parser("del", help="Delete a user.")
    db_user_del_parser.add_argument("username", type=str, help="Username of the user to delete.")
    db_user_del_parser.set_defaults(func="user_del")

    db_user_list_parser = db_user_sub_parsers.add_parser("list", help="List all users.")
    db_user_list_parser.set_defaults(func="user_list")

    db_user_reset_parser = db_user_sub_parsers.add_parser("reset", help="Reset a user's api key.")
    db_user_reset_parser.add_argument("username", type=str, help="Username of the user to reset.")
    db_user_reset_parser.set_defaults(func="user_reset")

    db_user_show_parser = db_user_sub_parsers.add_parser("show", help="Show a user's api key.")
    db_user_show_parser.add_argument("username", type=str, help="Username of the user to show.")
    db_user_show_parser.set_defaults(func="user_show")

    thumbs_parser = sub_parsers.add_parser("thumbs", help="Generate thumbnails for images.")
    thumbs_parser.add_argument("path", type=Path, help="Input folder containing PNG and JPEG images.")
    thumbs_parser.set_defaults(func="thumbs")

    mv_parser = sub_parsers.add_parser("move", help="Move images based on their predicted tags.")
    mv_parser.add_argument("path", type=Path, help="Input folder containing PNG and JPEG images.")
    mv_parser.add_argument("output", type=Path, help="Output folder to move the tagged images.")
    mv_parser.add_argument("--threshold", type=float, default=0.4, help="Score threshold for tag filtering.")
    mv_parser.add_argument("--save-json", action="store_true", help="Save the filtered tags as a JSON file.")
    mv_parser.add_argument("--full-json", action="store_true", help="Save the full prediction result as a JSON file.")
    mv_parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without actually moving files.")
    mv_parser.add_argument("--default-tag", type=str, default="tag", help="Default tag for unmatched files.")
    mv_parser.set_defaults(func="move")

    identify_parser = sub_parsers.add_parser("identify", help="Identify tags for a single image.")
    identify_parser.add_argument("file", type=Path, help="Path to the image / video file.")
    identify_parser.add_argument("--threshold", type=float, default=0.4, help="Score threshold for tag filtering.")
    identify_parser.add_argument("--save-json", action="store_true", help="Save prediction result as a JSON file.")
    identify_parser.set_defaults(func="identify")

    args = parser.parse_args()

    verus = Verus(args.tags, args.host, args.port)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    if args.verbose:
        verus.logger.setLevel(logging.DEBUG)

    try:
        if func := getattr(verus, args.func, None):
            func(args)
        else:
            raise ValueError(f"Invalid function {args.func}")
    except KeyboardInterrupt:
        verus.logger.info("Verus stopped by user")
    except Exception as e:
        if args.verbose:
            verus.logger.exception(e)
        verus.logger.error(f"Verus stopped due to an error, {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
