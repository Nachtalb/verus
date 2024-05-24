import json
import logging
from argparse import ArgumentParser, Namespace
from itertools import chain
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm

from verus.client import PredictClient
from verus.filter import Node, parse_node
from verus.utils import with_tqdm_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Verus:
    def __init__(self, tags_path: Path, host: str = "localhost", port: int = 65432):
        self.client = PredictClient(host, port)
        self.tags = self.load_tags(tags_path)
        self.logger = logging.getLogger("Verus")

    def load_tags(self, tags_path: Path) -> dict[str, Node]:
        return {tag: parse_node(data) for tag, data in json.loads(tags_path.read_text()).items()}

    def identify_main_tag(self, tags: dict[str, float], nodes: dict[str, Node]) -> str:
        for tag, node in nodes.items():
            if node.evaluate(tags):
                return tag
        return ""

    def identify(self, args: Namespace) -> None:
        self.logger.info("Identifying image %s", args.image)
        if not args.image.is_file():
            raise ValueError("Output must be a file")
        elif args.image.suffix not in [".png", ".jpeg", ".jpg"]:
            raise ValueError("Output must be a png or jpeg file")

        prediction = self.client.predict(args.image, args.threshold)
        main_tag = self.identify_main_tag(prediction.filtered_tags, self.tags)
        json_data = prediction.to_json() if args.full_json else prediction.filtered_tags

        if args.save_json:
            json_path = args.image.with_suffix(".json")
            json_path.write_text(json.dumps(json_data, indent=2))

        if not args.output_json:
            print(f"Image: {prediction.image_path}")
            print(f"SHA256: {prediction.sha256}")
            print(f"Main tag: {main_tag}")
            print(tabulate(prediction.filtered_tags.items(), headers=["Tag", "Probability"]))
        else:
            print(json.dumps(json_data, indent=2))

    def _identify_new_dir_name(self, tags: dict[str, float], nodes: dict[str, Node]) -> str | None:
        for tag, node in nodes.items():
            if node.evaluate(tags):
                return tag
        return None

    @with_tqdm_logging
    def move(self, args: Namespace) -> None:
        self.logger.info("Moving images from %s to %s", args.path, args.output)

        types = ["*.png", "*.jpeg", "*.jpg"]
        files = list(chain(*[args.path.rglob(t) for t in types]))

        for file in tqdm(files):
            prediction = self.client.predict(file, args.threshold)
            dir_name = self._identify_new_dir_name(prediction.filtered_tags, self.tags)

            if not dir_name and not args.no_move_unknown:
                dir_name = args.unknown_dir

            if dir_name:
                new_path = args.output / dir_name / file.name
                self.logger.info(f"{dir_name} \t{file.name}")
                if not args.dry_run:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    file.rename(new_path)

                    if args.save_json:
                        json_path = new_path.with_suffix(".json")
                        data = prediction.to_json() if args.full_json else prediction.filtered_tags
                        json_path.write_text(json.dumps(data, indent=2))
            else:
                self.logger.info(f"skipped \t{file.name}")


def main() -> None:
    parser = ArgumentParser(description="Verus - Image Tag Prediction and Organisation Tool")
    parser.add_argument("--tags", type=Path, default=Path("tags.json"), help="Path to the tags configuration file.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the daemon.")
    parser.add_argument("--port", type=int, default=65432, help="Port to run the daemon.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    sub_parsers = parser.add_subparsers()

    mv_parser = sub_parsers.add_parser("move", help="Move images based on their predicted tags.")
    mv_parser.add_argument("path", type=Path, help="Input folder containing PNG and JPEG images.")
    mv_parser.add_argument("output", type=Path, help="Output folder to move the tagged images.")
    mv_parser.add_argument("--threshold", type=float, default=0.4, help="Score threshold for tag filtering.")
    mv_parser.add_argument("--save-json", action="store_true", help="Save the filtered tags as a JSON file.")
    mv_parser.add_argument("--full-json", action="store_true", help="Save the full prediction result as a JSON file.")
    mv_parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without actually moving files.")
    mv_parser.add_argument("--unknown-dir", type=str, default="unknown", help="Directory for unmatched images.")
    mv_parser.add_argument("--no-move-unknown", action="store_true", help="Do not move unmatched files.")
    mv_parser.set_defaults(func="move")

    identify_parser = sub_parsers.add_parser("identify", help="Identify tags for a single image.")
    identify_parser.add_argument("image", type=Path, help="Path to the image file (PNG or JPEG).")
    identify_parser.add_argument("--threshold", type=float, default=0.4, help="Score threshold for tag filtering.")
    identify_parser.add_argument("--save-json", action="store_true", help="Save the filtered tags as a JSON file.")
    identify_parser.add_argument(
        "--output-json", action="store_true", help="Output the prediction result as a JSON file."
    )
    identify_parser.add_argument(
        "--full-json", action="store_true", help="Include full prediction details in the JSON output."
    )
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
            verus.logger.error(e, exc_info=True)
        verus.logger.error(f"Verus stopped due to an error, {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
