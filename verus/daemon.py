import argparse
import json
import logging
import socket
from hashlib import sha256
from pathlib import Path
from typing import Any

import deepdanbooru as dd
import huggingface_hub as hug
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

from verus.utils import receive_all

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.experimental.numpy.experimental_enable_numpy_behavior()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

CACHE: dict[str, dict[str, Any]] = {}


class PredictDaemon:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("PredictDaemon")
        self.model = self.load_model()
        self.labels = self.load_labels()

    def _create_hash(self, data: Any) -> str:
        return sha256(data).hexdigest()

    def load_model(self) -> tf.keras.Model[Any, Any]:
        self.logger.info("Loading model")
        path = hug.hf_hub_download("public-data/DeepDanbooru", "model-resnet_custom_v3.h5")
        model = tf.keras.models.load_model(path)
        return model  # type: ignore[no-any-return]

    def load_labels(self) -> list[str]:
        self.logger.info("Loading labels")
        path = hug.hf_hub_download("public-data/DeepDanbooru", "tags.txt")
        with open(path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def prepare_image(self, image_input: str) -> np.ndarray[Any, Any]:
        self.logger.info(f"Loading image from path: {image_input}")
        image = Image.open(image_input)
        if image.mode != "RGB":
            image = image.convert("RGB")
        _, height, width, _ = self.model.input_shape
        np_image = np.asarray(image)
        np_image = tf.image.resize(
            np_image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True
        )
        np_image = np_image.numpy()  # type: ignore[attr-defined]
        np_image = dd.image.transform_and_pad_image(np_image, width, height)
        np_image = np_image / 255.0
        return np_image

    def predict(self, image: np.ndarray[Any, Any], score_threshold: float) -> dict[str, float]:
        self.logger.info("Performing prediction")
        probs = self.model.predict(image[None, ...])[0]
        probs = probs.astype(float)

        indices = np.argsort(probs)[::-1]
        tags = {}
        for index in indices:
            label = self.labels[index]
            prob = probs[index]
            tags[label] = prob
        return tags

    def _build_result(
        self,
        image_path: str | None,
        sha256: str | None,
        np_sha256: str,
        tags: dict[str, float],
        score_threshold: float,
    ) -> dict[str, Any]:
        return {
            "filtered_tags": {tag: prob for tag, prob in tags.items() if prob >= score_threshold},
            "tags": tags,
            "threshold": score_threshold,
            "image_path": image_path,
            "sha256": sha256,
            "np_sha256": np_sha256,
        }

    def _get_cache(self, sha256: str | None) -> dict[str, Any]:
        if sha256 and sha256 in CACHE:
            self.logger.info(f"Cache hit for {sha256}")
            return CACHE[sha256]
        return {}

    def _save_cache(self, tags: dict[str, float], np_sha256: str, sha256: str | None) -> None:
        entry = {
            "tags": tags,
            "np_sha256": np_sha256,
            "sha256": sha256,
        }
        CACHE[np_sha256] = entry
        if sha256:
            CACHE[sha256] = entry

    def handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        self.logger.info(f"Connected by {addr}")
        with conn:
            data = receive_all(conn)
            if data:
                data_dict = json.loads(data.decode())
                score_threshold = data_dict["score_threshold"]
                np_sha256: str = ""
                sha256: str | None = None
                cache_result: dict[str, Any] = {}
                image_path: str | None = None

                if "image_path" in data_dict:
                    image_input = data_dict["image_path"]
                    sha256 = self._create_hash(Path(image_input).read_bytes())

                    cache_result = self._get_cache(sha256)

                    if not cache_result:
                        image = self.prepare_image(image_input)
                        np_sha256 = self._create_hash(image)
                        cache_result = self._get_cache(np_sha256)
                else:
                    image = np.array(data_dict["image"])
                    np_sha256 = self._create_hash(image)
                    cache_result = self._get_cache(np_sha256)

                if not cache_result:
                    tags = self.predict(image, score_threshold)

                if cache_result:
                    tags = cache_result["tags"]
                    np_sha256 = cache_result["np_sha256"]
                    sha256 = cache_result["sha256"]

                result = {
                    "filtered_tags": {tag: prob for tag, prob in tags.items() if prob >= score_threshold},
                    "tags": tags,
                    "threshold": score_threshold,
                    "image_path": image_path,
                    "sha256": sha256,
                    "np_sha256": np_sha256,
                }

                self._save_cache(tags, np_sha256, sha256)

                conn.sendall(json.dumps(result).encode())

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            self.logger.info(f"Daemon listening on {self.host}:{self.port}")
            while True:
                conn, addr = s.accept()
                self.handle_client(conn, addr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Verus prediction daemon.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the daemon.")
    parser.add_argument("--port", type=int, default=65432, help="Port to run the daemon.")
    args = parser.parse_args()

    daemon = PredictDaemon(host=args.host, port=args.port)
    try:
        daemon.run()
    except KeyboardInterrupt:
        daemon.logger.info("Daemon stopped by user")


if __name__ == "__main__":
    main()
