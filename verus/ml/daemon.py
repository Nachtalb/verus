import argparse
import json
import logging
import socket
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from typing import Any

import deepdanbooru as dd
import huggingface_hub as hug
import magic
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

from verus.const import FileLike, ImageLike
from verus.files import hash_bytes
from verus.image import first_frame, flatten_image
from verus.ml.client import Prediction
from verus.utils import receive_all

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.experimental.numpy.experimental_enable_numpy_behavior()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class PredictDaemon:
    """Daemon for predicting tags for images and videos.

    Args:
        host (str):
            The host to run the daemon on.
        port (int):
            The port to run the daemon on.

    Attributes:
        host (str):
            The host to run the daemon on.
        port (int):
            The port to run the daemon on.
        logger (logging.Logger):
            The logger for the daemon.
        model (tf.keras.Model):
            The loaded model for prediction.
        labels (list[str]):
            The labels for the model.
        _cache_file (Path):
            Path to the cache file.
        cache (dict[str, Prediction]):
            Cache for storing predictions.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("PredictDaemon")
        self.model = self.load_model()
        self.labels = self.load_labels()

        self._cache_file = Path("cache.jsonl")
        self.cache = self.load_cache()

    def load_cache(self) -> dict[str, Prediction]:
        """Load the cache from a jsonlines file.

        Returns:
            dict[str, Prediction]:
                The loaded cache.
        """
        cache = {}
        if not self._cache_file.is_file():
            return {}

        with self._cache_file.open() as f:
            for line in f:
                data = json.loads(line)
                prediction = Prediction.de_dict(data)
                cache[prediction.original_sha256] = prediction
        return cache

    def save_to_cache(self, prediction: Prediction) -> None:
        """Add prediction to cache jsonlines file.

        Args:
            prediction (Prediction):
                The prediction to add to the cache.
        """
        self.cache[prediction.original_sha256] = prediction
        with self._cache_file.open("a") as f:
            f.write(json.dumps(prediction.to_dict()) + "\n")

    def load_model(self) -> tf.keras.Model:  # type: ignore[type-arg]
        """Load the model for prediction.

        Returns:
            tf.keras.Model:
                The loaded model.
        """
        self.logger.info("Loading model")
        path = hug.hf_hub_download("public-data/DeepDanbooru", "model-resnet_custom_v3.h5")
        model = tf.keras.models.load_model(path)
        return model  # type: ignore[no-any-return]

    def load_labels(self) -> list[str]:
        """Load the labels for the model.

        Returns:
            list[str]:
                The labels for the model.
        """
        self.logger.info("Loading labels")
        path = hug.hf_hub_download("public-data/DeepDanbooru", "tags.txt")
        with open(path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def prepare_image(self, image: ImageLike) -> np.ndarray[Any, Any]:
        """Prepare an image for prediction.

        Args:
            image (Path | BytesIO | Image.Image):
                The image to prepare.

        Returns:
            np.ndarray:
                The prepared image.
        """
        self.logger.info("Preparing image")
        image = image if isinstance(image, Image.Image) else Image.open(image)
        image = flatten_image(image)
        np_image = np.asarray(image)

        _, height, width, _ = self.model.input_shape
        np_image = tf.image.resize(
            image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True
        )
        np_image = np_image.numpy()  # type: ignore[attr-defined]
        np_image = dd.image.transform_and_pad_image(np_image, width, height)
        np_image = np_image / 255.0
        return np_image

    def predict(self, image: np.ndarray[Any, Any]) -> dict[str, float]:
        """Predict the tags for an image.

        Args:
            image (np.ndarray):
                The image to predict tags for.

        Returns:
            dict[str, float]:
                The predicted tags and their probabilities.
        """
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

    def _get_file_type(self, file_path: FileLike) -> tuple[str, str]:
        """Get the MIME type of the file.

        Args:
            file_path (Path):
                Path to the file.

        Returns:
            tuple[str, str]:
                Tuple containing the file type and subtype.
        """
        mime = magic.Magic(mime=True)
        if isinstance(file_path, Path):
            mime_type = mime.from_file(str(file_path))
        else:
            mime_type = mime.from_buffer(file_path.getvalue())
        return tuple(mime_type.split("/"))  # type: ignore[return-value]

    def _prepare_file(self, file: FileLike) -> np.ndarray[Any, Any]:
        """Prepare a file for prediction.

        Args:
            file (Path | BytesIO):
                The file to prepare.

        Raises:
            ValueError:
                If the file does not exist.
                If the file type is not supported.

        Returns:
            np.ndarray:
                The prepared image.
        """
        if isinstance(file, Path) and not file.is_file():
            raise ValueError(f"File {file} does not exist")

        file_type, sub_type = self._get_file_type(file)

        match file_type:
            case "image":
                analysis_file = self.prepare_image(file)
            case "video":
                frame = first_frame(file)
                analysis_file = self.prepare_image(frame)
            case _:
                raise ValueError(f"Unsupported file type: {file_type}/{sub_type}")
        return analysis_file

    def prepare_and_predict(self, file: FileLike) -> tuple[dict[str, float], str]:
        """Prepare and predict the tags for a file.

        Args:
            file (Path | BytesIO):
                The file to predict tags for.

        Returns:
            tuple[dict[str, float], str]:
                Tuple containing the tags and the SHA-256 hash of the analysis file.
        """
        analysis_file = self._prepare_file(file)
        data = self.predict(analysis_file)
        return data, hash_bytes(analysis_file.tobytes())

    def predict_handle(self, data: dict[str, Any]) -> Prediction:
        """Predict the tags for a file.

        Args:
            data (dict[str, Any]):
                The data to predict tags for.
                Either "file_path" or "file_data" must be provided. file_path takes precedence.
                {
                    "original_file_path": str,    # path to the original file
                    "original_file_sha256": str,  # SHA-256 hash of the original file
                    "file_path": str,             # file to analyse (eg. thumbnail for videos)
                    "file_data": str,             # base64 encoded file to analyse (eg. thumbnail for videos)
                    "score_threshold": float      # threshold for filtering tags
                }

        Raises:
            ValueError:
                If the file data is missing.

        Returns:
            Prediction:
                The prediction object.
        """
        original_file_path = data["original_file_path"]
        original_file_sha256 = data["original_file_sha256"]

        self.logger.info(f"Predicting for {original_file_path} with SHA-256: {original_file_sha256}")

        if cache_result := self.cache.get(original_file_sha256):
            self.logger.info(f"Cache hit for {original_file_sha256}")
            if cache_result.file_path != original_file_path:
                cache_result.file_path = original_file_path
            if (threshold := data.get("score_threshold", 0.4)) and cache_result.threshold != threshold:
                cache_result = cache_result.with_threshold(threshold)
            return cache_result

        file_path: str | None = data.get("file_path")
        file_data: str | None = data.get("file_data")

        file: FileLike
        if file_path:
            file = Path(file_path)
        elif file_data:
            file = BytesIO(b64decode(file_data))
        else:
            raise ValueError('File data missing, either "file_path" or "file_data" must be provided')

        score_threshold = data.get("score_threshold", 0.4)

        tags, analysis_sha256 = self.prepare_and_predict(file)

        prediction = Prediction(
            file_path=original_file_path,
            original_sha256=original_file_sha256,
            analysis_sha256=analysis_sha256,
            filtered_tags={tag: prob for tag, prob in tags.items() if prob >= score_threshold},
            tags=tags,
            threshold=score_threshold,
        )

        self.save_to_cache(prediction)
        return prediction

    def send(self, conn: socket.socket, data: dict[Any, Any] | list[Any]) -> None:
        """Send data to the client as JSON.

        Args:
            conn (socket.socket):
                The connection to send the data to.
            data (dict[Any, Any] | list[Any]):
                The data to send.
        """
        conn.sendall(json.dumps(data).encode() + b"\n")

    def handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        """Handle a client connection.

        Args:
            conn (socket.socket):
                The connection to the client.
            addr (tuple[str, int]):
                The address of the client.
        """
        self.logger.info(f"Connected by {addr}")
        with conn:
            raw_data = receive_all(conn)
            if not raw_data:
                self.send(conn, {"error": "No data received"})

            try:
                data = json.loads(raw_data.decode())
            except json.JSONDecodeError:
                self.send(conn, {"error": "Invalid JSON data"})
                return

            try:
                prediction = self.predict_handle(data)
                self.send(conn, prediction.to_dict())
            except Exception as exc:
                self.logger.exception(f"Error occurred: {exc}")
                self.send(conn, {"error": str(exc)})
                return

    def run(self) -> None:
        """Run the daemon."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            self.logger.info(f"Daemon listening on {self.host}:{self.port}")
            while True:
                try:
                    conn, addr = s.accept()
                    self.handle_client(conn, addr)
                except Exception as e:
                    self.logger.exception(f"Error occurred: {e}")


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
