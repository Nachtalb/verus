import json
import socket
from base64 import b64encode
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

from PIL import Image

from verus.const import ImageLike
from verus.utils import receive_all


@dataclass
class Prediction:
    filtered_tags: dict[str, float]
    tags: dict[str, float]

    np_sha256: str  # SHA256 of the images np array

    threshold: float
    image_path: Path | None = None
    sha256: str | None = None  # SHA256 of the image file if available

    filtered_tag_string: str = field(init=False)
    tag_string: str = field(init=False)

    def __post_init__(self) -> None:
        self.filtered_tag_string = ", ".join(self.filtered_tags)
        self.tag_string = ", ".join(self.tags)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Prediction":
        return cls(
            filtered_tags=data["filtered_tags"],
            tags=data["tags"],
            threshold=data["threshold"],
            image_path=Path(data["image_path"]) if data["image_path"] else None,
            sha256=data.get("sha256"),
            np_sha256=data["np_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        data = self.to_dict()
        data["image_path"] = str(data["image_path"]) if data["image_path"] else None
        return json.dumps(data)

    def with_threshold(self, threshold: float) -> "Prediction":
        return Prediction(
            filtered_tags={tag: prob for tag, prob in self.filtered_tags.items() if prob >= threshold},
            tags=self.tags,
            threshold=threshold,
            image_path=self.image_path,
            sha256=self.sha256,
            np_sha256=self.np_sha256,
        )


class PredictClient:
    def __init__(self, host: str = "localhost", port: int = 65432):
        self.host = host
        self.port = port

    @contextmanager
    def create_connection(self) -> Generator[socket.socket, None, None]:
        with self._create_or_use_connection() as conn:
            yield conn

    @contextmanager
    def _create_or_use_connection(self, conn: socket.socket | None = None) -> Generator[socket.socket, None, None]:
        try:
            if conn is None:
                conn = self._connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def _connect(self) -> socket.socket:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((self.host, self.port))
        return conn

    def _request(self, data: Any, conn: socket.socket | None = None) -> Any:
        with self._create_or_use_connection(conn) as conn:
            conn.sendall(json.dumps(data).encode() + b"\n")
            response = receive_all(conn)

            return json.loads(response.decode())

    def to_base64(self, data: bytes) -> str:
        return b64encode(data).decode()

    def predict(
        self,
        image: ImageLike,
        score_threshold: float,
        conn: socket.socket | None = None,
    ) -> Prediction:
        data: dict[str, Any] = {
            "score_threshold": score_threshold,
        }
        if isinstance(image, Path):
            data["image_path"] = str(image)
        elif isinstance(image, BytesIO):
            data["image"] = self.to_base64(image.getvalue())
        elif isinstance(image, Image.Image):
            bytes_io = BytesIO()
            image.save(bytes_io, format="JPEG", quality=70)
            data["image"] = self.to_base64(bytes_io.getvalue())
        else:
            raise ValueError("Unsupported image type")

        response = self._request(data, conn)
        if not response["image_path"] and isinstance(image, Path):
            response["image_path"] = str(image)

        return Prediction.from_json(response)
