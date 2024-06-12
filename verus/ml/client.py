import json
import socket
from base64 import b64encode
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

from verus.const import FileLike
from verus.files import hash_file
from verus.utils import receive_all


@dataclass
class Prediction:
    """Prediction data for an image or video file.

    Args:
        file_path (`Path`):
            Path to the image / video file.
        original_sha256 (`str`):
            SHA256 of the original image / video file.
        analysis_sha256 (`str`):
            SHA256 of the image / video thumbnail used for analysis.
        filtered_tags (`dict[str, float]`):
            Tags with probabilities above the threshold.
        tags (`dict[str, float]`):
            All tags with their probabilities.
        threshold (`float`):
            Threshold used for filtering.

    Attributes:
        file_path (`Path`):
            Path to the image / video file.
        original_sha256 (`str`):
            SHA256 of the original image / video file.
        analysis_sha256 (`str`):
            SHA256 of the image / video thumbnail used for analysis.
        filtered_tags (`dict[str, float]`):
            Tags with probabilities above the threshold.
        tags (`dict[str, float]`):
            All tags with their probabilities.
        threshold (`float`):
            Threshold used for filtering.
    """

    file_path: Path  # Path to the image / video file
    original_sha256: str  # SHA256 of the original image / video file
    analysis_sha256: str  # SHA256 of the image / video thumbnail used for analysis

    filtered_tags: dict[str, float]  # Tags with probabilities above the threshold
    tags: dict[str, float]  # All tags with their probabilities

    threshold: float  # Threshold used for filtering

    @property
    def tag_string(self) -> str:
        """Return a string of all tags

        Returns:
            str:
                A string of all tags
        """

        return ", ".join(self.tags)

    @property
    def filtered_tag_string(self) -> str:
        """Return a string of all filtered tags.

        Returns:
            str:
                A string of all filtered tags.
        """
        return ", ".join(self.filtered_tags)

    @classmethod
    def de_json(cls, data: str) -> "Prediction":
        """Create a Prediction object from a JSON string.

        Args:
            data (`str`):
                The JSON string to convert.

        Returns:
            `Prediction`:
                The Prediction object.
        """
        return cls.de_dict(json.loads(data))

    @classmethod
    def de_dict(cls, data: dict[str, Any]) -> "Prediction":
        """Create a Prediction object from a dict string.

        Args:
            data (`dict[str, Any]`):
                The dictionary to convert.

        Returns:
            `Prediction`:
                The Prediction object.
        """
        return cls(
            file_path=Path(data["file_path"]),
            original_sha256=data["original_sha256"],
            analysis_sha256=data["analysis_sha256"],
            filtered_tags=data["filtered_tags"],
            tags=data["tags"],
            threshold=data["threshold"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary.

        Returns:
            `dict[str, Any]`:
                The dictionary representation of the object.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert the object to a JSON string.

        Returns:
            `str`:
                The JSON representation of the object.
        """
        data = self.to_dict()
        data["file_path"] = str(data["file_path"])
        return json.dumps(data)

    def with_threshold(self, threshold: float) -> "Prediction":
        """Return a new Prediction object with a different threshold.

        Args:
            threshold (`float`):
                The new threshold to use.

        Returns:
            `Prediction`:
                The new Prediction object.
        """
        return Prediction(
            file_path=self.file_path,
            original_sha256=self.original_sha256,
            analysis_sha256=self.analysis_sha256,
            filtered_tags={tag: prob for tag, prob in self.filtered_tags.items() if prob >= threshold},
            tags=self.tags,
            threshold=threshold,
        )


class PredictClient:
    """Client for the prediction server.

    Args:
        host (`str`, optional):
            The host of the server. Defaults to "localhost".
        port (`int`, optional):
            The port of the server. Defaults to 65432.

    Attributes:
        host (`str`):
            The host of the server.
        port (`int`):
            The port of the server.
    """

    def __init__(self, host: str = "localhost", port: int = 65432):
        self.host = host
        self.port = port

    @contextmanager
    def create_connection(self) -> Generator[socket.socket, None, None]:
        """Create a connection to the server.

        Yields:
            `socket.socket`:
                The connection to the server.
        """
        with self._create_or_use_connection() as conn:
            yield conn

    @contextmanager
    def _create_or_use_connection(self, conn: socket.socket | None = None) -> Generator[socket.socket, None, None]:
        """Create a connection to the server if not provided.

        Args:
            conn (`socket.socket`, optional):
                The connection to use, if provided.

        Yields:
            `socket.socket`:
                The connection to the server.
        """
        try:
            if conn is None:
                conn = self._connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def _connect(self) -> socket.socket:
        """Connect to the server.

        Returns:
            `socket.socket`: The connection to the server.
        """
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((self.host, self.port))
        return conn

    def _request(self, data: Any, conn: socket.socket | None = None) -> dict[str, Any] | list[Any]:
        """Send a request to the server and receive a response.

        Args:
            data (`Any`):
                The data to send to the server.
            conn (`socket.socket`, optional):
                The connection to use, if provided.

        Returns:
            `Any`:
                The response from the server.
        """
        with self._create_or_use_connection(conn) as conn:
            conn.sendall(json.dumps(data).encode() + b"\n")
            response = receive_all(conn)

            return json.loads(response.decode())  # type: ignore[no-any-return]

    def to_base64(self, data: bytes) -> str:
        """Convert bytes to a base64 string.

        Args:
            data (`bytes`):
                The bytes to convert.

        Returns:
            `str`:
                The base64 encoded string.
        """
        return b64encode(data).decode()

    def predict(
        self,
        file_path: Path,
        score_threshold: float,
        file: FileLike | None = None,
        conn: socket.socket | None = None,
    ) -> Prediction:
        """Predict tags for an image or video file.

        Args:
            file_path (`Path`):
                Path to the image / video file.
            score_threshold (`float`):
                The threshold for filtering tags.
            file (`FileLike`, optional):
                The file to use for prediction.
            conn (`socket.socket`, optional):
                The connection to use, if provided.

        Raises:
            ValueError:
                If the file does not exist and file is not provided.
                If the response from the server is bad.
                If the server returns an error.

        Returns:
            `Prediction`:
                The prediction data for the file.
        """
        if not file_path.is_file() and file is None:
            raise ValueError("Either existing file_path or file must be provided")

        file = file or file_path
        file_data = b64encode(file.getvalue()).decode() if isinstance(file, BytesIO) else None

        data: dict[str, Any] = {
            "original_file_path": str(file_path),
            "original_file_sha256": hash_file(file_path),
            "score_threshold": score_threshold,
            "file_path": str(file) if isinstance(file, Path) else None,
            "file_data": file_data,
        }

        response = self._request(data, conn)

        if isinstance(response, list):
            raise ValueError("Bad response from server")

        if "error" in response:
            raise ValueError(response["error"])

        return Prediction.de_dict(response)
