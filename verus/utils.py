import logging
import socket
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from PIL.Image import Image
from tqdm import tqdm


def receive_all(sock: socket.socket, buffer_size: int = 4096) -> bytes:
    data = b""
    while True:
        part = sock.recv(buffer_size)
        data += part
        if len(part) < buffer_size:
            break
    return data


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler to ensure logging messages appear above tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get the log message string
        msg = self.format(record)
        # Clear the tqdm progress bar and print the log message
        tqdm.write(msg)


@contextmanager
def tqdm_logging_context() -> Generator[None, None, None]:
    """Context manager to temporarily replace the global console handler with a TqdmLoggingHandler."""

    # Get the root logger
    root_logger = logging.getLogger()

    # Find the current console handler (if any)
    existing_handler = None
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            existing_handler = handler
            break

    # Create a TqdmLoggingHandler with the same formatter and level
    tqdm_handler = TqdmLoggingHandler()
    if existing_handler:
        tqdm_handler.setLevel(existing_handler.level)
        tqdm_handler.setFormatter(existing_handler.formatter)

        # Replace the existing console handler with the TqdmLoggingHandler
        root_logger.removeHandler(existing_handler)
        root_logger.addHandler(tqdm_handler)

    else:
        root_logger.addHandler(tqdm_handler)

    try:
        yield
    finally:
        # Restore the original console handler
        root_logger.removeHandler(tqdm_handler)
        if existing_handler:
            root_logger.addHandler(existing_handler)


def with_tqdm_logging(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to use tqdm logging context for a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tqdm_logging_context():
            return func(*args, **kwargs)

    return wrapper


def resize_image_to(image: Image, max_width: int, max_height: int) -> Image:
    """Resize an image to fit within the specified dimensions while maintaining the aspect ratio."""

    width, height = image.size
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)

    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return image.resize((width, height))


def resize_image_max_side_length(image: Image, max_total_side_length: int) -> Image:
    """Resize an image to fit within the specified total side length while maintaining the aspect ratio."""

    width, height = image.size
    total_side_length = width + height

    if total_side_length > max_total_side_length:
        new_total_side_length = max_total_side_length
        new_width = int(new_total_side_length * (width / total_side_length))
        new_height = int(new_total_side_length * (height / total_side_length))

        return image.resize((new_width, new_height))

    return image
