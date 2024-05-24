import logging
import socket
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

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
