import itertools
import logging
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def run_multiprocessed(
    func: Callable[..., T], /, *iterable: Any, desc: str | None = None, ordered: bool = False
) -> list[T]:
    """Run a function in parallel processes.

    Args:
        func (`Callable`):
            The function to run in parallel processes.
        *iterable (`Any`):
            The arguments to pass to the function.
        desc (`str`, optional):
            The description to use for the progress bar. Defaults to None.

    Returns:
        `list[T]`: The results of the function for each argument.

    """
    with ProcessPoolExecutor() as executor:
        if not ordered:
            futures = [executor.submit(func, *args) for args in zip(*iterable)]
            return list(future.result() for future in tqdm(as_completed(futures), total=len(iterable[0]), desc=desc))
        else:
            return list(tqdm(executor.map(func, *iterable), total=len(iterable[0]), desc=desc))


def bool_emoji(value: bool) -> str:
    """Return an emoji representing a boolean value.

    Args:
        value (`bool`):
            The boolean value to represent.

    Returns:
        `str`: The emoji representing the boolean value.
    """
    return "✅" if value else "❌"


def receive_all(sock: socket.socket, buffer_size: int = 4096) -> bytes:
    """Receive all data from a socket, given the data ends with a newline.

    Args:
        sock (`socket.socket`):
            The socket to receive data from.
        buffer_size (`int`, optional):
            The buffer size to use when receiving data. Defaults to 4096.

    Returns:
        `bytes`: The received data.
    """
    data = b""
    while True:
        part = sock.recv(buffer_size)
        if part[-1:] == b"\n":
            data += part[:-1]
            break
        data += part
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
    """Decorator to use tqdm logging context for a function.

    Args:
        func (`Callable`):
            The function to decorate.

    Returns:
        `Callable`: The decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tqdm_logging_context():
            return func(*args, **kwargs)

    return wrapper


def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> Iterable[Iterable[T]]:
    """Iterates over an iterator chunk by chunk.

    Taken from https://stackoverflow.com/a/8998040.
    See also https://github.com/huggingface/huggingface_hub/pull/920#discussion_r938793088.

    Args:
        iterable (`Iterable`):
            The iterable on which we want to iterate.
        chunk_size (`int`):
            Size of the chunks. Must be a strictly positive integer (e.g. >0).

    Yields:
        `Iterable[T]`: The next chunk of the iterable.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("`chunk_size` must be a strictly positive integer (>0).")

    iterator = iter(iterable)
    while True:
        try:
            next_item = next(iterator)
        except StopIteration:
            return
        yield itertools.chain((next_item,), itertools.islice(iterator, chunk_size - 1))
