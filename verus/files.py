import hashlib
import itertools
from pathlib import Path

from verus.const import SUPPORTED_EXTENSIONS_GLOB


def hash_file(file_path: Path) -> str:
    """Hash a file using SHA-256.

    Args:
        file_path (`str`):
            The path to the file to hash.

    Returns:
        `str`: The SHA-256 hash of the file.
    """
    sha256 = hashlib.sha256(file_path.read_bytes())
    return sha256.hexdigest()


def get_supported_files(root: Path, exclude_thumbs: bool = True) -> list[Path]:
    """Get all supported files in a directory recursively.

    Args:
        root (`Path`):
            The root directory to search for supported files.
        exclude_thumbs (`bool`, optional):
            Whether to exclude files with "thumb" in their name. Defaults to True.

    Returns:
        `list[Path]`: The paths of the supported files.
    """
    files = rglob_multiple_patterns(root, SUPPORTED_EXTENSIONS_GLOB)
    if exclude_thumbs:
        files = [path for path in files if "thumb" not in path.name]
    files.sort(key=lambda p: p.name)
    return files


def rglob_multiple_patterns(root: Path, patterns: list[str]) -> list[Path]:
    """Recursively glob files in a directory.

    Args:
        root (`Path`):
            The root directory to start globbing from.
        patterns (`list[str]`, optional):
            The patterns to glob for.

    Returns:
        `list[Path]`: The paths of the globbed files.
    """
    return list(itertools.chain.from_iterable(root.rglob(pattern) for pattern in patterns))
