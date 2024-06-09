from io import BytesIO
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = ["jpeg", "jpg", "png", "gif", "webp", "webm", "mp4"]
VIDEO_EXTENSIONS = ["webm", "mp4"]
IMAGE_EXTENSIONS = ["jpeg", "jpg", "png", "gif", "webp"]

SUPPORTED_EXTENSIONS_GLOB = ["*." + ext for ext in SUPPORTED_EXTENSIONS]

TG_MAX_TOTAL_IMAGE_SIDE_LENGTH = 10_000
TG_MAX_IMAGE_BYTES = 10_000_000  # 10 MB
TG_MAX_IMAGE_RATIO = 20

THUMB_MAX_SIDE_LENGTH = 1024

ImageLike = Path | BytesIO | Image.Image
