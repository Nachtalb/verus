from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

TG_MAX_TOTAL_IMAGE_SIDE_LENGTH = 10_000
TG_MAX_IMAGE_BYTES = 10_000_000  # 10 MB
TG_MAX_IMAGE_RATIO = 20

ImageLike = Path | BytesIO | Image.Image


def first_frame(video_path: Path) -> Image.Image:
    """Extract the first frame from a video file.

    Args:
        video_path (Path): Path to the video file

    Returns:
        Image.Image: First frame of the video
    """
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to extract frame from video")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def create_tg_thumbnail_from_video(video_path: Path, max_side_length: int = -1) -> Image.Image:
    """Create a Telegram compatible thumbnail from a video file.

    Args:
        video_path (Path): Path to the video file
        max_side_length (int): Maximum side length, defaults to -1 == no resizing

    Returns:
        Image.Image: Telegram compatible thumbnail
    """
    return create_tg_thumbnail(first_frame(video_path), max_side_length)


def create_tg_thumbnail(image: ImageLike, max_side_length: int = -1) -> Image.Image:
    """Create a Telegram compatible thumbnail.

    Args:
        image (Path | BytesIO | Image.Image): Image to convert
        max_side_length (int): Maximum side length, defaults to -1 == no resizing

    Returns:
        Image.Image: Telegram compatible thumbnail
    """
    image = _get_pil_image(image)
    if max_side_length != -1:
        image = resize_image_to(image, max_side_length, max_side_length)

    image = create_tg_compatible_image(image)
    image = flatten_image(image)

    return image


def create_and_save_tg_thumbnail(image: Path, max_side_length: int = -1, output_path: Path | None = None) -> Path:
    """Create and save a Telegram compatible thumbnail.

    Args:
        image (Path): Image to convert
        max_side_length (int): Maximum side length, defaults to -1 == no resizing
        output_path (Path): Output path, defaults to None == same directory as image

    Returns:
        Path: Path to the thumbnail
    """
    image = Path(image)
    thumb_path = output_path or image.with_name(f"{image.stem}.thumb.jpg")
    if not thumb_path.is_file():
        if image.suffix in [".mp4", ".webm"]:
            thumb = create_tg_thumbnail_from_video(image, max_side_length)
        else:
            thumb = create_tg_thumbnail(image, max_side_length)

        thumb.save(thumb_path, format="JPEG", quality=70)
        thumb.close()

    return thumb_path


def flatten_image(image: ImageLike) -> Image.Image:
    """Flatten image if it has an alpha channel.

    Args:
        image (Path | BytesIO | Image.Image): Image to convert

    Returns:
        Image.Image: Flattened image
    """
    image = _get_pil_image(image)
    if image.mode == "RGBA":
        white = Image.new("RGB", image.size, "WHITE")
        white.paste(image, (0, 0), image)
        image = white

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def create_tg_compatible_image(image: ImageLike) -> Image.Image:
    """Pipline to make image Telegram compatible.

    Args:
        image (Path | BytesIO | Image.Image): Image to convert

    Returns:
        Image.Image: Telegram compatible image
    """
    image = _get_pil_image(image)
    image = crop_image_to_ratio(image)
    if sum(image.size) > TG_MAX_TOTAL_IMAGE_SIDE_LENGTH:
        image = resize_image_max_side_length(image, TG_MAX_TOTAL_IMAGE_SIDE_LENGTH)
    return image


def crop_image_to_ratio(image: ImageLike, max_ratio: int = TG_MAX_IMAGE_RATIO) -> Image.Image:
    """Crop image if the width to height ratio is too high.

    Args:
        image (Path | BytesIO | Image.Image): Image to convert
        max_ratio (int): Maximum width to height ratio, defaults to TG_MAX_IMAGE_RATIO (20)

    Returns:
        Image.Image: Resized Image
    """
    image = _get_pil_image(image)

    if image.width / image.height > max_ratio:
        image = image.crop(
            (
                image.width // 2 - image.height // 2,
                0,
                image.width // 2 + image.height // 2,
                image.height,
            )
        )
    elif image.height / image.width > max_ratio:
        image = image.crop(
            (
                0,
                image.height // 2 - image.width // 2,
                image.width,
                image.height // 2 + image.width // 2,
            )
        )
    return image


def resize_image_to(image: ImageLike, max_width: int, max_height: int) -> Image.Image:
    """Resize an image to fit within the specified dimensions while maintaining the aspect ratio.

    Args:
        image (Path | BytesIO | Image.Image): Image to resize
        max_width (int): Maximum width
        max_height (int): Maximum height

    Returns:
        Image: Resized images
    """

    image = _get_pil_image(image)

    width, height = image.size
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)

    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return image.resize((width, height))


def resize_image_max_side_length(image: ImageLike, max_total_side_length: int) -> Image.Image:
    """Resize an image to fit within the specified total side length while maintaining the aspect ratio.

    Args:
        image (Path | BytesIO | Image.Image): Image to resize
        max_total_side_length (int): Maximum total side length

    Returns:
        Image: Resized images
    """
    image = _get_pil_image(image)

    width, height = image.size
    total_side_length = width + height

    if total_side_length > max_total_side_length:
        new_total_side_length = max_total_side_length
        new_width = int(new_total_side_length * (width / total_side_length))
        new_height = int(new_total_side_length * (height / total_side_length))

        return image.resize((new_width, new_height))

    return image


def _get_pil_image(image: ImageLike) -> Image.Image:
    """Get a PIL image from a file, BytesIO or PIL image.

    Args:
        image (Path | BytesIO | Image.Image): Image to convert

    Returns:
        Image.Image: PIL image
    """
    if isinstance(image, (Path, BytesIO)):
        return Image.open(image)
    return image
