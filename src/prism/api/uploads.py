"""Validate and store uploaded screenshots.

Nothing from the client is trusted: not the declared content type, not the file
name, not the extension. The bytes are decoded by Pillow, checked, and written
back out as a fresh file with a server-generated name. Re-encoding also drops
EXIF metadata and anything appended after the image data.
"""

import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PureWindowsPath

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Pillow format name -> extension we store it under.
ALLOWED_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
MIN_SIDE = 32
_CHUNK = 1024 * 1024
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class UploadRejected(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class StoredImage:
    key: str
    width: int
    height: int


class BodySizeLimit:
    """Refuse request bodies over max_bytes before they are read.

    Starlette spools multipart file parts to disk with no size cap, and FastAPI
    parses the form before the route runs, so without this a multi-GB upload
    would land on disk before read_limited ever saw it.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        detail = f"Request body is larger than {self.max_bytes // (1024 * 1024)} MB"
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > self.max_bytes:
            await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
            return

        seen = 0

        async def counted() -> Message:
            # Chunked uploads have no Content-Length: count as the body arrives.
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise HTTPException(413, detail)
            return message

        await self.app(scope, counted, send)


async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in chunks and stop as soon as it goes over the limit."""
    buf = bytearray()
    while chunk := await file.read(_CHUNK):
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise UploadRejected(413, f"File is larger than {max_bytes // (1024 * 1024)} MB")
    if not buf:
        raise UploadRejected(400, "File is empty")
    return bytes(buf)


def store_image(data: bytes, upload_dir: Path, max_pixels: int) -> StoredImage:
    """Decode, check and re-encode an image. Blocking: call it from a thread."""
    try:
        with Image.open(BytesIO(data)) as probe:
            fmt = probe.format or ""
            width, height = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise UploadRejected(415, "Only PNG, JPEG and WebP images are accepted")
            # Check dimensions from the header before decoding any pixels.
            if width * height > max_pixels:
                raise UploadRejected(413, "Image has too many pixels")
            if min(width, height) < MIN_SIDE:
                raise UploadRejected(422, f"Image must be at least {MIN_SIDE}px on each side")
            probe.verify()

        # verify() leaves the image unusable, so open again to decode.
        with Image.open(BytesIO(data)) as img:
            img.load()
            clean = _normalise_mode(img)
    except UploadRejected:
        raise
    except Image.DecompressionBombError as exc:
        raise UploadRejected(413, "Image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise UploadRejected(422, "File is not a readable image") from exc

    ext = ALLOWED_FORMATS[fmt]
    key = f"{uuid.uuid4().hex}.{ext}"
    _atomic_save(clean, upload_dir, key, fmt)
    return StoredImage(key=key, width=width, height=height)


def resolve_key(upload_dir: Path, key: str) -> Path:
    """Map a stored key back to a path, refusing anything that escapes upload_dir."""
    base = upload_dir.resolve()
    path = (base / key).resolve()
    if path.parent != base:
        raise ValueError("invalid image key")
    return path


def display_name(filename: str | None) -> str | None:
    """Keep the user's file name for display only: last path part, no control chars."""
    if not filename:
        return None
    name = PureWindowsPath(filename).name  # handles both / and \ separators
    name = _CONTROL_CHARS.sub("", name).strip()
    return name[:255] or None


def _normalise_mode(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img.copy()
    if img.mode in ("LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA")
    return img.convert("RGB")


def _atomic_save(img: Image.Image, upload_dir: Path, key: str, fmt: str) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    options = {"quality": 95} if fmt in ("JPEG", "WEBP") else {}
    fd, tmp = tempfile.mkstemp(dir=upload_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            img.save(fh, format=fmt, **options)
        os.replace(tmp, upload_dir / key)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
