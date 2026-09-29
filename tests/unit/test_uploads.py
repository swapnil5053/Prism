import asyncio
import struct
import zlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image

from prism.api.uploads import (
    UploadRejected,
    display_name,
    read_limited,
    resolve_key,
    store_image,
)

MAX_PIXELS = 4_000_000


def encode(fmt: str, size: tuple[int, int] = (64, 48), mode: str = "RGB", **kw: object) -> bytes:
    buf = BytesIO()
    Image.new(mode, size, "white").save(buf, format=fmt, **kw)
    return buf.getvalue()


def png_header_only(width: int, height: int) -> bytes:
    """A PNG that claims huge dimensions but has no pixel data."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


@pytest.mark.parametrize(("fmt", "ext"), [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp")])
def test_accepts_supported_formats(tmp_path: Path, fmt: str, ext: str) -> None:
    stored = store_image(encode(fmt), tmp_path, MAX_PIXELS)
    assert stored.key.endswith(f".{ext}")
    assert (stored.width, stored.height) == (64, 48)
    assert (tmp_path / stored.key).is_file()


def test_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected) as err:
        store_image(encode("GIF"), tmp_path, MAX_PIXELS)
    assert err.value.status_code == 415


def test_rejects_non_image(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected) as err:
        store_image(b"<script>alert(1)</script>", tmp_path, MAX_PIXELS)
    assert err.value.status_code == 422


def test_rejects_truncated_image(tmp_path: Path) -> None:
    data = encode("PNG", (256, 256))
    with pytest.raises(UploadRejected):
        store_image(data[: len(data) // 2], tmp_path, MAX_PIXELS)


def test_rejects_pixel_bomb_before_decoding(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected) as err:
        store_image(png_header_only(50_000, 50_000), tmp_path, MAX_PIXELS)
    assert err.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_rejects_tiny_image(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected) as err:
        store_image(encode("PNG", (8, 8)), tmp_path, MAX_PIXELS)
    assert err.value.status_code == 422


def test_strips_exif_and_trailing_bytes(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x010F] = "SecretCamera"  # Make
    data = encode("JPEG", exif=exif.tobytes()) + b"PAYLOAD-AFTER-EOI"
    stored = store_image(data, tmp_path, MAX_PIXELS)
    saved = (tmp_path / stored.key).read_bytes()
    assert b"SecretCamera" not in saved
    assert b"PAYLOAD-AFTER-EOI" not in saved


def test_keeps_transparency(tmp_path: Path) -> None:
    stored = store_image(encode("PNG", mode="RGBA"), tmp_path, MAX_PIXELS)
    with Image.open(tmp_path / stored.key) as img:
        assert img.mode == "RGBA"


def test_no_partial_files_left(tmp_path: Path) -> None:
    store_image(encode("PNG"), tmp_path, MAX_PIXELS)
    assert not list(tmp_path.glob("*.part"))


def test_resolve_key_refuses_traversal(tmp_path: Path) -> None:
    assert resolve_key(tmp_path, "abc.png") == (tmp_path / "abc.png").resolve()
    for bad in ["../etc/passwd", "sub/abc.png", "/etc/passwd", ".."]:
        with pytest.raises(ValueError):
            resolve_key(tmp_path, bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("screen.png", "screen.png"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\shot.png", "shot.png"),
        ("bad\x00name\n.png", "badname.png"),
        ("", None),
        (None, None),
        ("a" * 400, "a" * 255),
    ],
)
def test_display_name(raw: str | None, expected: str | None) -> None:
    assert display_name(raw) == expected


def test_read_limited_stops_at_limit() -> None:
    upload = UploadFile(BytesIO(b"x" * 3000))
    with pytest.raises(UploadRejected) as err:
        asyncio.run(read_limited(upload, max_bytes=2048))
    assert err.value.status_code == 413


def test_read_limited_rejects_empty() -> None:
    with pytest.raises(UploadRejected) as err:
        asyncio.run(read_limited(UploadFile(BytesIO(b"")), max_bytes=10))
    assert err.value.status_code == 400


def test_rejects_pixel_count_over_configured_limit(tmp_path: Path) -> None:
    # Below Pillow's own bomb threshold, so only our limit catches it.
    with pytest.raises(UploadRejected) as err:
        store_image(png_header_only(3000, 2000), tmp_path, MAX_PIXELS)
    assert err.value.status_code == 413
