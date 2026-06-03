"""Pure-stdlib PNG writer — zero external dependencies.

Radiance writes 8-bit RGB PNGs using only `zlib` and `struct` from the standard
library (no PIL, no numpy). A PNG is: signature + IHDR + IDAT (zlib-deflated
scanlines, each prefixed with a 0 filter byte) + IEND, each chunk carrying a
CRC32. That's the whole format we need.
"""
from __future__ import annotations

import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(path: str, width: int, height: int, rgb: bytes) -> str:
    """Write an 8-bit RGB PNG. `rgb` is width*height*3 bytes, row-major."""
    if len(rgb) != width * height * 3:
        raise ValueError(f"expected {width*height*3} bytes, got {len(rgb)}")
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, colour-type 2 (RGB)
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)                              # filter type 0 (none) per scanline
        raw.extend(rgb[y * stride:(y + 1) * stride])
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", idat))
        f.write(_chunk(b"IEND", b""))
    return path
