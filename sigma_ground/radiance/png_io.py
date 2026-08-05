"""Minimal PNG codec — stdlib only (zlib + struct), 8-bit RGB, non-interlaced.

The R0 pixel-regression net commits its baselines as PNG so a human can eyeball
a diff, but the AUTHORITATIVE artifact is the canonical RGB byte stream
(top-down rows, no alpha) whose SHA-256 lives in the baselines manifest.
``write_rgb_png`` uses filter 0 on every scanline, so
``sha256(read_rgb_png(p)[2]) == manifest sha256_rgb`` always — every committed
PNG is self-verifying.

Zero-dependency doctrine: no PIL, no numpy. ``read_rgb_png`` decodes all five
standard filters (0-4) so a baseline that passed through a PNG editor still
reads, and honestly rejects anything that is not 8-bit RGB non-interlaced.
"""
from __future__ import annotations

import struct
import zlib

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_rgb_png(path: str, w: int, h: int, rgb: bytes) -> None:
    """Write ``rgb`` (top-down rows, 3 bytes/pixel, len == w*h*3) as a PNG.

    Filter 0 on every scanline (the byte stream stays literal inside the
    zlib layer), zlib level 9.
    """
    if len(rgb) != w * h * 3:
        raise ValueError(f"rgb stream is {len(rgb)} bytes, expected {w}x{h}x3={w*h*3}")
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8-bit, color type 2 (RGB)
    raw = b"".join(b"\x00" + rgb[y*w*3:(y+1)*w*3] for y in range(h))
    idat = zlib.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat)
                + _chunk(b"IEND", b""))


def read_rgb_png(path: str) -> tuple:
    """→ (w, h, rgb bytes, top-down rows). Rejects non-RGB8/interlaced PNGs."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != _SIG:
        raise ValueError(f"{path}: not a PNG")
    pos, w, h, idat = 8, None, None, b""
    while pos + 8 <= len(data):
        (ln,) = struct.unpack(">I", data[pos:pos+4])
        tag = data[pos+4:pos+8]
        body = data[pos+8:pos+8+ln]
        pos += 12 + ln                               # length + tag + data + crc
        if tag == b"IHDR":
            w, h, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", body)
            if depth != 8 or ctype != 2 or inter != 0 or comp != 0 or filt != 0:
                raise ValueError(
                    f"{path}: only 8-bit RGB non-interlaced supported "
                    f"(depth={depth} ctype={ctype} interlace={inter})")
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if w is None:
        raise ValueError(f"{path}: no IHDR chunk")
    raw = zlib.decompress(idat)
    stride = w * 3
    if len(raw) != h * (stride + 1):
        raise ValueError(f"{path}: decompressed {len(raw)} bytes, "
                         f"expected {h*(stride+1)}")
    out = bytearray()
    prev = bytearray(stride)
    p = 0
    for _y in range(h):
        ftype = raw[p]
        line = bytearray(raw[p+1:p+1+stride])
        p += 1 + stride
        if ftype == 0:                               # None
            pass
        elif ftype == 1:                             # Sub
            for i in range(3, stride):
                line[i] = (line[i] + line[i-3]) & 0xFF
        elif ftype == 2:                             # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:                             # Average
            for i in range(stride):
                a = line[i-3] if i >= 3 else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:                             # Paeth
            for i in range(stride):
                a = line[i-3] if i >= 3 else 0
                b = prev[i]
                c = prev[i-3] if i >= 3 else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        else:
            raise ValueError(f"{path}: unknown scanline filter {ftype}")
        out += line
        prev = line
    return w, h, bytes(out)
