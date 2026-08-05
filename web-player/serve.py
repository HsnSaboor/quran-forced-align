#!/usr/bin/env python3
"""Dev server with HTTP Range support, for serving this repo locally.

Python's stock `http.server` does NOT support Range requests (it always
returns the whole file with 200 OK, never 206 Partial Content, and never
sends an Accept-Ranges header) -- this makes Chrome's <audio> element
report `seekable.length === 0` for any file served through it, so seeking
(currentTime = X) silently fails to actually move playback position even
though the file loads and plays from the start just fine. That's a dev-server
limitation, not a bug in web-player/index.html's own code: any real web
server (nginx, Caddy, a CDN, GitHub Pages, etc.) already supports Range
requests correctly, so this only matters for local development.

Usage (same as `python -m http.server`, run from the repo root):
    python3 web-player/serve.py [port]
"""
import http.server
import os
import re
import sys


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self._send_common_headers(path, file_size)
            self.end_headers()
            f = open(path, "rb")
            return f

        match = re.match(r"bytes=(\d*)-(\d*)\Z", range_header)
        if not match:
            self.send_response(416)
            self.end_headers()
            return None

        start_str, end_str = match.groups()
        if not start_str and end_str:
            # Suffix-length range (RFC 7233): "bytes=-N" means "the last N
            # bytes of the resource", not "starting at byte 0".
            suffix_len = int(end_str)
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        else:
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        length = end - start + 1
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self._send_common_headers(path, length)
        self.end_headers()

        f = open(path, "rb")
        f.seek(start)
        # Wrap so copyfile() (called by the base class after send_head
        # returns) only ever reads exactly `length` bytes, not the rest of
        # the file.
        return _LimitedReader(f, length)

    def _send_common_headers(self, path, content_length):
        ctype = self.guess_type(path)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(content_length))


class _LimitedReader:
    """Wraps a file object so reads never go past `limit` bytes total --
    needed because http.server's copyfile() just calls shutil.copyfileobj
    with no length argument, which would otherwise read to EOF instead of
    stopping at the requested range's end."""

    def __init__(self, f, limit):
        self._f = f
        self._remaining = limit

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._f.read(size)
        self._remaining -= len(data)
        return data

    def close(self):
        self._f.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    http.server.test(HandlerClass=RangeRequestHandler, port=port)
