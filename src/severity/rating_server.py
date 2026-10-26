"""Serve a rating page locally and write every answer straight to a CSV.

    python rating_server.py      # then open http://localhost:8766

Generic over the answer fields: the page POSTs a list of row dicts and the CSV columns
are taken from the first row. Binds to 127.0.0.1 only, so images never leave this machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "severity_val_subset"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # atomic: a crash mid-write can't corrupt earlier answers


def make_handler(page: Path, results: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/answers":
                self._send(200, json.dumps(load_rows(results)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/save":
                self._send(404, b"not found", "text/plain")
                return
            save_rows(results, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send(200, b'{"ok": true}', "application/json")

        def log_message(self, *args) -> None:
            pass

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default=str(DEFAULT_DIR / "severity_rating.html"))
    parser.add_argument("--results", default=str(DEFAULT_DIR / "severity_ratings.csv"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    page = Path(args.page)
    if not page.exists():
        raise SystemExit(f"{page} not found -- run build_severity_rating.py first")
    print(f"[rating_server] open http://localhost:{args.port}  (answers save to {args.results})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(page, Path(args.results))).serve_forever()
