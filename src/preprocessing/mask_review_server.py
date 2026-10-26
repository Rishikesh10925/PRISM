"""Serve the mask review page locally and save every rating straight to disk.

    python mask_review_server.py      # then open http://localhost:8765

Each rating is written to data/merged/review_sample/mask_review_results.csv as soon as
it's made, so the reviewer never has to export or save a file. Binds to 127.0.0.1 only:
the review images never leave this machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parents[2] / "data" / "merged" / "review_sample"
PAGE = REVIEW_DIR / "mask_review.html"
RESULTS = REVIEW_DIR / "mask_review_results.csv"
FIELDS = ["review_id", "image", "rating", "missed_pothole", "false_mask", "artifact", "notes"]


def load_answers(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["review_id"]: {
                "rating": row["rating"],
                "missed": row["missed_pothole"] == "true",
                "false_mask": row["false_mask"] == "true",
                "artifact": row["artifact"] == "true",
                "notes": row["notes"],
            }
            for row in csv.DictReader(f)
        }


def save_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(row.get(k, "")).lower() if isinstance(row.get(k), bool) else row.get(k, "") for k in FIELDS})
    tmp.replace(path)  # atomic, so a crash mid-write can't corrupt earlier ratings


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/answers":
            self._send(200, json.dumps(load_answers(RESULTS)).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, b"not found", "text/plain")
            return
        rows = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        save_rows(RESULTS, rows)
        self._send(200, b'{"ok": true}', "application/json")

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not PAGE.exists():
        raise SystemExit(f"{PAGE} not found -- run build_mask_review.py first")
    print(f"[mask_review_server] open http://localhost:{args.port}  (ratings save to {RESULTS})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
