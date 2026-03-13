#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        # Chrome private-network access checks for HTTPS -> localhost requests.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        # Avoid stale cached ndjson/grid responses between docs.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve WizMap files locally with CORS headers.")
    p.add_argument(
        "--dir",
        default="results/wizmap/grampian_wizmap_files",
        help="Directory containing data.ndjson and grid.json",
    )
    p.add_argument("--host", default="127.0.0.1", help="Host/IP to bind.")
    p.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    for required in ("data.ndjson", "grid.json"):
        if not (base_dir / required).exists():
            raise FileNotFoundError(f"Missing required file: {base_dir / required}")

    data_url = f"http://{args.host}:{args.port}/data.ndjson"
    grid_url = f"http://{args.host}:{args.port}/grid.json"
    wizmap_url = (
        "https://poloclub.github.io/wizmap/"
        f"?dataURL={quote(data_url, safe='')}&gridURL={quote(grid_url, safe='')}"
    )

    print(f"Serving directory: {base_dir}")
    print(f"Data URL: {data_url}")
    print(f"Grid URL: {grid_url}")
    print(f"Open this in browser:\n{wizmap_url}")

    handler = lambda *h_args, **h_kwargs: CORSRequestHandler(
        *h_args, directory=str(base_dir), **h_kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
