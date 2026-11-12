"""
kb_server.py - Lets Continue (VS Code) search your offline knowledge base.

Continue's "http" context provider sends a POST request with the question to
http://127.0.0.1:8765/retrieve. This server finds the most relevant chunks in
PostgreSQL (using the same retrieval as ask.py) and returns them as context items.

Only listens on 127.0.0.1, so it is not reachable from other computers.

Run from C:\\kb with:  .venv\\Scripts\\python kb_server.py
In the background (no window):  .venv\\Scripts\\pythonw kb_server.py
Messages are also written to kb_server.log.
"""
import json
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from ask import retrieve   # same search, cutoff and passage merging as ask.py

HOST = "127.0.0.1"
PORT = 8765

LOG_FILE = Path(__file__).with_name("kb_server.log")


def log(message: str):
    """Print to the terminal (if any) and append to kb_server.log."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    if sys.stdout:            # pythonw (background mode) has no terminal
        print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect(retries: int = 24, wait: int = 5):
    """Connect to PostgreSQL, retrying while it starts up after login."""
    for attempt in range(1, retries + 1):
        try:
            # Password comes from the PGPASSWORD environment variable
            c = psycopg.connect(dbname="knowledge", user="postgres",
                                host="localhost", autocommit=True)
            register_vector(c)
            return c
        except psycopg.OperationalError as e:
            log(f"Database not ready (attempt {attempt}/{retries}): {str(e).strip()[:100]}")
            time.sleep(wait)
    log("Could not connect to the database. Server not started.")
    sys.exit(1)


conn = None   # set in main


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/retrieve":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}

        # Continue sends the question as "query"; fall back to the full chat input
        text = (body.get("query") or "").strip() or (body.get("fullInput") or "").strip()

        items = []
        if text:
            for n, p in enumerate(retrieve(conn, text), start=1):
                name = Path(p["source"]).name
                items.append({
                    "name": f"[{n}] {name}",
                    "description": f"part {p['parts']}, relevance {p['relevance']:.2f}",
                    "content": f"[{n}] From {name}:\n{p['content']}",
                })
        if not items:
            items = [{
                "name": "Knowledge base",
                "description": "no relevant passages",
                "content": "The knowledge base has no passages relevant to this question.",
            }]

        data = json.dumps(items).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        log(f"Sent {len(items)} passage(s) for: {text[:70]!r}")

    def log_message(self, *args):
        pass   # keep the terminal quiet; we print our own one-line log


if __name__ == "__main__":
    conn = connect()
    try:
        server = HTTPServer((HOST, PORT), Handler)
    except OSError:
        log(f"Port {PORT} is already in use - the server is probably already running.")
        sys.exit(0)
    log(f"Knowledge base server running at http://{HOST}:{PORT}/retrieve")
    if sys.stdout:
        print("Leave this window open while using @Knowledge Base in Continue. Ctrl+C stops it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        log("Server stopped.")
