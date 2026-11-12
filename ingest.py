"""
ingest.py - Load documents into the offline knowledge base.

Reads files from DOCS_DIR, splits them into chunks, embeds each chunk with
nomic-embed-text (via Ollama), and stores them in PostgreSQL (pgvector).

Safe to re-run: unchanged files are skipped, changed files are replaced,
and files deleted from the folder are removed from the database.

PDFs are read with PyMuPDF (pip install pymupdf).

Run from C:\\kb with:  .venv\\Scripts\\python ingest.py
"""
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ollama
import psycopg
from pgvector.psycopg import register_vector

# ---------------- Settings ----------------
DOCS_DIR = Path(r"D:\kb_docs")   # folder with your documents (on the HDD)
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000                # max characters per chunk
CHUNK_OVERLAP = 150              # characters repeated between neighbouring chunks
BATCH_SIZE = 16                  # chunks embedded per request

TEXT_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".sql", ".json", ".yaml", ".yml",
    ".html", ".css", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs",
    ".php", ".rb", ".sh", ".bat", ".ps1", ".ini", ".toml", ".csv", ".xml",
}
SUPPORTED = TEXT_EXTS | {".pdf", ".docx"}


def read_file(path: Path) -> str:
    """Return the plain text of a supported file."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        import docx
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_block(text: str) -> str:
    """Join the lines of one PDF text block into a single clean line."""
    text = re.sub(r"-\n(?=[a-z])", "", text)        # rejoin hyphenated words: rep-\nresents
    text = re.sub(r"\s*\n\s*", " ", text)          # line breaks inside a paragraph
    return re.sub(r"[ \t]+", " ", text).strip()


def _edge_key(text: str) -> str:
    """Normalise a header/footer line so 'Chapter 15 ... 506' matches on every page."""
    return re.sub(r"[\d\s.]+", " ", text).strip().lower()


def _merge_fragments(blocks: list[str]) -> list[str]:
    """Merge runs of tiny blocks (figure labels, table cells) into one line."""
    merged, run = [], []
    for b in blocks:
        if len(b.split()) <= 3:
            run.append(b)
            continue
        if run:
            merged.append(" ".join(run))
            run = []
        merged.append(b)
    if run:
        merged.append(" ".join(run))
    return merged


def read_pdf(path: Path) -> str:
    """Extract PDF text with PyMuPDF: paragraphs kept, headers/footers removed."""
    import pymupdf
    from collections import Counter

    doc = pymupdf.open(str(path))
    pages = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        texts = [_clean_block(b[4]) for b in blocks if b[6] == 0]   # 0 = text block
        pages.append([t for t in texts if t])
    doc.close()

    # Lines that appear at the top/bottom of many pages are headers or footers
    edge_counts = Counter()
    for texts in pages:
        edge_idx = set(range(min(2, len(texts)))) | set(range(max(0, len(texts) - 2), len(texts)))
        for i in edge_idx:
            edge_counts[_edge_key(texts[i])] += 1
    repeated = {k for k, n in edge_counts.items() if n >= 5} | {""}

    paragraphs = []
    for texts in pages:
        n = len(texts)
        kept = [t for i, t in enumerate(texts)
                if not ((i < 2 or i >= n - 2) and _edge_key(t) in repeated)]
        paragraphs.extend(_merge_fragments(kept))
    return "\n\n".join(paragraphs)


def chunk_text(text: str) -> list[str]:
    """Split text into chunks of about CHUNK_SIZE characters.
    Keeps paragraphs together and never cuts in the middle of a word."""
    text = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        # A single paragraph longer than CHUNK_SIZE is cut at word boundaries
        while len(p) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            cut = p.rfind(" ", 0, CHUNK_SIZE)
            if cut < CHUNK_SIZE // 2:
                cut = CHUNK_SIZE
            chunks.append(p[:cut].strip())
            start = p.find(" ", cut - CHUNK_OVERLAP)
            p = p[start + 1:] if 0 <= start < cut else p[cut:]
        if len(current) + len(p) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            tail = current[-CHUNK_OVERLAP:] if current else ""
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]     # start the overlap on a whole word
            current = f"{tail}\n\n{p}" if tail else p
    if current:
        chunks.append(current)
    return chunks


def embed(texts: list[str]) -> list[np.ndarray]:
    """Embed texts with nomic-embed-text. Documents use the 'search_document: ' prefix."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = ["search_document: " + t for t in texts[i:i + BATCH_SIZE]]
        result = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors.extend(np.array(v, dtype=np.float32) for v in result["embeddings"])
    return vectors


def main():
    if not DOCS_DIR.is_dir():
        sys.exit(f"Folder not found: {DOCS_DIR}")

    files = sorted(p for p in DOCS_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED)
    print(f"Found {len(files)} supported file(s) in {DOCS_DIR}\n")

    # Password comes from the PGPASSWORD environment variable
    conn = psycopg.connect(dbname="knowledge", user="postgres",
                           host="localhost", autocommit=True)
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT source, max(created_at) FROM chunks GROUP BY source")
        stored = dict(cur.fetchall())

    added = skipped = problems = 0
    start = time.time()

    for path in files:
        source = str(path)
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if source in stored and stored[source] >= modified:
            print(f"  skip     {path.name} (unchanged)")
            skipped += 1
            continue
        try:
            chunks = chunk_text(read_file(path))
            if not chunks:
                print(f"  empty    {path.name} (no text found - scanned PDF?)")
                problems += 1
                continue
            t0 = time.time()
            vectors = embed(chunks)
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
                    cur.executemany(
                        "INSERT INTO chunks (source, chunk_index, content, embedding) "
                        "VALUES (%s, %s, %s, %s)",
                        [(source, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
                    )
            print(f"  added    {path.name}: {len(chunks)} chunk(s) in {time.time() - t0:.1f}s")
            added += 1
        except Exception as e:
            print(f"  error    {path.name}: {e}")
            problems += 1

    # Remove chunks of files that were deleted from the folder
    on_disk = {str(p) for p in files}
    prefix = str(DOCS_DIR) + os.sep
    removed = [s for s in stored if s.startswith(prefix) and s not in on_disk]
    with conn.cursor() as cur:
        for s in removed:
            cur.execute("DELETE FROM chunks WHERE source = %s", (s,))
            print(f"  removed  {Path(s).name} (no longer in folder)")
        cur.execute("SELECT count(*) FROM chunks")
        total = cur.fetchone()[0]
    conn.close()

    print(f"\nDone in {time.time() - start:.1f}s. Added {added}, skipped {skipped}, "
          f"removed {len(removed)}, problems {problems}.")
    print(f"Total chunks in knowledge base: {total}")


if __name__ == "__main__":
    main()