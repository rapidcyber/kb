# Offline AI Assistant with a Local RAG Knowledge Base

A private, fully offline question-answering assistant that runs on an ordinary Windows PC with no graphics card. It answers questions from your own documents (notes, PDFs, Word files, code) using a local language model, cites where each answer came from, and checks its own answers against the sources. It can also plug into VS Code through the Continue extension, and it can optionally search the web when you ask it to.

Everything runs locally: [Ollama](https://ollama.com) serves the AI models, and [PostgreSQL](https://www.postgresql.org) with the [pgvector](https://github.com/pgvector/pgvector) extension stores the knowledge base. Your documents never leave your computer.

## How it works

`ingest.py` reads every supported file in your documents folder, splits it into overlapping chunks of about 1,000 characters, turns each chunk into a 768-number embedding with `nomic-embed-text`, and stores the chunks in PostgreSQL. It is safe to re-run: unchanged files are skipped, edited files are replaced, and deleted files are removed. PDFs are read with PyMuPDF, which keeps paragraphs together and strips repeated page headers and footers.

`ask.py` answers questions in the terminal. For each question it runs a hybrid search: a meaning search over the embeddings, plus a keyword search in which each word is weighted by how rare it is across the knowledge base, so specific terms like `PGPASSWORD` or an error message count more than common words. The two result lists are merged with Reciprocal Rank Fusion, each match is joined with the chunk that follows it so definitions cut at a chunk boundary come through complete, and the passages are given to the chat model (`qwen2.5-coder:7b` by default) with instructions to cite them.

`kb_server.py` is a small local web service that exposes the same search to Continue in VS Code, so you can type `@Knowledge Base` in the chat.

## Safeguards against common AI mistakes

Small local models are fluent but fallible, so the assistant includes checks that do not depend on the model's own judgment.

| Safeguard | What it does |
|---|---|
| Source list | Shows only the passages the answer actually cites, with each file's date. If an answer cites nothing, it says so and tells you whether any passages were found |
| Spot-check | Extracts the answer's specific details (distinctive numbers, `code`, command lines, names) and reports which source file each one appears in, and which appear in none |
| Neutral rephrasing | Questions containing an assumption ("I'm pretty sure...", "right?") are rewritten as neutral questions first, so the model answers from the sources instead of agreeing with you |
| Question splitting | A request with several numbered items or questions is answered one question at a time, each with its own search |
| "Latest version" notice | Reminds you that an offline assistant only knows what you saved, and suggests web search |
| `/show` | Lets you read the exact passages the model was given |

These reduce mistakes but do not remove them. Treat uncited answers and anything listed as "NOT found" as unverified.

## Requirements

The project was built and tested on Windows with an AMD Ryzen 5 5600G, 16 GB of RAM and no graphics card, using Python 3.14, PostgreSQL 18.6 with pgvector 0.8.7, and Ollama. On that machine `qwen2.5-coder:7b` generates about 5 tokens per second and `qwen2.5-coder:3b` about 12.7.

Install Ollama and pull the models:

```
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

`qwen2.5-coder:3b` (chat) and `qwen2.5-coder:1.5b` (autocomplete) are useful additions if you also use Continue in VS Code.

Install PostgreSQL and the pgvector extension. On Windows, pgvector is built from source with `nmake /F Makefile.win` in the "x64 Native Tools Command Prompt for VS"; see the [pgvector installation notes](https://github.com/pgvector/pgvector#windows).

## Setup

Clone the repository and create the Python environment:

```
git clone https://github.com/rapidcyber/kb.git C:\kb
cd C:\kb
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Store your PostgreSQL password in an environment variable so it never appears in code, then close and reopen your terminal:

```
setx PGPASSWORD "your_password"
```

Create the database. In `psql`, run:

```sql
CREATE DATABASE knowledge;
\c knowledge
CREATE EXTENSION vector;

CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
```

Check the connection with `.venv\Scripts\python test_db.py`. It should print "Database: connected" and "Embedding size: 768".

Set `DOCS_DIR` at the top of `ingest.py` to your documents folder (the default is `D:\kb_docs`), put some files in it, and load them:

```
.venv\Scripts\python ingest.py
```

Supported file types are `.txt`, `.md`, `.pdf`, `.docx` and common code and data files (`.py`, `.js`, `.sql`, `.json`, `.yaml`, `.csv` and others). Scanned PDFs contain images rather than text and are skipped, since there is no OCR.

## Usage

```
.venv\Scripts\python ask.py
```

Type a question at the `You:` prompt. The assistant remembers the last three questions and answers, so follow-up questions work.

| Command | What it does |
|---|---|
| `/new` | Start a new conversation (clears memory). Recommended before any important question |
| `/show` | List the passages given to the model for the last answer; `/show 2` prints passage 2 in full |
| `/web <question>` | Search the web with DuckDuckGo instead of your documents. Needs internet; only the question is sent |
| `exit` | Quit |

Ask neutral questions ("What is the difference between X and Y?") rather than asking it to confirm something, and ask for one specific fact per question when exact details matter.

## Using the knowledge base in VS Code (Continue)

Start the server:

```
.venv\Scripts\python kb_server.py
```

It listens only on `127.0.0.1:8765`, so it is not reachable from other computers. Then add it to Continue's `config.yaml` as an `http` context provider. Note the address uses `http`, not `https`:

```yaml
context:
  - provider: http
    params:
      url: "http://127.0.0.1:8765/retrieve"
      title: kb
      displayTitle: Knowledge Base
      description: Search my offline knowledge base
```

Listing a `context:` section replaces Continue's default providers, so also list the ones you use, such as `code`, `file`, `currentFile`, `diff`, `terminal`, `problems`, `folder` and `codebase`. In Continue's Chat mode, type `@`, choose Knowledge Base, and ask your question.

To start the server automatically in the background when you log in, copy `kb_server_start.bat` into your Startup folder (Windows key + R, then `shell:startup`). It runs the server with `pythonw`, so no window appears. Messages go to `kb_server.log`, and the server waits for PostgreSQL to finish starting before it connects.

## Configuration

The main settings are at the top of `ask.py`.

| Setting | Default | Meaning |
|---|---|---|
| `CHAT_MODEL` | `qwen2.5-coder:7b` | Model that writes the answers. `qwen2.5-coder:3b` is about 2.5 times faster but reasons less reliably |
| `TOP_K` | 4 | Matches taken from each search |
| `MIN_RELEVANCE` | 0.62 | Meaning matches below this are ignored |
| `HYBRID` | True | Also run the keyword search (needs the `tsv` column) |
| `KEYWORD_MIN_SCORE` | 2.0 | Keyword matches need at least one fairly rare word |
| `EXPAND_NEXT` | True | Add the chunk that follows each match |
| `ANSWER_WITHOUT_SOURCES` | True | When nothing relevant is found, answer from the model's general knowledge (clearly labeled) instead of replying "not covered" |
| `MAX_TURNS` | 3 | Question/answer pairs remembered |
| `TEMPERATURE` | 0.1 | Low values make answers more literal |
| `NEUTRALIZE` | True | Rephrase questions that contain assumptions |
| `ABOUT_ME` | (profile) | A short description of your setup that is always given to the model, so advice fits your situation. Edit it to describe yours |

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Loads documents into the knowledge base |
| `ask.py` | Terminal assistant |
| `kb_server.py` | Local search service for Continue |
| `kb_server_start.bat` | Starts the server in the background |
| `test_db.py` | Checks the database and embedding model |
| `test_server.py` | Sends a test request to the running server |
| `compare_pdf.py` | Compares pypdf and PyMuPDF text extraction on one PDF page |

## Backup

Back up the knowledge base with `pg_dump`:

```
& "C:\Program Files\PostgreSQL\18\bin\pg_dump.exe" -U postgres -h localhost -F c -f D:\kb_backup\knowledge.dump knowledge
```

Restore it with `pg_restore -U postgres -d knowledge --clean D:\kb_backup\knowledge.dump`. Your original documents can always rebuild it with `ingest.py`.

## Limitations

The models are small enough to run on a CPU, and that has costs. They can agree with a wrong premise, state errors confidently, and give different answers to the same question on different runs. The safeguards above make these problems visible, but the final check is yours: read the cited passage when an answer matters. Answers that use many retrieved passages can take a minute or more to start on a CPU, because the model reads at roughly 50 tokens per second.

Web search reads DuckDuckGo's HTML results page. It is unofficial and may stop working if the page changes or requests are limited. Web pages are untrusted content, and the model is instructed to treat passages as information rather than instructions.

## Licenses of dependencies

PyMuPDF, used to read PDFs, is licensed under the AGPL. That does not matter for personal use, but if you distribute this project, check that your license is compatible, or replace PyMuPDF with the `pypdf` reader (the original reader, which produces less clean text).
