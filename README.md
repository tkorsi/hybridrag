# Game of Thrones RAG (Streamlit + LangChain + Chroma + Groq)

Local Retrieval-Augmented Generation (RAG) over **A Game of Thrones** (plain text), with:

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (CPU)
- **Vector DB:** Chroma (persisted locally)
- **LLM:** Groq via `ChatGroq` (`llama-3.1-8b-instant`)
- **UI:** Streamlit chat app

## What you get

- **Semantic queries (RAG):** Ask questions like:
  - `List Ned Stark's children`
  - `Write a biography of Khal Drogo`
- **Analytical queries (spaCy when available):** Quick approximations like:
  - `Top 5 most mentioned characters`
  - `How many characters total?`
  - `Count mentions of Arya Stark`

Analytical queries do **not** go through RAG. If spaCy is available, the app counts `PERSON` entities; otherwise it falls back to a lightweight regex/`Counter` heuristic.

## Project layout

- `app.py` — Streamlit chat app
- `ingest.py` — optional: build the persisted Chroma index in `./chroma_db`
- `data/game_of_thrones.txt` — **you provide this file locally** (ignored by git)
- `chroma_db/` — generated vector store (ignored by git)

## Setup

### 1) Put the book text in place

Create/confirm the file:

- `data/game_of_thrones.txt`

The full text is **not** committed to this repo (by design).

### 2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

> Note: `sentence-transformers` will download the embedding model the first time you build the index (network required for that initial download).

## Configure Groq (required for semantic RAG answers)

### Streamlit Cloud

Set a secret named:

- `GROQ_API_KEY`

### Local development

Create `.streamlit/secrets.toml` (not committed):

```toml
GROQ_API_KEY="YOUR_KEY_HERE"
```

## Run the app

```bash
python preload.py
streamlit run app.py
```

On first run, the app will build a Chroma index in `./chroma_db` if it does not exist yet.

## Deploy to Streamlit Cloud

1. Deploy this repo as a Streamlit app (main file: `app.py`).
2. Add `GROQ_API_KEY` in the app’s Secrets.
3. Ensure the book text is available at `data/game_of_thrones.txt` (the repo ignores `data/*.txt` by default).
4. Run `python preload.py` before starting the app so `en_core_web_sm` is available for analytics.

The first cold start may take longer because the embedding model and the vector index are built/cached.

## Optional: Ingest (pre-build the vector store)

```bash
python ingest.py --input data/game_of_thrones.txt
```

This creates a persisted Chroma DB in `./chroma_db` and prints how many chunks were created.

## Query behavior

### RAG prompt discipline

The system prompt forces:

> “You are a Maester of the Citadel. Answer strictly based on the provided context. If the answer is not in the context, say 'I do not know'.”

### Biography queries fetch more context

Queries matching:

- `Write a biography of ...`
- `Write a biography for ...`

automatically retrieve **more chunks** (`k=7` by default) so the model has enough context for a coherent biography.

Tune in the Streamlit sidebar:

- `k (semantic)` (default 4)
- `k (biography)` (default 7)

## Analytics mode (spaCy PERSON NER)

Examples:

- `Top 5 most mentioned characters`
- `How many characters total?`
- `Count mentions of Jon Snow`

If `spacy` + `en_core_web_sm` are available, the app extracts and counts only `PERSON` entities.
If spaCy isn't available, the app falls back to a lightweight regex heuristic.
If spaCy is installed but the model is missing, run `python preload.py` once to download `en_core_web_sm` before starting the app.

### spaCy setup

Install spaCy + the English model:

```bash
python -m pip install spacy
python preload.py
```

There is also an optional script:

- `count_names_spacy.py`

It uses spaCy NER to count `PERSON` entities. To use it:

```bash
python -m pip install spacy
python preload.py
python count_names_spacy.py
```

## Troubleshooting

- **Missing `GROQ_API_KEY`**
  - Set it in Streamlit Cloud secrets, or create `.streamlit/secrets.toml` locally.
- **ChromaDB / SQLite errors on Streamlit Cloud**
  - This repo includes `pysqlite3-binary` and `app.py` prefers it at runtime to avoid older system SQLite builds.
- **Chroma index issues**
  - Delete `./chroma_db` and reload to rebuild, or run `python ingest.py --input data/game_of_thrones.txt`.
- **Dependency import errors**
  - Reinstall: `python -m pip install -r requirements.txt`
