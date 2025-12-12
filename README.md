# Game of Thrones RAG CLI (LangChain + Chroma + Ollama)

Local Retrieval-Augmented Generation (RAG) over **A Game of Thrones** (plain text), with:

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (CPU)
- **Vector DB:** Chroma (persisted locally)
- **LLM:** `ChatOllama` (e.g. `llama3` or `mistral`) with a safe fallback if Ollama isn't available

## What you get

- **Semantic queries (RAG):** Ask questions like:
  - `List Ned Stark's children`
  - `Write a biography of Khal Drogo`
- **Analytical queries (heuristic):** Quick approximations like:
  - `Top 5 most mentioned characters`
  - `How many characters total?`
  - `Count mentions of Arya Stark`

Analytical queries do **not** go through RAG. They use a local regex/`Counter` heuristic because RAG retrieval cannot reliably scan the entire book for global counts.

## Project layout

- `ingest.py` — builds the persisted Chroma index in `./chroma_db`
- `main.py` — interactive CLI for asking questions
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

> Note: `sentence-transformers` will download the embedding model the first time you run ingestion (network required for that initial download).

## Ingest (build the vector store)

```bash
python ingest.py --input data/game_of_thrones.txt
```

This creates a persisted Chroma DB in `./chroma_db` and prints how many chunks were created.

## Run the CLI

### With Ollama (recommended)

Make sure Ollama is installed and running, then pull a model:

```bash
ollama pull llama3
```

Start the CLI:

```bash
python main.py --ollama-model llama3
```

### Without Ollama (dummy fallback)

If you don't have local inference set up yet, you can still test retrieval:

```bash
python main.py --use-dummy-llm
```

This uses a simple extractive fallback over retrieved chunks (not a true generative answer).

## Query behavior

### RAG prompt discipline

The system prompt forces:

> “You are a Maester of the Citadel. Answer strictly based on the provided context. If the answer is not in the context, say 'I do not know'.”

### Biography queries fetch more context

Queries matching:

- `Write a biography of ...`
- `Write a biography for ...`

automatically retrieve **more chunks** (`k=7` by default) so the model has enough context for a coherent biography.

Tune via flags:

- `--k 4` (default for normal questions)
- `--bio-k 7` (default for biography questions)

## Analytics mode (heuristic)

Examples:

- `Top 5 most mentioned characters`
- `How many characters total?`
- `Count mentions of Jon Snow`

These are approximations based on regex-matched “name-like” spans and can be inaccurate (false positives/negatives).

### Higher-quality alternative (spaCy NER)

There is an optional script:

- `count_names_spacy.py`

It uses spaCy NER to count `PERSON` entities. To use it:

```bash
python -m pip install spacy
python -m spacy download en_core_web_sm
python count_names_spacy.py
```

## Troubleshooting

- **`Chroma persistence directory not found`**
  - Run ingestion first: `python ingest.py --input data/game_of_thrones.txt`
- **Ollama errors / connection refused**
  - Start Ollama and verify: `ollama list`
  - Or run: `python main.py --use-dummy-llm`
- **Dependency import errors**
  - Reinstall: `python -m pip install -r requirements.txt`

