"""Ingest a local text file into a persisted Chroma vector store.

Usage:
  python ingest.py --input data/game_of_thrones.txt

This script:
  1) Loads the input .txt file
  2) Splits it into overlapping chunks (1000 chars, 200 overlap)
  3) Embeds the chunks with HuggingFaceEmbeddings on CPU
  4) Persists the vectors to ./chroma_db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _import_chroma() -> Any:
    try:
        from langchain_chroma import Chroma  # type: ignore

        return Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma  # type: ignore

        return Chroma


def _import_embeddings() -> Any:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings


def build_embeddings(model_name: str) -> Any:
    """Create CPU-only HuggingFace embeddings."""
    HuggingFaceEmbeddings = _import_embeddings()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(*, chunks: list[Any], embeddings: Any, persist_dir: Path, collection: str) -> Any:
    """Create and persist a Chroma vector store from documents."""
    Chroma = _import_chroma()

    persist_dir.mkdir(parents=True, exist_ok=True)

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            collection_name=collection,
        )
    except TypeError:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_name=collection,
        )

    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        persist()

    return vectorstore


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Game of Thrones text into ChromaDB.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/game_of_thrones.txt"),
        help="Path to the input .txt file (default: data/game_of_thrones.txt).",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Directory to persist ChromaDB (default: ./chroma_db).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="game_of_thrones",
        help="Chroma collection name (default: game_of_thrones).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name (default: sentence-transformers/all-MiniLM-L6-v2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(
            f"ERROR: Input file not found: {args.input}\n"
            "Place your book text at that path, or pass --input /path/to/book.txt",
            file=sys.stderr,
        )
        return 2

    try:
        text = args.input.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"ERROR: Failed to read {args.input}: {exc}", file=sys.stderr)
        return 2

    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    doc = Document(page_content=text, metadata={"source": str(args.input)})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents([doc])

    if not chunks:
        print("ERROR: No chunks were produced from the input text.", file=sys.stderr)
        return 2

    try:
        embeddings = build_embeddings(args.embedding_model)
    except Exception as exc:
        print(f"ERROR: Failed to initialize embeddings: {exc}", file=sys.stderr)
        return 2

    try:
        build_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
            persist_dir=args.persist_dir,
            collection=args.collection,
        )
    except Exception as exc:
        print(f"ERROR: Failed to build/persist ChromaDB: {exc}", file=sys.stderr)
        return 2

    print("Ingestion complete.")
    print(f"- Input:        {args.input}")
    print(f"- Persist dir:  {args.persist_dir}")
    print(f"- Collection:   {args.collection}")
    print(f"- Chunks:       {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
