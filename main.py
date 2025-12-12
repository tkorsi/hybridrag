"""Interactive CLI for querying Game of Thrones with RAG (LangChain + Chroma).

Workflow:
  1) Run ingestion once:
       python ingest.py --input data/game_of_thrones.txt
  2) Start the chat CLI:
       python main.py

Notes on query types:
  - Semantic queries (facts/biographies) go through RAG + LLM.
  - Analytical queries (counts/top mentions) are approximated locally with regex/Counter
    because RAG retrieval cannot reliably scan the entire book at once.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a Maester of the Citadel. "
    "Answer strictly based on the provided context. "
    "If the answer is not in the context, say 'I do not know'."
)

_PROMPT: Any | None = None


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


def _import_chat_ollama() -> Any:
    try:
        from langchain_ollama import ChatOllama  # type: ignore

        return ChatOllama
    except Exception:
        from langchain_community.chat_models import ChatOllama  # type: ignore

        return ChatOllama


def build_embeddings(model_name: str) -> Any:
    HuggingFaceEmbeddings = _import_embeddings()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def load_vectorstore(*, persist_dir: Path, collection: str, embeddings: Any) -> Any:
    Chroma = _import_chroma()

    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        raise FileNotFoundError(
            f"Chroma persistence directory not found or empty: {persist_dir} "
            "(run ingest.py first)."
        )

    try:
        return Chroma(
            collection_name=collection,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    except TypeError:
        return Chroma(
            collection_name=collection,
            persist_directory=str(persist_dir),
            embedding=embeddings,
        )


def is_biography_query(question: str) -> bool:
    return bool(re.search(r"\bwrite\s+(?:a\s+)?biography\s+(?:of|for)\b", question, re.IGNORECASE))


def is_analytics_query(question: str) -> bool:
    q = question.lower()
    if "top " in q and "mentioned" in q:
        return True
    if "most mentioned" in q:
        return True
    if "how many characters" in q:
        return True
    if q.startswith("count mentions of ") or q.startswith("mentions of "):
        return True
    return False


def parse_top_n(question: str, default: int = 5) -> int:
    m = re.search(r"\btop\s+(\d+)\b", question, re.IGNORECASE)
    if not m:
        return default
    try:
        n = int(m.group(1))
    except ValueError:
        return default
    return max(1, min(n, 50))


NAME_REGEX = re.compile(
    r"\b[A-Z][a-z]+(?:['-][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:['-][A-Z][a-z]+)?){0,2}\b"
)

STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "By",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "I",
    "In",
    "Is",
    "It",
    "Its",
    "My",
    "No",
    "Not",
    "Of",
    "On",
    "Or",
    "She",
    "So",
    "The",
    "Their",
    "There",
    "They",
    "This",
    "To",
    "Was",
    "We",
    "Were",
    "What",
    "When",
    "Where",
    "Who",
    "With",
    "You",
}

TITLES = {"Ser", "Lord", "Lady", "King", "Queen", "Prince", "Princess", "Maester", "Septa", "Septon"}


def normalize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


@dataclass
class NameFrequencyAnalyzer:
    """Approximate name frequencies via regex matching.

    This is intentionally heuristic and will include some false positives/negatives.
    For higher quality counts, consider spaCy NER (see existing count_names_spacy.py).
    """

    book_path: Path
    _counts: Counter[str] | None = None

    def _load_text(self) -> str:
        return self.book_path.read_text(encoding="utf-8", errors="replace")

    def counts(self) -> Counter[str]:
        if self._counts is not None:
            return self._counts

        text = self._load_text()
        counts: Counter[str] = Counter()
        for match in NAME_REGEX.finditer(text):
            candidate = normalize_name(match.group(0))
            if candidate in STOPWORDS:
                continue

            first_token = candidate.split(" ", 1)[0]
            if first_token in STOPWORDS:
                continue

            if candidate in TITLES:
                continue

            counts[candidate] += 1

        self._counts = counts
        return counts

    def top(self, n: int) -> list[tuple[str, int]]:
        return self.counts().most_common(n)

    def unique_estimate(self, min_mentions: int = 5) -> int:
        return sum(1 for _, c in self.counts().items() if c >= min_mentions)

    def mentions_of(self, name: str) -> int:
        name = normalize_name(name)
        counts = self.counts()
        direct = counts.get(name, 0)
        if direct:
            return direct
        # Fallback: case-insensitive substring count (very rough).
        text = self._load_text()
        return len(re.findall(re.escape(name), text, flags=re.IGNORECASE))


def extractive_fallback_answer(*, context: str, question: str, max_sentences: int = 3) -> str:
    """Fallback when no local LLM is available: return top-matching sentences from context."""
    if not context.strip():
        return "I do not know"

    question_terms = {
        t
        for t in re.findall(r"[a-zA-Z']{2,}", question.lower())
        if t not in {"the", "and", "for", "with", "from", "that", "this", "what", "who", "how", "many"}
    }
    if not question_terms:
        return "I do not know"

    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    scored: list[tuple[int, str]] = []
    for s in sentences:
        s_terms = set(re.findall(r"[a-zA-Z']{2,}", s.lower()))
        score = len(question_terms & s_terms)
        if score > 0:
            scored.append((score, s))

    if not scored:
        return "I do not know"

    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(s for _, s in scored[:max_sentences]).strip()


def format_docs(docs: Iterable[Any]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        text = getattr(d, "page_content", "")
        if not text:
            continue
        parts.append(f"[Chunk {i}]\n{text}")
    return "\n\n".join(parts)


def get_prompt() -> Any:
    """Return a cached PromptTemplate for the RAG instruction."""
    global _PROMPT
    if _PROMPT is not None:
        return _PROMPT

    from langchain_core.prompts import PromptTemplate

    _PROMPT = PromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\n" + "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    return _PROMPT


def answer_with_llm(*, llm: Any, context: str, question: str) -> str:
    prompt = get_prompt()
    prompt_text = prompt.format(context=context, question=question)
    result = llm.invoke(prompt_text)
    return getattr(result, "content", str(result)).strip()


def retrieve_docs(retriever: Any, question: str) -> list[Any]:
    """Compatibility wrapper around retriever invocation across LangChain versions."""
    invoke = getattr(retriever, "invoke", None)
    if callable(invoke):
        return invoke(question)
    get_relevant_documents = getattr(retriever, "get_relevant_documents", None)
    if callable(get_relevant_documents):
        return get_relevant_documents(question)
    raise TypeError("Unsupported retriever interface (expected .invoke or .get_relevant_documents)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG CLI for Game of Thrones (LangChain + Chroma).")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Chroma persistence directory (default: ./chroma_db).",
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
        help="HuggingFace embedding model name.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3",
        help="Local Ollama model to use (default: llama3).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/game_of_thrones.txt"),
        help="Book text path used for analytics mode (default: data/game_of_thrones.txt).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-k chunks for normal questions (default: 4).",
    )
    parser.add_argument(
        "--bio-k",
        type=int,
        default=7,
        help="Top-k chunks for biography questions (default: 7).",
    )
    parser.add_argument(
        "--use-dummy-llm",
        action="store_true",
        help="Skip Ollama and use an extractive fallback answerer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        embeddings = build_embeddings(args.embedding_model)
    except Exception as exc:
        print(f"ERROR: Failed to initialize embeddings: {exc}", file=sys.stderr)
        return 2

    try:
        vectorstore = load_vectorstore(
            persist_dir=args.persist_dir, collection=args.collection, embeddings=embeddings
        )
    except Exception as exc:
        print(f"ERROR: Failed to load ChromaDB: {exc}", file=sys.stderr)
        return 2

    default_retriever: Any | None = None
    bio_retriever: Any | None = None
    as_retriever = getattr(vectorstore, "as_retriever", None)
    if callable(as_retriever):
        try:
            default_retriever = as_retriever(search_kwargs={"k": args.k})
            bio_retriever = as_retriever(search_kwargs={"k": args.bio_k})
        except Exception:
            default_retriever = None
            bio_retriever = None

    llm: Any | None = None
    if not args.use_dummy_llm:
        try:
            ChatOllama = _import_chat_ollama()
            llm = ChatOllama(model=args.ollama_model, temperature=0)
        except Exception as exc:
            print(f"WARNING: ChatOllama unavailable ({exc}); using extractive fallback.", file=sys.stderr)
            llm = None

    analyzer: NameFrequencyAnalyzer | None = None

    print("Game of Thrones RAG CLI")
    print("Type your question, or 'quit'/'exit' to leave.")
    print("Tip: analytics queries like 'Top 5 most mentioned characters' use a local heuristic.")
    print()

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit"}:
            break

        if is_analytics_query(question):
            if not args.data_path.exists():
                print(
                    f"Analytics mode requires the raw text file at {args.data_path} "
                    "(or pass --data-path).",
                    file=sys.stderr,
                )
                continue

            analyzer = analyzer or NameFrequencyAnalyzer(args.data_path)

            if question.lower().startswith(("count mentions of ", "mentions of ")):
                name = re.sub(r"^(count mentions of|mentions of)\s+", "", question, flags=re.IGNORECASE).strip()
                if not name:
                    print("Provide a name, e.g. 'Count mentions of Arya Stark'.")
                    continue
                print(
                    f"Approximate mentions of '{name}': {analyzer.mentions_of(name)} "
                    "(heuristic; may be inaccurate)"
                )
                continue

            if "how many characters" in question.lower():
                est = analyzer.unique_estimate(min_mentions=5)
                print(
                    "Approximate unique character-like names (>=5 mentions): "
                    f"{est} (heuristic; may be inaccurate)"
                )
                continue

            n = parse_top_n(question, default=5)
            top = analyzer.top(n)
            print(f"Top {n} most-mentioned character-like names (heuristic):")
            for name, count in top:
                print(f"- {name}: {count}")
            continue

        bio = is_biography_query(question)
        k = args.bio_k if bio else args.k
        try:
            retriever = bio_retriever if bio else default_retriever
            if retriever is None:
                docs = vectorstore.similarity_search(question, k=k)
            else:
                docs = retrieve_docs(retriever, question)
        except Exception as exc:
            print(f"ERROR: Retrieval failed: {exc}", file=sys.stderr)
            continue

        context = format_docs(docs)

        try:
            if llm is None:
                answer = extractive_fallback_answer(context=context, question=question)
            else:
                try:
                    answer = answer_with_llm(llm=llm, context=context, question=question)
                except Exception as exc:
                    print(f"WARNING: LLM call failed ({exc}); using extractive fallback.", file=sys.stderr)
                    answer = extractive_fallback_answer(context=context, question=question)
        except Exception as exc:
            print(f"ERROR: Failed to generate answer: {exc}", file=sys.stderr)
            continue

        print(answer)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
