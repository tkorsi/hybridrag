"""Streamlit RAG web app for querying *A Game of Thrones*.

Deployable to Streamlit Cloud:
  - UI: Streamlit chat interface
  - LLM: Groq via `langchain-groq` (model: llama-3.1-8b-instant)
  - Retrieval: Chroma (local persistence) + HuggingFaceEmbeddings (CPU)

This app routes queries into two paths:
  1) Analytics (no RAG): approximate counting/frequency questions using regex + Counter.
  2) Semantic (RAG): retrieve relevant chunks and ask the LLM to answer strictly from context.
"""

from __future__ import annotations

import sys
import os

# Streamlit Cloud may ship with an older system SQLite build, which can break Chroma.
# If available, prefer the bundled `pysqlite3` (installed via `pysqlite3-binary`).
try:  # pragma: no cover
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import streamlit as st


BOOK_PATH = Path("data/game_of_thrones.txt")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "game_of_thrones"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a Maester of the Citadel. "
    "Answer strictly based on the provided context. "
    "If the answer is not in the context, say 'I do not know'."
)


ANALYTICS_PATTERNS = [
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bcount\b", re.IGNORECASE),
    re.compile(r"\btop\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bmost\s+frequent\b", re.IGNORECASE),
    re.compile(r"\bmost\s+mentioned\b", re.IGNORECASE),
]


def is_analytics_query(question: str) -> bool:
    return any(p.search(question) for p in ANALYTICS_PATTERNS)


def is_biography_query(question: str) -> bool:
    return bool(re.search(r"\b(?:biography|bio)\b", question, re.IGNORECASE)) or bool(
        re.search(r"\bwrite\s+(?:a\s+)?biography\s+(?:of|for)\b", question, re.IGNORECASE)
    )


def parse_top_n(question: str, default: int = 5) -> int:
    m = re.search(r"\btop\s+(\d+)\b", question, re.IGNORECASE)
    if not m:
        return default
    try:
        n = int(m.group(1))
    except ValueError:
        return default
    return max(1, min(n, 50))


def extract_name_for_count(question: str) -> str | None:
    m = re.search(r"\bcount(?:\s+mentions)?\s+of\s+(.+)$", question, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip().strip("\"' ")
    return name or None


SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_CHARS_PER_CHUNK = 100_000


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


STOPWORDS_LOWER = {w.lower() for w in STOPWORDS}
TITLES_LOWER = {t.lower() for t in TITLES}
NAME_PART_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def significant_name_parts(text: str) -> list[str]:
    """Extract "significant" name parts from user input or entity text.

    Used for fuzzy matching in analytics (e.g., "Arya Stark" should match "Arya").
    """
    parts = [p.strip("'-").lower() for p in NAME_PART_RE.findall(text)]
    return [
        p
        for p in parts
        if len(p) > 1 and p not in STOPWORDS_LOWER and p not in TITLES_LOWER
    ]


def get_counts(counts: Counter[str], target_name: str, *, ambiguity_threshold: int = 2) -> tuple[int, list[str], list[str]]:
    """Return a fuzzy mention count for a target character name.

    Matching strategy:
      - Split `target_name` into significant tokens (e.g., ["arya", "stark"]).
      - Always include the first token (first-name priority).
      - Include additional tokens only if they are not ambiguous across many entities
        (e.g., "stark" appears in many names, so it is often too broad).
      - Count any PERSON entity whose token set overlaps with the chosen tokens.
    """
    target_name = normalize_name(target_name)
    target_parts = significant_name_parts(target_name)
    if not target_parts:
        return 0, [], []

    # Build an index of which entity-names contain which tokens (case-insensitive).
    token_to_entities: dict[str, set[str]] = defaultdict(set)
    entity_tokens: dict[str, set[str]] = {}
    for entity in counts.keys():
        tokens = set(significant_name_parts(entity))
        entity_tokens[entity] = tokens
        for t in tokens:
            token_to_entities[t].add(entity)

    primary = target_parts[0]
    chosen_tokens: list[str] = [primary]

    if len(target_parts) > 1:
        for token in target_parts[1:]:
            if len(token_to_entities.get(token, set())) <= ambiguity_threshold:
                chosen_tokens.append(token)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    chosen_tokens = [t for t in chosen_tokens if not (t in seen or seen.add(t))]

    matched_entities = [e for e, toks in entity_tokens.items() if toks.intersection(chosen_tokens)]
    total = sum(counts[e] for e in matched_entities)

    # Prefer showing more frequent variants first.
    matched_entities.sort(key=lambda e: counts[e], reverse=True)
    return total, matched_entities, chosen_tokens


def iter_chunks(text: str, size: int) -> Iterable[str]:
    for start in range(0, len(text), size):
        yield text[start : start + size]


def clean_person_name(name: str) -> str:
    name = name.strip(" \t\n\r\"'()[]{}-:;.,!?")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"(?:'s|’s)$", "", name).strip()

    first = name.split(" ", 1)[0] if name else ""
    if first in TITLES:
        name = name.split(" ", 1)[1] if " " in name else ""

    return name.strip()


def _spacy_installed() -> bool:
    try:
        import spacy  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class SpacyNlpResult:
    nlp: Any | None
    note: str | None = None


def find_spacy_model_path(base_path: str) -> str | None:
    """Recursively search for a spaCy model root by locating `config.cfg`.

    Some environments/vendor packaging can produce nested directories (e.g. versioned folders).
    We treat the directory containing `config.cfg` as the model root for `spacy.load(path)`.
    """
    for root, _dirs, files in os.walk(base_path):
        if "config.cfg" in files:
            return root
    return None


@st.cache_resource(show_spinner="Loading spaCy NER model…")
def load_spacy_model(model_name: str = SPACY_MODEL_NAME) -> SpacyNlpResult:
    import spacy

    disable = ["tagger", "parser", "attribute_ruler", "lemmatizer"]

    try:
        # 1) Standard: load by installed package name.
        nlp = spacy.load(model_name, disable=disable)
        return SpacyNlpResult(nlp=nlp)
    except Exception as name_exc:
        # 2) Pre-built fallback: load from a local committed directory.
        base_path = f"./spacy_models/{model_name}"
        discovered_path = find_spacy_model_path(base_path)
        if not discovered_path:
            return SpacyNlpResult(
                nlp=None,
                note=(
                    f"spaCy model '{model_name}' could not be loaded.\n"
                    f"- Tried package: spacy.load('{model_name}') -> {type(name_exc).__name__}: {name_exc}\n"
                    f"- Tried path discovery under: {base_path} (no config.cfg found)\n"
                    "Falling back to regex heuristic."
                ),
            )

        try:
            nlp = spacy.load(discovered_path, disable=disable)
            return SpacyNlpResult(nlp=nlp, note=f"Loaded spaCy model from local path: {discovered_path}")
        except Exception as path_exc:
            return SpacyNlpResult(
                nlp=None,
                note=(
                    f"spaCy model '{model_name}' could not be loaded.\n"
                    f"- Tried package: spacy.load('{model_name}') -> {type(name_exc).__name__}: {name_exc}\n"
                    f"- Found config at: {discovered_path}\n"
                    f"- Tried path:    spacy.load('{discovered_path}') -> {type(path_exc).__name__}: {path_exc}\n"
                    "Falling back to regex heuristic."
                ),
            )


def analyze_text_spacy(full_text: str) -> Counter[str]:
    """Count PERSON entity mentions using spaCy NER (preferred)."""
    nlp_result = load_spacy_model()
    if nlp_result.nlp is None:
        raise RuntimeError(nlp_result.note or "spaCy model unavailable.")
    nlp = nlp_result.nlp

    counts_lower: Counter[str] = Counter()
    display_names: dict[str, str] = {}

    for doc in nlp.pipe(iter_chunks(full_text, SPACY_CHARS_PER_CHUNK), batch_size=4):
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            name = clean_person_name(ent.text)
            if not name:
                continue

            key = name.lower()
            counts_lower[key] += 1
            display_names.setdefault(key, name)

    # Present human-friendly casing while still aggregating case-insensitively.
    return Counter({display_names[k]: v for k, v in counts_lower.items()})


def analyze_text_regex_heuristic(full_text: str) -> Counter[str]:
    """Fallback name frequency heuristic (used when spaCy isn't available)."""
    counts: Counter[str] = Counter()
    for match in NAME_REGEX.finditer(full_text):
        candidate = normalize_name(match.group(0))
        if not candidate:
            continue
        if candidate in STOPWORDS or candidate in TITLES:
            continue

        first_token = candidate.split(" ", 1)[0]
        if first_token in STOPWORDS:
            continue

        counts[candidate] += 1
    return counts


@dataclass(frozen=True)
class AnalyticsResult:
    backend: str
    counts: Counter[str]
    note: str | None = None


def analyze_text(full_text: str) -> AnalyticsResult:
    """Analyze character frequencies from the full book text.

    Prefer spaCy PERSON NER when available; otherwise fall back to a regex heuristic.
    """
    if _spacy_installed():
        try:
            return AnalyticsResult(
                backend="spaCy PERSON NER",
                counts=analyze_text_spacy(full_text),
            )
        except Exception as exc:
            return AnalyticsResult(
                backend="Regex heuristic",
                counts=analyze_text_regex_heuristic(full_text),
                note=str(exc)
                or f"spaCy unavailable ({type(exc).__name__}). Install `spacy` + `{SPACY_MODEL_NAME}` for PERSON-only counts.",
            )

    return AnalyticsResult(
        backend="Regex heuristic",
        counts=analyze_text_regex_heuristic(full_text),
        note=f"Install `spacy` + `{SPACY_MODEL_NAME}` for PERSON-only counts.",
    )


@st.cache_data(show_spinner=False)
def load_book_text(book_path: str) -> str:
    return Path(book_path).read_text(encoding="utf-8", errors="replace")


@st.cache_data(show_spinner="Analyzing the full text…")
def cached_analyze_text(book_path: str, book_mtime: float) -> AnalyticsResult:
    _ = book_mtime  # cache-buster when the file changes
    full_text = load_book_text(book_path)
    return analyze_text(full_text)


def answer_analytics(*, question: str, book_path: str, book_mtime: float) -> str:
    result = cached_analyze_text(book_path, book_mtime)
    counts = result.counts

    footer = f"_Count source: {result.backend}._"
    if result.note:
        footer += f"\n\n_Note: {result.note}_"

    requested_name = extract_name_for_count(question)
    if requested_name:
        direct, matched_variants, chosen_tokens = get_counts(counts, requested_name)
        suggestions: list[str] = []
        if direct == 0:
            # Suggest close variants by token overlap.
            requested_parts = set(significant_name_parts(requested_name))
            if requested_parts:
                suggestions = [
                    k
                    for k in counts.keys()
                    if set(significant_name_parts(k)).intersection(requested_parts)
                ][:5]
        return (
            f"Mentions of **{requested_name}** (approx): **{direct}**\n\n"
            + (
                f"Matched on tokens: {', '.join(f'`{t}`' for t in chosen_tokens)}\n\n"
                if chosen_tokens
                else ""
            )
            + (
                "Counted variants: "
                + ", ".join(f"`{v}`" for v in matched_variants[:8])
                + ("\n\n" if matched_variants else "")
            )
            + (f"Did you mean: {', '.join(f'`{s}`' for s in suggestions)}\n\n" if suggestions else "")
            + footer
        )

    if re.search(r"\bhow many\b.*\bcharacters\b", question, re.IGNORECASE):
        unique_est = sum(1 for _, c in counts.items() if c >= 5)
        return (
            "Approximate unique characters (>= 5 PERSON mentions): "
            f"**{unique_est}**\n\n{footer}"
        )

    n = parse_top_n(question, default=5)
    top = counts.most_common(n)
    if not top:
        return f"No PERSON entities were detected.\n\n{footer}"

    lines = [f"Top **{n}** most-mentioned characters (approx):", ""]
    for name, count in top:
        lines.append(f"- **{name}**: {count}")
    lines.append("")
    lines.append(footer)
    return "\n".join(lines)


def format_docs(docs: Iterable[Any]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        text = getattr(d, "page_content", "")
        if not text:
            continue
        parts.append(f"[Chunk {i}]\n{text}")
    return "\n\n".join(parts)


def retrieve_docs(vectorstore: Any, *, question: str, k: int) -> list[Any]:
    as_retriever = getattr(vectorstore, "as_retriever", None)
    if callable(as_retriever):
        retriever = as_retriever(search_kwargs={"k": k})
        invoke = getattr(retriever, "invoke", None)
        if callable(invoke):
            return invoke(question)
        get_relevant_documents = getattr(retriever, "get_relevant_documents", None)
        if callable(get_relevant_documents):
            return get_relevant_documents(question)

    similarity_search = getattr(vectorstore, "similarity_search", None)
    if callable(similarity_search):
        return similarity_search(question, k=k)

    raise TypeError("Unsupported vectorstore interface (expected .as_retriever or .similarity_search)")


def build_llm() -> Any:
    """Build the Groq chat model using Streamlit secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "Missing `GROQ_API_KEY` in Streamlit secrets. "
            "For local dev, create `.streamlit/secrets.toml` with:\n"
            'GROQ_API_KEY="..."\n'
        ) from exc

    from langchain_groq import ChatGroq

    try:
        return ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL, temperature=0)
    except TypeError:
        # Compatibility with older parameter names
        return ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0)


@st.cache_resource(show_spinner="Loading vector database…")
def get_vectorstore(*, persist_dir: str, collection: str, embedding_model: str) -> Any:
    """Load a persisted Chroma vector store.

    This app is designed for a pre-built deployment: the Chroma persistence directory
    should already exist (e.g., committed by CI) so Streamlit doesn't have to build it.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"})

    persist_path = Path(persist_dir)
    if not persist_path.exists() or not any(persist_path.iterdir()):
        raise FileNotFoundError(
            f"Chroma persistence directory is missing/empty: {persist_path}. "
            "Run `python ingest.py` to generate `./chroma_db` (or ensure CI committed prebuilt artifacts)."
        )

    try:
        return Chroma(
            collection_name=collection,
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )
    except TypeError:
        return Chroma(
            collection_name=collection,
            persist_directory=str(persist_path),
            embedding=embeddings,
        )


@st.cache_resource(show_spinner=False)
def get_prompt() -> Any:
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
        ]
    )


@dataclass
class AppConfig:
    book_path: str
    persist_dir: str
    collection: str
    embedding_model: str
    k: int
    bio_k: int
    show_context: bool


def render_sidebar() -> AppConfig:
    st.sidebar.header("Settings")

    book_path = st.sidebar.text_input("Book path", value=str(BOOK_PATH))
    persist_dir = st.sidebar.text_input("Chroma directory", value=str(CHROMA_DIR))
    collection = st.sidebar.text_input("Chroma collection", value=COLLECTION_NAME)
    embedding_model = st.sidebar.text_input("Embedding model", value=EMBEDDING_MODEL)

    k = st.sidebar.slider("k (semantic)", min_value=1, max_value=10, value=4, step=1)
    bio_k = st.sidebar.slider("k (biography)", min_value=3, max_value=15, value=7, step=1)
    show_context = st.sidebar.checkbox("Show retrieved context", value=False)

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []

    st.sidebar.markdown("---")
    st.sidebar.caption("Analytics queries (count/top/how many) bypass RAG and compute heuristics on the full text.")

    return AppConfig(
        book_path=book_path,
        persist_dir=persist_dir,
        collection=collection,
        embedding_model=embedding_model,
        k=k,
        bio_k=bio_k,
        show_context=show_context,
    )


def main() -> None:
    st.set_page_config(page_title="Game of Thrones RAG", layout="wide")
    st.title("Game of Thrones RAG")
    st.caption("Ask questions about the book. Analytics queries are approximations.")

    cfg = render_sidebar()

    book_file = Path(cfg.book_path)
    book_available = book_file.exists()
    if not book_available:
        st.warning(
            f"Book text file not found at `{cfg.book_path}`. "
            "Semantic (RAG) queries can still work if `./chroma_db` is present, "
            "but analytics queries require the full text."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if is_analytics_query(question):
                if not book_available:
                    answer = (
                        f"Analytics requires the full book text at `{cfg.book_path}`.\n\n"
                        "Provide the file (or change the path in the sidebar) and try again."
                    )
                else:
                    book_mtime = book_file.stat().st_mtime
                    answer = answer_analytics(
                        question=question,
                        book_path=cfg.book_path,
                        book_mtime=book_mtime,
                    )
                st.markdown(answer)
            else:
                vectorstore = get_vectorstore(
                    persist_dir=cfg.persist_dir,
                    collection=cfg.collection,
                    embedding_model=cfg.embedding_model,
                )

                k = cfg.bio_k if is_biography_query(question) else cfg.k
                docs = retrieve_docs(vectorstore, question=question, k=k)
                context = format_docs(docs)

                llm = build_llm()
                prompt = get_prompt()
                messages = prompt.format_messages(context=context, question=question)
                response = llm.invoke(messages)
                answer = getattr(response, "content", str(response)).strip()

                st.markdown(answer)

                if cfg.show_context:
                    with st.expander("Retrieved context"):
                        st.text(context)
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
