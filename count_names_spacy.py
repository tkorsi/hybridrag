"""Count character names in the book using spaCy NER.

This script takes a simple first pass: run spaCy's English NER model over
the text, keep PERSON entities, normalize spacing/punctuation a bit, and
report the most frequently mentioned names.
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys
from typing import Iterator

import spacy

BOOK_PATH = pathlib.Path("data/game_of_thrones.txt")
# Keep chunks modest so spaCy processes the long book without memory issues.
CHARS_PER_CHUNK = 100_000
TOP_K = 40


def iter_chunks(text: str, size: int) -> Iterator[str]:
    """Yield successive slices of `size` characters from `text`."""
    for start in range(0, len(text), size):
        yield text[start : start + size]


def clean_name(name: str) -> str:
    """Trim punctuation and collapse whitespace for more consistent counting."""
    name = name.strip(" \t\n\r\"'()[]{}-:;.,!?")
    name = re.sub(r"\s+", " ", name)
    return name


def main() -> int:
    if not BOOK_PATH.exists():
        print(f"Book not found at {BOOK_PATH}", file=sys.stderr)
        return 1

    nlp = spacy.load("en_core_web_sm")
    text = BOOK_PATH.read_text(encoding="utf-8")

    counts: collections.Counter[str] = collections.Counter()
    display_names: dict[str, str] = {}

    for doc in nlp.pipe(iter_chunks(text, CHARS_PER_CHUNK), batch_size=4):
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            name = clean_name(ent.text)
            if not name:
                continue

            key = name.lower()
            counts[key] += 1
            display_names.setdefault(key, name)

    total_mentions = sum(counts.values())
    unique_names = len(counts)

    print(f"Total PERSON mentions: {total_mentions}")
    print(f"Unique PERSON names:   {unique_names}")
    print(f"Top {TOP_K} names:\n")

    for key, count in counts.most_common(TOP_K):
        print(f"{display_names[key]:25s} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
