"""Preload dependencies required at runtime (Streamlit-friendly).

Currently: ensure the spaCy English model `en_core_web_sm` is available.

Why this exists:
  - Streamlit Cloud can have issues downloading spaCy models during app runtime.
  - This script is intended to be run once *before* starting the Streamlit app.

Usage:
  python preload.py
"""

from __future__ import annotations

import sys
from typing import Any


MODEL_NAME = "en_core_web_sm"


def _print(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_spacy_model(model_name: str = MODEL_NAME) -> int:
    """Ensure `model_name` can be loaded via `spacy.load`.

    Returns:
      0 on success, non-zero on failure.
    """
    try:
        import spacy
    except Exception as exc:
        _print(f"ERROR: spaCy is not installed ({exc}). Install it first: `pip install spacy`.")
        return 2

    def try_load() -> Any | None:
        try:
            return spacy.load(model_name)
        except Exception:
            return None

    if try_load() is not None:
        _print(f"spaCy model already available: {model_name}")
        return 0

    _print(f"spaCy model missing: {model_name}. Attempting download…")
    try:
        from spacy.cli import download as spacy_download

        spacy_download(model_name)
    except BaseException as exc:  # includes SystemExit from CLI wrappers
        _print(f"ERROR: Failed to download spaCy model '{model_name}': {exc}")
        return 3

    if try_load() is not None:
        _print(f"spaCy model installed: {model_name}")
        return 0

    _print(
        f"ERROR: spaCy model '{model_name}' still cannot be loaded after download. "
        f"Try running: `python -m spacy download {model_name}`"
    )
    return 4


def main() -> int:
    return ensure_spacy_model(MODEL_NAME)


if __name__ == "__main__":
    raise SystemExit(main())

