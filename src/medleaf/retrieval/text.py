"""Sparse tokenisation for hybrid retrieval.

SPEC section 17 lists Greek tokenisation as a risk: accents and inflection weaken
lexical matching. The mitigation implemented here is the one the SPEC prescribes —
lowercase, strip accents (Greek tonos and dialytika included, via NFD) and normalise
final sigma, so ``ΑΝΕΠΙΘΥΜΗΤΕΣ``, ``ανεπιθύμητες`` and ``ανεπιθυμητες`` share tokens.

Tokens are mapped to Qdrant sparse indices with a stable 32-bit blake2b hash, so the
same text yields the same index in every process and no vocabulary file has to be
shipped. The vector values are raw term frequencies; IDF weighting is applied by
Qdrant itself through ``Modifier.IDF`` on the ``sparse`` vector (see
:mod:`medleaf.retrieval.index`).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

from qdrant_client import models

#: Tokens shorter than this carry no retrieval signal in this corpus and are dropped.
MIN_TOKEN_LEN = 2

#: Word characters (letters and digits, any script) without the underscore.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Bump when the tokenisation rules change, so the index manifest shows the difference.
TOKENIZER_VERSION = "medleaf-tokenizer-1"


def fold_text(text: str) -> str:
    """Lowercase and strip diacritics so inflected Greek forms share a token.

    NFD decomposition makes stripping complete: ``ά`` becomes ``α`` + U+0301, so
    removing combining marks also removes tonos and dialytika. Final sigma is folded
    to ``σ`` because it is a positional variant of the same letter.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return stripped.replace("ς", "σ")


def tokenize(text: str) -> list[str]:
    """Split text into folded tokens of at least :data:`MIN_TOKEN_LEN` characters."""
    folded = fold_text(text)
    return [token for token in _WORD_RE.findall(folded) if len(token) >= MIN_TOKEN_LEN]


def token_id(token: str) -> int:
    """Stable 32-bit index for one folded token.

    blake2b is used because the built-in ``hash()`` is randomised per process, which
    would make the sparse index unreadable by the API process that built it. A 32-bit
    space can collide; with a corpus vocabulary of tens of thousands of tokens the
    expected number of collisions is far below one, and a collision only merges two
    lexical terms rather than corrupting the vector shape.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


def sparse_pairs(text: str) -> list[tuple[int, float]]:
    """Sorted ``(index, value)`` pairs for one passage, values being term frequency."""
    counts = Counter[str](tokenize(text))
    idf_values = {token_id(token): float(count) for token, count in counts.items()}
    return sorted(idf_values.items())


def sparse_vector(text: str) -> models.SparseVector:
    """Qdrant sparse vector for a passage or a query."""
    pairs = sparse_pairs(text)
    return models.SparseVector(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )


def vocabulary_size(texts: Iterable[str]) -> int:
    """Distinct folded tokens across ``texts``; used in the index manifest."""
    tokens: set[str] = set()
    for text in texts:
        tokens.update(tokenize(text))
    return len(tokens)


def dedupe(values: Sequence[str]) -> list[str]:
    """Order-preserving de-duplication, used when building filter value lists."""
    return list(dict.fromkeys(values))


__all__ = [
    "MIN_TOKEN_LEN",
    "TOKENIZER_VERSION",
    "dedupe",
    "fold_text",
    "sparse_pairs",
    "sparse_vector",
    "token_id",
    "tokenize",
    "vocabulary_size",
]
