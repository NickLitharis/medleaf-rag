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
TOKENIZER_VERSION = "medleaf-tokenizer-2"

#: Folded Greek inflectional endings, longest first. One is stripped per token so
#: ``ναυτίας``/``ναυτία`` and ``δόσεις``/``δόση`` share a token (SPEC section 17 risk).
_GREEK_ENDINGS: tuple[str, ...] = (
    "ουσ", "εισ", "εων",
    "ησ", "εσ", "ασ", "οσ", "ισ", "ων", "ου", "οι", "αι", "ει",
    "α", "η", "ο", "ι", "ε",
)

#: A stem shorter than this would merge unrelated Greek words, so the ending is kept.
_MIN_STEM_LEN = 3

_GREEK_WORD_RE = re.compile(r"[α-ω]+")

#: Query-side function words. They carry no retrieval signal, and in a long chunk their
#: raw term frequency can outrank the single rare content word that decides the answer
#: (the sparse values are term frequencies; Qdrant's IDF cannot saturate them). The
#: index keeps these tokens, so this is a query-only filter and needs no rebuild.
STOPWORDS: frozenset[str] = frozenset(
    {
        "what", "are", "the", "of", "is", "for", "a", "an", "to", "in", "on",
        "and", "or", "how", "does", "do", "with", "by", "at", "it", "this",
        "that", "which", "from", "as", "was", "were", "be", "been", "being",
        "have", "has", "had", "you", "your", "can", "should", "must", "may",
        "who", "whom", "when", "where", "why",
        # Greek
        "ποιες", "ποιεσ", "ειναι", "οι", "του", "τησ", "των", "και", "να",
        "το", "η", "ο", "σε", "με", "για", "που", "τι", "πωσ", "πως", "απο",
        "δεν", "ενα", "μια", "τον", "την", "στο", "στη", "θα", "ποια", "ποιο",
    }
)


def fold_text(text: str) -> str:
    """Lowercase and strip diacritics so inflected Greek forms share a token.

    NFD decomposition makes stripping complete: ``ά`` becomes ``α`` + U+0301, so
    removing combining marks also removes tonos and dialytika. Final sigma is folded
    to ``σ`` because it is a positional variant of the same letter.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return stripped.replace("ς", "σ")


def stem(token: str) -> str:
    """Strip one inflectional ending from a folded, all-Greek token; others pass through.

    ponytail: suffix list, not a real Greek stemmer; swap in one if the eval gate
    shows over- or under-stemming.
    """
    if not _GREEK_WORD_RE.fullmatch(token):
        return token
    for ending in _GREEK_ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= _MIN_STEM_LEN:
            return token[: -len(ending)]
    return token


def _folded_words(text: str) -> list[str]:
    folded = fold_text(text)
    return [token for token in _WORD_RE.findall(folded) if len(token) >= MIN_TOKEN_LEN]


def tokenize(text: str) -> list[str]:
    """Split text into folded, stemmed tokens of at least :data:`MIN_TOKEN_LEN` characters."""
    return [stem(token) for token in _folded_words(text)]


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


def query_sparse_vector(text: str) -> models.SparseVector:
    """Sparse vector for a user query, with function words removed.

    The stored passages keep their function words; matching them from the query is what
    is dropped here, so a long passage cannot win on ``the``/``of`` frequency alone.
    """
    # Stopwords are matched before stemming: they are listed in folded, unstemmed form.
    counts = Counter[str](
        stem(token) for token in _folded_words(text) if token not in STOPWORDS
    )
    pairs = sorted((token_id(token), float(count)) for token, count in counts.items())
    return models.SparseVector(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )


def drop_terms(text: str, terms: Sequence[str]) -> str:
    """Remove tokens already applied as an exact filter (e.g. the medicine name).

    ``medleaf search --medicine eliquis`` already restricts retrieval to that medicine,
    so the brand token only adds a signal every chunk shares; dropping it lets the
    section words drive the query. The comparison is on folded text, so an accented or
    capitalised spelling still matches.
    """
    dropped = {fold_text(term) for term in terms if term}
    if not dropped:
        return text
    return " ".join(word for word in _WORD_RE.findall(text) if fold_text(word) not in dropped)


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
    "STOPWORDS",
    "TOKENIZER_VERSION",
    "dedupe",
    "drop_terms",
    "fold_text",
    "query_sparse_vector",
    "sparse_pairs",
    "sparse_vector",
    "stem",
    "token_id",
    "tokenize",
    "vocabulary_size",
]
