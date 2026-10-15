"""Dense encoder wrapper that enforces the e5 ``passage:`` / ``query:`` contract.

SPEC section 9.1: use ``intfloat/multilingual-e5-base`` and prefix every passage with
``passage: `` and every query with ``query: ``. The prefixes are part of the e5 input
format — omitting them silently degrades retrieval instead of failing, so this module
makes the convention explicit and testable:

* :class:`Encoder` is the interface the index builder and the API depend on.
* :class:`E5Encoder` is the real implementation (lazy import, so the fast test suite
  never needs torch).
* every ``E5Encoder`` encode call passes the prefixed strings through
  :func:`prefixed`, which rejects a query passed as a passage (and vice versa).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class EncoderError(RuntimeError):
    """The embedding model could not be loaded or run."""


@runtime_checkable
class Encoder(Protocol):
    """What the index builder and the API need from an embedding model."""

    model_id: str
    revision: str

    @property
    def dimension(self) -> int:
        """Dense vector length."""
        ...

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks, with the passage prefix applied."""
        ...

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed user queries, with the query prefix applied."""
        ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Model tokenizer counts, so the chunker can respect the embedding window."""
        ...


def prefixed(texts: Sequence[str], prefix: str, *, forbidden: str) -> list[str]:
    """Prepend ``prefix`` to every input, refusing input carrying ``forbidden``.

    The ``forbidden`` check is the failure-sensitive part: feeding a query through
    :meth:`E5Encoder.encode_passages` would otherwise produce a plausible-looking but
    wrong vector, so it is rejected loudly instead.
    """
    wrong = [text for text in texts if text.startswith(forbidden)]
    if wrong:
        sample = wrong[0][:80]
        msg = (
            f"input already carries {forbidden!r} (e.g. {sample!r}); "
            "use the matching encoder method"
        )
        raise ValueError(msg)
    return [prefix + text for text in texts]


class E5Encoder:
    """Real ``multilingual-e5-base`` encoder, loaded from a pinned revision."""

    def __init__(
        self,
        model: object,
        *,
        model_id: str,
        revision: str,
        batch_size: int = 32,
    ) -> None:
        self._model = model
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        """Dense vector length reported by the loaded model."""
        size = self._model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
        if not isinstance(size, int) or size <= 0:
            msg = f"{self.model_id}: model reported an invalid embedding size {size!r}"
            raise EncoderError(msg)
        return size

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 512,
        cache_dir: Path | None = None,
    ) -> E5Encoder:
        """Load the model, preferring safetensors and never remote code.

        Imported here rather than at module import: torch and sentence-transformers are
        only needed by the indexer and the API process, not by the fast test suite.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - environment dependent
            msg = (
                "sentence-transformers is not installed; install the retrieval extra "
                '(`pip install -e ".[retrieval]"`)'
            )
            raise EncoderError(msg) from error
        try:
            model = SentenceTransformer(
                model_id,
                revision=revision,
                device=device,
                cache_folder=str(cache_dir) if cache_dir is not None else None,
                model_kwargs={"use_safetensors": True},
            )
        except Exception as error:  # reported as an encoder failure, not a crash
            msg = f"could not load {model_id}@{revision}: {error}"
            raise EncoderError(msg) from error
        model.max_seq_length = max_seq_length
        model.eval()
        return cls(model, model_id=model_id, revision=revision, batch_size=batch_size)

    def _encode(self, texts: Sequence[str], prefix: str, forbidden: str) -> list[list[float]]:
        inputs = prefixed(texts, prefix, forbidden=forbidden)
        vectors = self._model.encode(  # type: ignore[attr-defined]
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunks with ``passage: `` prepended."""
        return self._encode(texts, PASSAGE_PREFIX, forbidden=QUERY_PREFIX)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries with ``query: `` prepended."""
        return self._encode(texts, QUERY_PREFIX, forbidden=PASSAGE_PREFIX)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Real tokenizer counts, including the model's special tokens."""
        tokenizer = self._model.tokenizer  # type: ignore[attr-defined]
        encoded = tokenizer(list(texts), add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]


class PrefixEnforcingEncoder:
    """Wrap any encoder and re-assert the prefix contract on every call.

    Used where a caller could pass a mismatched encoder (tests, future rerankers): the
    boundary check makes a silent mismatch impossible.
    """

    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder
        self.model_id = encoder.model_id
        self.revision = encoder.revision

    @property
    def dimension(self) -> int:
        return self._encoder.dimension

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed(texts, PASSAGE_PREFIX, forbidden=QUERY_PREFIX)
        return self._encoder.encode_passages(texts)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed(texts, QUERY_PREFIX, forbidden=PASSAGE_PREFIX)
        return self._encoder.encode_queries(texts)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return self._encoder.count_tokens(texts)


__all__ = [
    "PASSAGE_PREFIX",
    "QUERY_PREFIX",
    "E5Encoder",
    "Encoder",
    "EncoderError",
    "PrefixEnforcingEncoder",
    "prefixed",
]
