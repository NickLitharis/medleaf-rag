"""Prompt, streaming LLM call and citation post-check for ``/v1/ask``.

The LLM is reached through any OpenAI-compatible chat API (OpenRouter by default; OpenAI
and Ollama work by changing ``MEDLEAF_LLM_BASE_URL``). Every stage is a plain function so
it can be tested without a network.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence

import httpx

from medleaf.config import Settings
from medleaf.retrieval.service import SearchHit

GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
LETTER = re.compile(r"[^\W\d_]")
LANGUAGE_NAMES = {"en": "English", "el": "Greek"}
DISCLAIMERS = {
    "en": (
        "This summarises the official EU product information and is not medical advice. "
        "Check with your doctor or pharmacist."
    ),
    "el": (
        "Αυτή είναι σύνοψη των επίσημων πληροφοριών προϊόντος της ΕΕ και δεν αποτελεί "
        "ιατρική συμβουλή. Συμβουλευτείτε τον γιατρό ή τον φαρμακοποιό σας."
    ),
}
SYSTEM_PROMPT = """You answer questions about EU medicines using ONLY the sources given \
by the user. The sources are excerpts from official EMA product information.
Rules:
- Write the answer in {language}.
- After every sentence that states a fact, cite its source key in square brackets, \
exactly as written, e.g. [Eliquis SmPC 4.5].
- Use only the keys listed with the sources. Never invent a key.
- If the sources do not answer the question, or the question assumes something the \
sources contradict, say so plainly and do not guess.
- Report what the documents say. Do not give personal medical advice."""


class LLMError(RuntimeError):
    """The LLM provider failed, timed out or returned an error."""


def detect_lang(text: str) -> str:
    """``el`` when at least 30% of the letters are Greek, else ``en`` (SPEC §9 step 1)."""
    letters = LETTER.findall(text)
    greek = sum(1 for letter in letters if GREEK.match(letter))
    return "el" if letters and greek / len(letters) >= 0.3 else "en"


def citation_key(hit: SearchHit) -> str:
    """Citation key for a chunk, e.g. ``Eliquis SmPC 4.5`` (all parts share it)."""
    doc = "SmPC" if hit.doc == "smpc" else "Leaflet"
    return f"{hit.medicine.capitalize()} {doc} {hit.section}"


def select_sources(hits: Sequence[SearchHit], lang: str, k: int) -> list[SearchHit]:
    """First ``k`` distinct chunks by rank; a chunk found in both languages keeps the
    version in the question's language, so its text and URL match the answer."""
    chosen: dict[str, SearchHit] = {}
    for hit in hits:
        current = chosen.get(hit.chunk_id)
        if current is None:
            if len(chosen) < k:
                chosen[hit.chunk_id] = hit
        elif current.lang != lang and hit.lang == lang:
            chosen[hit.chunk_id] = hit
    return list(chosen.values())


def build_messages(question: str, sources: Sequence[SearchHit], lang: str) -> list[dict[str, str]]:
    """Chat messages: the rules as system prompt, the keyed sources and the question."""
    blocks = [f"[{citation_key(hit)}] {hit.title}\n{hit.text}" for hit in sources]
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[lang])},
        {"role": "user", "content": user},
    ]


class CitationFilter:
    """Streams answer text through, dropping ``[keys]`` that are not real sources.

    Text from an opening ``[`` is held back until its ``]`` arrives, so an invented key
    never reaches the client (SPEC §9 step 6). ``used`` and ``invented`` record what was
    seen, in order and without repeats.
    """

    MAX_HELD = 80  # an unclosed "[" longer than this is ordinary text, not a citation

    def __init__(self, allowed: set[str]) -> None:
        self.allowed = allowed
        self.used: list[str] = []
        self.invented: list[str] = []
        self._held = ""

    def feed(self, text: str) -> str:
        self._held += text
        out: list[str] = []
        while True:
            start = self._held.find("[")
            if start < 0:
                out.append(self._held)
                self._held = ""
                break
            out.append(self._held[:start])
            self._held = self._held[start:]
            end = self._held.find("]")
            if end < 0:
                if len(self._held) <= self.MAX_HELD:
                    break
                out.append(self._held[0])
                self._held = self._held[1:]
                continue
            out.append(self._check(self._held[1:end]))
            self._held = self._held[end + 1 :]
        return "".join(out)

    def flush(self) -> str:
        """Release whatever is still held at the end of the stream."""
        held, self._held = self._held, ""
        return held

    def _check(self, inner: str) -> str:
        # Models sometimes group keys: "[Eliquis SmPC 4.5; Eliquis SmPC 4.3]".
        keys = [key.strip() for key in re.split(r"[;,]", inner) if key.strip()]
        valid = [key for key in keys if key in self.allowed]
        for key in keys:
            seen = self.used if key in self.allowed else self.invented
            if key not in seen:
                seen.append(key)
        return f"[{'; '.join(valid)}]" if valid else ""


async def stream_chat(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[str]:
    """Yield answer text from an OpenAI-compatible ``/chat/completions`` stream.

    ``llm_timeout_s`` bounds each network wait (connect, and the gap between tokens), so a
    stalled provider fails fast while a long answer that keeps streaming is not cut off.
    """
    headers = {}
    if settings.llm_api_key and settings.llm_api_key.get_secret_value():  # Ollama needs none
        headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
    payload = {"model": settings.llm_model, "messages": messages, "stream": True, "temperature": 0}
    try:
        async with (
            httpx.AsyncClient(
                base_url=settings.llm_base_url, timeout=settings.llm_timeout_s, transport=transport
            ) as client,
            client.stream("POST", "chat/completions", json=payload, headers=headers) as response,
        ):
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:300]
                msg = f"{settings.llm_model}: HTTP {response.status_code}: {body}"
                raise LLMError(msg)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank lines and ": keep-alive" comments
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if "error" in chunk:
                    msg = f"{settings.llm_model}: {chunk['error']}"
                    raise LLMError(msg)
                choices = chunk.get("choices") or []
                text = choices[0].get("delta", {}).get("content") if choices else None
                if text:
                    yield text
    except httpx.HTTPError as error:
        msg = f"{settings.llm_model}: {type(error).__name__}: {error}"
        raise LLMError(msg) from error
