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
- For clinical numbers, quote the complete source sentence verbatim by writing ONLY its \
[quote:N] placeholder from the Source quotations list. The server inserts that sentence \
in its original language and adds its citation. Never type the quotation yourself.
- Do not write any digits outside source citation keys or [quote:N] placeholders. \
Use bullet points, not numbered lists. Do not repeat numbers from the question, even to deny them.
- Never translate, calculate, round, reorder or restate clinical numbers. Preserve units, \
comparison groups and values linked by "respectively" or "αντίστοιχα" by using the placeholder. \
If the sources do not explicitly state the requested relationship, say that it cannot be determined.
- Otherwise summarize qualitatively and concisely in the answer language. Keep each quotation \
separate from your explanation. Do not imply that a quoted figure belongs to a different group.
- Report what the documents say. Do not give personal medical advice."""


class LLMError(RuntimeError):
    """The LLM provider failed, timed out or returned an error."""


class NumericalGroundingError(ValueError):
    """Generated digits or an unknown quote reference cannot be shown safely."""


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


def build_messages(
    question: str, sources: Sequence[SearchHit], lang: str, *, source_quotes_only: bool = False,
) -> list[dict[str, str]]:
    """Chat messages: the rules as system prompt, the keyed sources and the question."""
    blocks = [f"[{citation_key(hit)}] {hit.title}\n{hit.text}" for hit in sources]
    quotes = "\n".join(
        f"[{key}] {text} [{citation}]"
        for key, (text, citation) in _source_quotes(sources).items()
    )
    user = (
        "Sources:\n\n" + "\n\n".join(blocks)
        + "\n\nSource quotations (select placeholders; do not rewrite):\n" + quotes
        + f"\n\nQuestion: {question}"
    )
    system = SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[lang])
    if source_quotes_only:
        system = (
            "Select up to five source quotations that directly state the requested facts "
            "or correct a false premise. Do not select section headings, navigation labels, "
            "or text that merely mentions the drug or its doses. If the retrieved text does "
            "not state the requested facts, return an empty list. Return only the quote_ids "
            "in the required JSON format. Never write or paraphrase clinical facts."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _source_quotes(sources: Sequence[SearchHit]) -> dict[str, tuple[str, str]]:
    """Numbered source lines, without sentence splitting that could break decimal values.

    The parser emits sentence lines. A source with no line breaks becomes one longer
    quotation. Either way, rendering preserves its text rather than reconstructing facts.
    """
    quotes = {}
    for hit in sources:
        for line in hit.text.splitlines():
            if line.strip():
                quotes[f"quote:{len(quotes) + 1}"] = (line.strip(), citation_key(hit))
    return quotes


def quote_selection_format(sources: Sequence[SearchHit]) -> dict[str, object]:
    """Provider-enforced source IDs, never model-written medical prose."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "source_selection", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {"quote_ids": {
                    "type": "array", "maxItems": 5,
                    "items": {"type": "string", "enum": list(_source_quotes(sources))},
                }},
                "required": ["quote_ids"],
            },
        },
    }


def parse_quote_selection(raw: str, sources: Sequence[SearchHit], lang: str) -> str:
    """Validate again locally, even if a provider ignores the requested schema."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LLMError("Invalid source-selection JSON") from error
    if not isinstance(value, dict) or set(value) != {"quote_ids"}:
        raise LLMError("Invalid source-selection fields")
    refs = value["quote_ids"]
    if not isinstance(refs, list) or len(refs) > 5:
        raise LLMError("Invalid source-selection list")
    allowed = _source_quotes(sources)
    if any(not isinstance(ref, str) or ref not in allowed for ref in refs):
        raise NumericalGroundingError("Unknown source quotation")
    if not refs:
        return (
            "Οι πηγές που ανακτήθηκαν δεν απαντούν στην ερώτηση." if lang == "el"
            else "The retrieved sources do not answer this question."
        )
    heading = (
        "Σχετικά αποσπάσματα από τις επίσημες πληροφορίες προϊόντος:" if lang == "el"
        else "Relevant excerpts from the official product information:"
    )
    return heading + "\n\n" + "\n\n".join(f"- [{ref}]" for ref in dict.fromkeys(refs))


def render_numerical_quotes(answer: str, sources: Sequence[SearchHit]) -> str:
    """Expand trusted source references, rejecting generated digits before display.

    This is not a semantic verifier: a model can still choose an irrelevant quotation
    or give an incorrect qualitative explanation. Call before emitting ANY answer text.
    """
    quotes = _source_quotes(sources)
    reference = re.compile(r"\[quote:(\d+)\]")

    def resolve(match: re.Match[str]) -> str:
        key = f"quote:{match[1]}"
        if key not in quotes:
            raise NumericalGroundingError("Unknown numerical source quotation")
        text, citation = quotes[key]
        return f'“{text}” [{citation}]'

    # Validate references even when the rest of the answer contains no raw digits.
    for match in reference.finditer(answer):
        resolve(match)
    unquoted = reference.sub("", answer)
    citations = CitationFilter({citation_key(hit) for hit in sources})
    unquoted = citations.feed(unquoted) + citations.flush()
    unquoted = re.sub(r"\[[^\]]*\]", "", unquoted)  # remaining keys are known citations
    if any(char.isdigit() for char in unquoted):
        raise NumericalGroundingError("Unquoted generated number")
    return reference.sub(resolve, answer)


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
        # Treat typographic spaces like ASCII spaces, without accepting new source keys.
        keys = [" ".join(key.split()) for key in re.split(r"[;,]", inner) if key.strip()]
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
    response_format: dict[str, object] | None = None,
) -> AsyncIterator[str]:
    """Yield answer text from an OpenAI-compatible ``/chat/completions`` stream.

    ``llm_timeout_s`` bounds each network wait (connect, and the gap between tokens), so a
    stalled provider fails fast while a long answer that keeps streaming is not cut off.
    """
    headers = {}
    if settings.llm_api_key and settings.llm_api_key.get_secret_value():  # Ollama needs none
        headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
    payload: dict[str, object] = {
        "model": settings.llm_model, "messages": messages, "stream": True, "temperature": 0,
    }
    if response_format is not None:
        payload["response_format"] = response_format
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
