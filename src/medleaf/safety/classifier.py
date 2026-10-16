"""Label a question ``info``, ``personal`` or ``emergency`` before retrieval (SPEC §9).

Order: emergency keyword rules first (they override everything and skip the LLM), then
the LLM label, raised to ``personal`` when a personal keyword rule matches. The safer
label always wins, and an LLM failure counts as ``personal``, never ``info``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import AsyncIterator, Callable
from typing import Literal

from medleaf.generation.answer import LLMError

Label = Literal["info", "personal", "emergency"]
Generate = Callable[[list[dict[str, str]]], AsyncIterator[str]]
SEVERITY: dict[Label, int] = {"info": 0, "personal": 1, "emergency": 2}


def fold(text: str) -> str:
    """Lower-case and strip accents, so ``Πήρα`` matches ``πηρα``. Note that casefold
    turns the final sigma ``ς`` into ``σ``; the patterns below are written that way."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


# Patterns run on fold(question). They need a first-person or present-tense signal, so
# "What are the symptoms of overdose?" or "Does Lyrica cause suicidal thoughts?" stay info.
EMERGENCY = [re.compile(pattern) for pattern in (
    # English
    r"\b(i|i've|ive|i have|we|he|she|they|my \w+)\s+(just\s+)?(took|taken|swallowed|ate|"
    r"injected|used)\b.*\b(too (many|much)|whole (box|pack|bottle)|all (of )?(the|my)|"
    r"\d{2,} (tablets|pills|capsules|doses|injections))",
    r"\b(my|our) (child|son|daughter|baby|kid|toddler)\b.*\b(swallowed|ate|took|got into)\b",
    r"\b(i|i've|ive|i have|he|she|they) (just )?(overdosed|od'd)\b",
    r"\b(can't|cant|cannot|can not|unable to|struggling to) breathe\b",
    r"\b(not breathing|unconscious|passed out|unresponsive|won't wake|wont wake)\b",
    r"\b(throat|tongue|lips|face) (is |are )?(swelling|swollen|closing)\b",
    r"\b(having|had|is having) (a )?(seizure|fit|anaphyla\w*)\b",
    r"\b(is |he's |she's )(seizing|convulsing)\b",
    r"\b(i have|i've got|ive got|having) (severe |bad |sudden )?chest pain\b",
    r"\b(kill|killing|hurt|hurting|harm|harming) (myself|my self)\b",
    r"\b(end|ending|take|taking) my (own )?life\b",
    r"\b(want|wanna|going|plan|planning) to (die|commit suicide|overdose)\b",
    r"\b(i feel|i am|i'm|im|feeling) suicidal\b",
    r"\b(thinking|thought) (about|of) (suicide|killing myself|ending it)\b",
    # Greek (accent-free, σ for ς)
    r"\bπηρα\b.*\b(πολλ\w*|ολο|ολα|ολοκληρο|ολη|\d{2,} (χαπια|δισκια|καψακια|δοσεισ))\b",
    r"\b(εκανα|πηρα|εχω κανει|εκανε|πηρε)\s+υπερδοσ\w*",
    r"\b(παιδι|μωρο|γιοσ|κορη) μου\b.*\b(εφαγε|καταπιε|ηπιε|πηρε)\b",
    r"\bδεν μπορω να (αναπνευσω|ανασανω)\b",
    r"\b(λαιμοσ|γλωσσα|προσωπο|χειλη) μου\b.*\b(πρηζ|πρηστ)\w*",
    r"\b(πρηζεται|πρηστηκε|πρηστηκαν) (ο λαιμοσ|η γλωσσα|το προσωπο|τα χειλη)\b",
    r"\b(λιποθυμησε|λιποθυμησα|ειναι αναισθητ\w*|δεν ξυπναει|δεν αναπνεει)\b",
    r"\b(κανει|εχει|επαθε|παθαινει) (κριση|σπασμουσ|επιληπτικ\w*|αναφυλαξια)\b",
    r"\bεχω (εντονο |δυνατο |ξαφνικο )?πονο στο στηθοσ\b",
    r"\bαυτοκτον(ησω|ησει|ω)\b",
    r"\bθελω να πεθανω\b",
    r"\b(σκοτωσω|βλαψω) τον εαυτο μου\b",
    r"\b(τελειωσω τη|δωσω τελοσ στη)ν? ζωη μου\b",
)]

# A narrow backstop: the user's own situation or a decision asked for them.
PERSONAL = [re.compile(pattern) for pattern in (
    r"\bshould i\b",
    r"\b(i'm|i am|im) (currently )?(on|taking|using|pregnant|breastfeeding|breast-feeding)\b",
    r"\bi (missed|forgot|skipped)\b",
    r"\bmy (dose|doctor|prescription|kidneys|liver)\b",
    r"\bis it (safe|ok|okay) for me\b",
    r"\b(να|μπορω να) (διπλασιασω|σταματησω|αυξησω|μειωσω|κοψω)\b",
    r"\b(παιρνω|ξεχασα|θηλαζω)\b",
    r"\bειμαι (εγκυοσ|σε θηλασμο)\b",
    r"\bμου (εγραψε|συνταγογραφησε)\b",
    r"\bγια μενα\b",
)]

SAFETY_PROMPT = """Classify a question sent to a medicine-information service. Reply \
with exactly one word: info, personal or emergency.
- info: a general question about what a medicine's official document says, including \
"Can X be taken with Y?" or "What are the side effects of X?".
- personal: the person describes their own situation (age, dose, symptoms, condition, \
pregnancy) or asks for a decision for themselves or someone they care for.
- emergency: overdose, a severe or allergic reaction happening now, breathing \
trouble, loss of consciousness, or any thought of suicide or self-harm.
If unsure, pick the safer label (emergency over personal over info)."""
LABEL_WORD = re.compile(r"\b(emergency|personal|info)\b")

MESSAGES: dict[Label, dict[str, str]] = {
    "personal": {
        "en": (
            "This sounds like a question about your own treatment. I can only explain what "
            "the official product information says, not what you should do. Please ask your "
            "doctor or pharmacist, who know your situation."
        ),
        "el": (
            "Αυτή φαίνεται να είναι ερώτηση για τη δική σας θεραπεία. Μπορώ μόνο να εξηγήσω "
            "τι λένε οι επίσημες πληροφορίες του προϊόντος, όχι τι πρέπει να κάνετε. "
            "Ρωτήστε τον γιατρό ή τον φαρμακοποιό σας, που γνωρίζουν την περίπτωσή σας."
        ),
    },
    "emergency": {
        "en": (
            "This may be an emergency. Call 112 now (EU emergency number, answered in Greek, "
            "English and French). In Greece you can also call 166 for an ambulance (EKAB) "
            "or the Poison Information Centre on 210 779 3777. If you are thinking about "
            "suicide or self-harm, call the 24-hour helpline 1018 (Klimaka)."
        ),
        "el": (
            "Μπορεί να πρόκειται για επείγον περιστατικό. Καλέστε τώρα το 112 (ευρωπαϊκός "
            "αριθμός έκτακτης ανάγκης). Στην Ελλάδα μπορείτε επίσης να καλέσετε το 166 "
            "(ΕΚΑΒ) ή το Κέντρο Δηλητηριάσεων στο 210 779 3777. Αν σκέφτεστε την αυτοκτονία "
            "ή να βλάψετε τον εαυτό σας, καλέστε την 24ωρη γραμμή 1018 (Κλίμακα)."
        ),
    },
}


def keyword_label(question: str) -> Label | None:
    """The keyword-rule label, or ``None`` when no rule matches."""
    text = fold(question)
    if any(pattern.search(text) for pattern in EMERGENCY):
        return "emergency"
    if any(pattern.search(text) for pattern in PERSONAL):
        return "personal"
    return None


async def classify(question: str, llm: Generate) -> tuple[Label, str]:
    """Return ``(label, source)``; source is ``keywords``, ``llm`` or ``fallback``."""
    rule = keyword_label(question)
    if rule == "emergency":
        return rule, "keywords"  # never wait on the LLM for an emergency
    messages = [
        {"role": "system", "content": SAFETY_PROMPT},
        {"role": "user", "content": question},
    ]
    # ponytail: one retry for a slow provider; add backoff if it keeps timing out.
    for attempt in range(2):
        try:
            reply = "".join([piece async for piece in llm(messages)])
            break
        except LLMError:
            if attempt == 0:
                continue
            # A matched personal rule is already a safe, fixed redirect.
            return ("personal", "keywords") if rule == "personal" else ("personal", "fallback")
    found = LABEL_WORD.search(reply.casefold())
    if found is None:
        return "personal", "fallback"
    label: Label = found.group(1)  # type: ignore[assignment]
    if rule is not None and SEVERITY[rule] > SEVERITY[label]:
        return rule, "keywords"
    return label, "llm"
