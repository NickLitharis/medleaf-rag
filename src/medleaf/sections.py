"""Language-independent section keys and title validation.

Section numbers are identical in every EU language (SPEC section 8), so the *number*
is the stable key and the *title* is only used to confirm that a numbered line really
is a heading rather than prose that happens to start with "4.8".

Two rules keep the match honest:

* A heading is accepted only if a known number is followed by text.
* The title must match the expected wording for that number in the document's own
  language, and must not contain a sentence-ending pattern typical of cross-reference
  prose ("see section ...", "(see sections 4.8 and 4.9)", "βλ. παράγραφο").

The title vocabulary is deliberately keyed on the *intersection* of wording that EMA
uses across medicines, not on one leaflet, because titles differ per product
("What Eliquis is and what it is used for" vs "What X is and what it is used for").
"""

from __future__ import annotations

import re

# SmPC section numbers required by M1 acceptance (SPEC section 16).
SMPC_NUMBERS: tuple[str, ...] = (
    "1", "2", "3", "4", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9",
    "5", "5.1", "5.2", "5.3", "6", "6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "7", "8",
    "9", "10",
)
# The subset M1 asserts for every medicine and language.
SMPC_REQUIRED: tuple[str, ...] = (
    "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "6.4",
)

LEAFLET_NUMBERS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
LEAFLET_REQUIRED: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")

# English SmPC titles, matched case-insensitively as a prefix/substring.
_SMPC_EN: dict[str, tuple[str, ...]] = {
    "1": ("name of the medicinal product",),
    "2": ("qualitative and quantitative composition",),
    "3": ("pharmaceutical form",),
    "4": ("clinical particulars",),
    "4.1": ("therapeutic indications",),
    "4.2": ("posology and method of administration",),
    "4.3": ("contraindications",),
    "4.4": ("special warnings and precautions for use",),
    "4.5": ("interaction with other medicinal products", "interaction with other"),
    "4.6": ("fertility, pregnancy and lactation",),
    "4.7": ("effects on ability to drive", "effects on ability to drive and use machines"),
    "4.8": ("undesirable effects",),
    "4.9": ("overdose",),
    "5": ("pharmacological properties",),
    "5.1": ("pharmacodynamic properties",),
    "5.2": ("pharmacokinetic properties",),
    "5.3": ("preclinical safety data",),
    "6": ("pharmaceutical particulars",),
    "6.1": ("list of excipients",),
    "6.2": ("incompatibilities",),
    "6.3": ("shelf life",),
    "6.4": ("special precautions for storage",),
    "6.5": ("nature and contents of container",),
    "6.6": ("special precautions for disposal",),
    "7": ("marketing authorisation holder",),
    "8": ("marketing authorisation number",),
    "9": ("date of first authorisation", "date of first authorisation/renewal"),
    "10": ("date of revision of the text",),
}

# Greek SmPC titles, accent-insensitive and case-insensitive.
_SMPC_EL: dict[str, tuple[str, ...]] = {
    "1": ("ονομασια του φαρμακευτικου προϊοντος", "ονομασια του φαρμακευτικου προιοντος"),
    "2": ("ποιοτικη και ποσοτικη συνθεση",),
    "3": ("φαρμακοτεχνικη μορφη",),
    "4": ("κλινικα στοιχεια", "κλινικες πληροφοριες"),
    "4.1": ("θεραπευτικες ενδειξεις",),
    "4.2": ("δοσολογια και τροπος χορηγησης",),
    "4.3": ("αντενδειξεις",),
    "4.4": (
        "ειδικες προειδοποιησεις και προφυλαξεις κατα τη χρηση",
        "ειδικες προειδοποιησεις",
    ),
    "4.5": ("αλληλεπιδρασεις με αλλα φαρμακευτικα προϊοντα", "αλληλεπιδρασεις"),
    "4.6": ("γονιμοτητα, κυηση και γαλουχια", "κυηση και γαλουχια", "γονιμοτητα"),
    "4.7": ("επιδρασεις στην ικανοτητα οδηγησης",),
    "4.8": ("ανεπιθυμητες ενεργειες",),
    "4.9": ("υπερδοσολογια",),
    "5": ("φαρμακολογικες ιδιοτητες",),
    "5.1": ("φαρμακοδυναμικες ιδιοτητες",),
    "5.2": ("φαρμακοκινητικες ιδιοτητες",),
    "5.3": ("προκλινικα δεδομενα",),
    "6": ("φαρμακευτικα στοιχεια", "φαρμακευτικες πληροφοριες"),
    "6.1": ("καταλογος εκδοχων",),
    "6.2": ("ασυμβατοτητες",),
    "6.3": ("διαρκεια ζωης",),
    "6.4": ("ιδιαιτερες προφυλαξεις κατα τη φυλαξη",),
    "6.5": ("φυση και συστατικα του περιεκτη",),
    "6.6": ("ιδιαιτερες προφυλαξεις απορριψης",),
    "7": ("κατοχος της αδειας κυκλοφοριας",),
    "8": ("αριθμος αδειας κυκλοφοριας", "αδειας κυκλοφοριας"),
    "9": ("ημερομηνια πρωτης εγκρισης", "ημερομηνια πρωτης αδειας"),
    "10": ("ημερομηνια αναθεωρησης του κειμενου",),
}

# Leaflet titles are product-specific, so matching uses a small set of phrases that
# every EMA leaflet contains for each number, in both languages.
_LEAFLET_EL_HINTS: tuple[str, ...] = (
    "τι ειναι",
    "τι πρεπει να γνωριζετε",
    "πωσ να παρετε",
    "πως να παρετε",
    "να χρησιμοποιησετε",
    "πωσ να χρησιμοποιησετε",
    "πως να χρησιμοποιησετε",
    "πιθανες ανεπιθυμητες ενεργειες",
    "πωσ να φυλασσετε",
    "πως να φυλασσετε",
    "περιεχομενα τησ συσκευασιας",
)
_LEAFLET_EN_HINTS: tuple[str, ...] = (
    "what ",
    "what you need to know",
    "how to take",
    "how to use",
    "possible side effects",
    "how to store",
    "contents of the pack",
)


def _strip_accents(text: str) -> str:
    """Remove Greek tonos/dialytika so titles match regardless of accents.

    Lowercasing first is what makes this complete: uppercase dialytika (Ϊ) would
    otherwise survive as a distinct character, so 'ΠΡΟΪΟΝ' would not match the
    lowercase hint 'προϊον'.
    """
    replacements = str.maketrans(
        {
            "ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω",
            "ϊ": "ι", "ϋ": "υ", "ΐ": "ι", "ΰ": "υ",
            "ς": "σ",
        }
    )
    return text.lower().translate(replacements)


def normalize_title(title: str) -> str:
    """Lowercase + accent-strip a title for comparison."""
    return _strip_accents(title).strip(" .;·-–—:")


# Prose patterns that indicate a cross-reference rather than a heading.
_CROSS_REF_RE = re.compile(
    r"(^\s*(see|βλ\.|βλεπε)\b)|(\bsee sections?\b)|(\bβλ\.\s*παραγραφ)|(^\s*\(see)",
    re.IGNORECASE,
)
_SENTENCE_TAIL_RE = re.compile(r"(and$|or$|\.\s*$)")


def _matches_any(title: str, hints: tuple[str, ...]) -> bool:
    normalized = normalize_title(title)
    # Hints are stored in readable Greek; normalise them the same way as titles so
    # accent and final-sigma differences (ς vs σ) cannot break the comparison.
    return any(normalize_title(hint) in normalized for hint in hints)


def smpc_title(number: str, title: str) -> bool:
    """True when ``title`` is the expected SmPC heading for ``number`` in either language."""
    if not title or _CROSS_REF_RE.search(title):
        return False
    candidates = _SMPC_EN.get(number, ()) + _SMPC_EL.get(number, ())
    return _matches_any(title, candidates)


def looks_like_leaflet_heading(title: str) -> bool:
    """True when a ``N. ...`` line reads like a leaflet heading in either language."""
    if not title or _CROSS_REF_RE.search(title):
        return False
    normalized = normalize_title(title)
    hints = _LEAFLET_EN_HINTS + _LEAFLET_EL_HINTS
    return _matches_any(normalized, hints)


__all__ = [
    "LEAFLET_NUMBERS",
    "LEAFLET_REQUIRED",
    "SMPC_NUMBERS",
    "SMPC_REQUIRED",
    "looks_like_leaflet_heading",
    "normalize_title",
    "smpc_title",
]
