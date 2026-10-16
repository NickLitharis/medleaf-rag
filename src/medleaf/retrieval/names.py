"""Find the medicines a question names, to filter the search (SPEC §9 step 2).

Brand names and English substances come from the index manifest. Greek substance names
live here, not in ``medicines.yaml``: that file's hash is the corpus version, so editing
it would invalidate the index. They are stems, because Greek inflects the ending
(``τικαγρελόρη``, ``τικαγρελόρης``), taken from each Greek SmPC section 2. Medicines whose
Greek documents write the substance in Latin script need no entry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from medleaf.safety.classifier import fold

GREEK_SUBSTANCES: dict[str, tuple[str, ...]] = {
    "aimovig": ("ερενουμαμπ",),
    "brilique": ("τικαγρελορ",),
    "eliquis": ("απιξαμπαν",),
    "enbrel": ("ετανερσεπτ",),
    "entresto": ("σακουμπιτριλ", "βαλσαρταν"),
    "eylea": ("αφλιβερσεπτ",),
    "fasenra": ("μπενραλιζουμαμπ",),
    "forxiga": ("δαπαγλιφλοζιν",),
    "herceptin": ("τραστουζουμαμπ",),
    "humira": ("αδαλιμουμαμπ",),
    "imfinzi": ("δουρβαλουμαμπ",),
    "jardiance": ("εμπαγλιφλοζιν",),
    "keytruda": ("πεμπρολιζουμαμπ",),
    "lixiana": ("εδοξαμπαν",),
    "lyrica": ("πρεγκαμπαλιν",),
    "mounjaro": ("τιρζεπατιδ",),
    "nubeqa": ("δαρολουταμιδ",),
    "ocrevus": ("οκρελιζουμαμπ",),
    "ozempic": ("σεμαγλουτιδ",),
    "praluent": ("αλιροκουμαμπ",),
    "rinvoq": ("ουπαδασιτινιμπ",),
    "rybelsus": ("σεμαγλουτιδ",),
    "soliris": ("εκουλιζουμαμπ",),
    "tagrisso": ("οσιμερτινιμπ",),
    "tecfidera": ("φουμαρικου διμεθυλεστερ",),
    "trulicity": ("ντουλαγλουτιδ",),
    "tysabri": ("ναταλιζουμαμπ",),
    "victoza": ("λιραγλουτιδ",),
    "xarelto": ("ριβαροξαμπαν",),
}


def medicine_names(
    medicines: list[str], active_substances: Mapping[str, str | None]
) -> dict[str, list[str]]:
    """Folded names per indexed slug: the brand (the slug), English and Greek substances."""
    names: dict[str, list[str]] = {}
    for slug in medicines:
        substances = (active_substances.get(slug) or "").split(";")
        found = [slug, *substances, *GREEK_SUBSTANCES.get(slug, ())]
        names[slug] = [fold(name.strip()) for name in found if name.strip()]
    return names


def detect_medicines(question: str, names: Mapping[str, list[str]]) -> list[str]:
    """Slugs whose names start a word in the question, accent- and case-insensitive."""
    text = fold(question)
    return [
        slug
        for slug, forms in names.items()
        if any(re.search(rf"\b{re.escape(form)}", text) for form in forms)
    ]
