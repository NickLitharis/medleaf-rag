# MedLeaf RAG

Bilingual (Greek / English) question answering over official EU medicine
information, with cited answers and medical safety guardrails. The project is at
milestone **M1**: it resolves, downloads and parses the product-information PDFs.
Retrieval, the API and evaluation are later milestones; see `SPEC.md`.

## Status

- **M1 — download + parse for five medicines (EN + EL):** acceptance evidence in
  `docs/M1-acceptance.md`, milestone ledger in `docs/milestones.json`.
- **M2–M7:** not started.

## Setup

Python 3.11 or later.

```bash
python -m pip install -e ".[dev]"
```

## Checks

```bash
PYTHONPATH=src python -m pytest -q          # fast tests; corpus checks stay skipped
python -m ruff check .
```

The ten real EMA PDFs are not committed (`data/raw/` is ignored). To run the
corpus checks, download them first and then opt in:

```bash
PYTHONPATH=src python -m medleaf.cli resolve Eliquis Humira Jardiance Ozempic Xarelto
PYTHONPATH=src python -m medleaf.cli download
MEDLEAF_M1_CORPUS=1 PYTHONPATH=src python -m pytest tests/test_m1_acceptance.py -q
```

## Medical disclaimer

This project is not medical advice and is not a substitute for a doctor or
pharmacist. Do not use it to decide on a dose, to stop or change a medicine, or in
an emergency. In an emergency in the EU, call 112.

## Licence and attribution

The code licence and the corpus licence are different. The medicine-information
PDFs and the text extracted from them are **copyright of the European Medicines
Agency (EMA)** and are not covered by any code licence; every copy must
acknowledge EMA as the source. The recorded EMA legal terms are in
`docs/M1-acceptance.md`. This repository claims no legal or clinical clearance.
