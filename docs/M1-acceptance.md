# M1 acceptance — download + parse for 5 medicines (EN + EL)

Status: **download and section-coverage checks pass locally (2026-10-15)** for all ten
language files below. Full M1 acceptance remains open: some grid-table rows lose their
frequency label. The table counts below show readable rows, not complete table fidelity.
Official licence terms are recorded; this report does not assert legal clearance.

## Corpus (5 medicines × 2 languages = 10 distinct PDFs)

All ten were resolved from EMA's nightly bulk report + each EPAR page, downloaded, and had
their detected language checked against the requested language (`verified` = detected ==
requested). `sha256` is pinned in `data/medicines.yaml`.

| Medicine | Lang | Revision | SHA-256 (12) | Bytes | SmPC sections | Leaflet sections | Readable 4.8 rows |
|----------|------|----------|--------------|-------|---------------|------------------|-------------------|
| eliquis  | en   | 2026-04-01 | 2c0830820630 | 1846165 | 28/28 | 6/6 | 36 |
| eliquis  | el   | 2026-04-01 | abc0d9cb77ba | 1878044 | 28/28 | 6/6 | 33 |
| humira   | en   | 2026-10-14 | 0118f9ea31f7 | 1742903 | 28/28 | 6/6 | 54 |
| humira   | el   | 2026-10-14 | 461e0a3562dc | 2037367 | 28/28 | 6/6 | 55 |
| jardiance| en   | 2026-03-24 | f37c74965875 | 1272518 | 28/28 | 6/6 | 4 |
| jardiance| el   | 2026-03-24 | 9e52657ac24a | 3221173 | 28/28 | 6/6 | 3 |
| ozempic  | en   | 2026-08-25 | 57219aa6be0c | 1400861 | 28/28 | 6/6 | 6 |
| ozempic  | el   | 2026-08-25 | 6a66cb6ff9fa | 1820238 | 28/28 | 6/6 | 6 |
| xarelto  | en   | 2026-09-22 | 296cd19bd9d5 | 2884512 | 28/28 | 6/6 | 29 |
| xarelto  | el   | 2026-09-22 | ae1220c9504c | 8342780 | 28/28 | 6/6 | 29 |

"SmPC sections 28/28" = every number in `SMPC_NUMBERS`; required subset
(`4.1–4.9`, `6.4`) present in all ten. English URLs under `/en/` redirect Greek files to
English, so every `el` URL is locale-prefixed `/el/` and the downloader verifies the
detected language from the PDF text.

## Reproducible commands

```bash
# Resolve URLs + revision dates, download all ten, pin SHA-256.
PYTHONPATH=src python -m medleaf.cli resolve Eliquis Humira Jardiance Ozempic Xarelto
PYTHONPATH=src python -m medleaf.cli download

# Section coverage for all ten (exit 1 if any required section is missing).
PYTHONPATH=src python -m medleaf.cli check

# Representative flattened 4.8 side-effect rows (exit 1 if any file has none).
PYTHONPATH=src python -m medleaf.cli tables --samples 2 --max-chars 110

# Tests.
PYTHONPATH=src python -m pytest -q                       # 19 passed, 12 skipped
MEDLEAF_M1_CORPUS=1 PYTHONPATH=src python -m pytest tests/test_m1_acceptance.py -q  # 15 passed
# Opt-in with missing assets fails (no green skip):
MEDLEAF_M1_CORPUS=1 MEDLEAF_DATA_DIR=.tmp/emptydata PYTHONPATH=src \
    python -m pytest tests/test_m1_acceptance.py::test_xarelto_en_frequency_associations -q
MEDLEAF_SMOKE_PDF=data/raw/xarelto/xarelto_el.pdf PYTHONPATH=src \
    python -m pytest tests/test_parser_perf_smoke.py -q -s  # 1 passed (4.5 s, 7 of 372 pages table-scanned)

# Lint the files changed this round.
python -m ruff check src/medleaf/config.py src/medleaf/ingest/ema.py \
    src/medleaf/ingest/chunk.py src/medleaf/ingest/pipeline.py \
    src/medleaf/cli.py tests/test_m1_acceptance.py   # All checks passed
```

## Readable side-effect tables (representative rows)

Rows are printed as `Reaction - frequency` (labelled tables) or grouped under the column
frequency (grid tables). Samples, truncated for display:

```
eliquis  en: 4.8 frequency rows=36
    Anaemia - Prevention of VTE in adult patients ... (VTEp): Common; Prevention of stroke ...: Uncommon
humira   el: 4.8 frequency rows=55
    Λοιμώξεις και παρασιτώσεις* - Συχνότητα: Πολύ συχνές; Ανεπιθύμητη Ενέργεια: Λοιμώξεις αναπνευστικού ...
jardiance el: 4.8 frequency rows=3
    Μεταβολικές και διατροφικές διαταραχές - Πολύ συχνές: Υπογλυκαιμία ...; Συχνές: Δίψα; Όχι συχνές: Κετοξέωση*
ozempic  el: 4.8 frequency rows=6
    Διαταραχές του ανοσοποιητικού συστήματος - Όχι συχνές: Υπερευαισθησία; Σπάνιες: Αναφυλακτική αντίδραση
xarelto  en: 4.8 frequency rows=29
    Anaemia (incl. respective laboratory parameters) - Common
    Thrombocytosis (incl. platelet count increased)A, thrombocytopenia - Uncommon
```

The **grid** shape (Xarelto-class: header row is `Common | Uncommon | Rare | ...`, no
reaction column) previously flattened the two cells of one row into the lie
`Anaemia - Thrombocytosis`. The flattener now reads the frequency from the column
position. Regression test: `test_grid_table_keeps_the_column_frequency`, backed by
`test_xarelto_en_frequency_associations` on the real PDF.

Deliberate mutation check: forcing `grid_form = False` makes
`test_grid_table_keeps_the_column_frequency` **fail**; restoring the code makes it pass.

## Independent review

The parent reviewer enabled both corpus and performance checks and ran the full suite:
31 tests passed, none skipped, in 32.47 seconds. The whole process took 38 seconds and
exited with code 0. Ruff passed on all six changed Python files.

The reviewer also reran `medleaf check` and `medleaf tables`: both exited with code 0,
with the same ten coverage and row-count results above. All ten local PDF hashes matched
the pinned manifest entries, and all ten hashes were distinct.

A missing-manifest control failed as expected with corpus checks enabled. The same
Xarelto frequency-association test then passed against the real assets. These checks do
not close the first-column frequency-loss gap described below.

## Official EMA legal source (recorded, not a legal clearance)

Source: EMA Legal notice — https://www.ema.europa.eu/en/legal-notice (retrieved 2026-10-15).
Relevant terms, with quoted excerpts:

- The contents of EMA webpages are © EMA [1995–2026]; EMA owns copyright and other IP rights
  for documents published on its website.
- Documents "may be reproduced and / or distributed ... for non-commercial and commercial
  purposes, provided that EMA is always acknowledged as the source of the material. Such
  acknowledgement must be included in each copy of the material."
- "Citations may be made from such material without prior permission, provided the source is
  always acknowledged."
- "The above-mentioned permissions do not apply to content supplied by third parties" —
  for third-party copyright, permission must be obtained from that rightsholder.
- "The above-mentioned permissions also do not apply to photos or other materials that are
  not under the copyright of EMA."
- The EMA logo is the exclusive property of EMA; use is prohibited without prior written
  permission.

Consequences recorded honestly: this repository's **code** is licensed separately (SPEC says
MIT); the **corpus PDFs and extracted text are not MIT** — they are EMA copyright under the
notice above and every copy must acknowledge EMA as the source. The project has **not**
obtained legal review or a written permission, and no uncertain permission is treated as
cleared. `data/medicines.yaml` carries the EMA attribution header; the README must keep it.

## Limitations / open gaps

1. **Table extraction quality.** PDF text extraction splits some words across line breaks
   (`Hypersensiti vity`, `Δυσγευσία` fragments). Sample associations and regression tests
   pass, but the full extraction has not been checked for clinical accuracy.
2. **Grid-table heuristic.** A reaction that appears only in the first frequency column of a
   grid table is emitted as a bare heading line, because PyMuPDF's `extract()` exposes no
   font weight to tell a system-organ-class heading from a Common-only reaction. No wrong
   association is produced, but the frequency is dropped on such rows.
3. **Eliquis 4.8 layout.** Uses a multi-indication labelled table; rows are several hundred
   characters and are truncated in the `tables` display (`--max-chars`), not in the stored
   chunk text.
4. **EMA bulk report drift.** The `Content type` header is now around row 281 (not within the
   first 50) and reads `Content type:` — the resolver was fixed to scan all rows and accept
   the colon. This is why `resolve` needed a change this round.
5. **Scope.** Only download/parse/coverage/tables tooling was touched. No search, embeddings,
   API, or deployment work. Pre-existing whole-repo lint/type issues elsewhere were not
   refactored; changed files are ruff-clean, `mypy --strict` was not run.
6. **Duplicate manifest model.** `src/medleaf/corpus.py` is an unused second manifest model
   (`sources`/`ema_url`) distinct from the live `models.py` + `ingest/store.py`
   (`documents`/`epar_url`). Left untouched to keep the blast radius minimal; flagged for a
   later cleanup.
