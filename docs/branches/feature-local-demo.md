# Local demo

## Scope

Reuse the existing browser page, hybrid index and streamed answering API. Keep
fine-tuning, new infrastructure and broader release work outside this change.

## Changes

- Add sample questions and a persistent medical warning to the page.
- Describe the status badge as index readiness, not provider readiness.
- Prevent concurrent Search and Ask requests from replacing each other's output.
- Report failed safety classification as HTTP 503 with `safety_unavailable`.
  Do not present a provider outage as a personal-treatment refusal.
- Add regression tests for a failed provider and an invalid classifier response.
- Normalize typographic whitespace in citation keys. A real model used a narrow
  space in `Eliquis Leaflet 1`, and the filter incorrectly dropped the valid key.
  The filter still rejects source keys that are absent from the retrieved evidence.
- Update README status, startup instructions and known answer-quality limits.

## Evidence

- The two new regression cases failed against the old API with HTTP 200 instead of 503.
- Three Unicode-space regression cases failed before the citation fix. API tests
  also cover a streamed citation with a narrow space and its final refusal flag.
- Initial local-demo suite: 127 passed, 32 skipped. The skipped cases require explicit
  corpus, real-Qdrant or PDF performance settings.
- Ruff and `git diff --check`: all checks passed.
- Headless Edge rendered the index status, sample questions, factual-error warning
  and populated medicines dropdown. This is a page-load check, not a full browser
  interaction suite.
- Existing live Qdrant index: 8,797 chunks across 40 medicines.
- Live NVIDIA `openai/gpt-oss-20b`: English Eliquis answer with EMA citation,
  17.4 seconds. Greek Ozempic answer with Greek EMA citations, 25.1 seconds.
- Personal-treatment question: redirected, no citations, `refused=true`.
- Emergency example: redirected without generation, `refused=true`.
- Unsupported common-cold claim: no citations, `refused=true`.
- Final rebuilt container: citation fix present in the installed package, index
  ready with 8,797 chunks, and live Eliquis Ask completed with one EMA citation
  and `refused=false`. The first request after rebuilding took 70 seconds.
- These requests are smoke tests, not an answer-quality benchmark.

## Runtime configuration

The user returned the local API to NVIDIA NIM: `https://integrate.api.nvidia.com/v1`
with `openai/gpt-oss-20b` and the existing NVIDIA credential. The container was
recreated without rebuilding. A direct provider probe returned HTTP 200 and `OK`
in 1.19 seconds; local index readiness reported 8,797 points. This is a provider
connectivity check, not successful RAG answer acceptance. README examples now
show these NVIDIA overrides. No credential was written to the repository.

OpenCode fallback is deferred at the user's request. Screening twelve active,
explicitly zero-priced models through their direct compatible API produced only
HTTP 403 `FreeTierError`: free-tier use requires the OpenCode client. There is
no successful fastest-model ranking and no fallback installed. The exploratory
benchmark script and report were moved to temporary diagnostics, not retained
as unfinished repository features. No client headers were spoofed.

Earlier, the local API used `https://openrouter.ai/api/v1` with `openrouter/free`
and the existing OpenRouter credential. At that point, the container was
recreated without rebuilding the image; the numerical safeguard remains installed.
Model and base URL were checked through Docker inspection. Two direct provider
probes returned HTTP 429, so successful OpenRouter generation is not verified.
Those checks found 8,797 indexed points. README startup examples temporarily
used OpenRouter. No credential was added to repository files.

The numerical evaluation results below were collected earlier with NVIDIA
`openai/gpt-oss-20b`; they are not OpenRouter results. `openrouter/free` may choose
different underlying models, so it does not pin a reproducible evaluation model.
Recreating the container requires the provider environment variables shown in
README. Restarting the current container preserves its configuration.

Further diagnosis on 2026-10-16 confirmed the account's daily free-model quota,
not a local configuration error. Key metadata was reachable, but completion
returned `openrouter_free_tier_daily`: limit 50, remaining 0. The provider's
`X-RateLimit-Reset` value was `1792195200000`: 2026-10-17 00:00 UTC, or 03:00
in Athens. Do not retry OpenRouter completions before this reset. No credits were purchased,
no account was changed, and no alternate model was selected.

These quota findings apply to the earlier OpenRouter configuration, not the
restored NVIDIA runtime. Live quote-selection work and bilingual acceptance
remain unfinished. Avoid speculative prompt changes without real-provider evidence.
Rechecked offline: 144 passed, 38 skipped; Ruff passed. The API index was ready
with 8,797 points. This does not establish successful OpenRouter answering.

## Numerical-faithfulness safeguard

The Greek smoke answer reversed the dose associations for two percentages from
the Ozempic source. The retrieved sentence was correct; citation filtering did
not prevent the numerical paraphrase from reversing its meaning.

- Add `eval/answer-faithfulness.json`: attributed English and Greek EMA excerpts,
  with general, numerical and misleading questions. The source associates nausea
  with 17% at the lower dose and 19.9% at the higher dose, not the reverse.
- Add a narrow fixed-context checker. It rejects the swapped values even when a
  correct quotation is also present. Explicit numerical questions cannot pass
  by returning an empty answer or omitting the requested source sentence.
- An initial prompt-only mitigation failed the real-provider quotation checks.
  Replace it with `[quote:N]` references: the server inserts original retrieved
  lines and citations, instead of letting the model rewrite their figures.
- Buffer model output before displaying it. Raw generated digits and unknown
  numerical quote references produce `numerical_grounding_failed`; the API
  emits neither answer tokens nor a successful `done` event for that response.
- Preserve existing safety redirects, citation filtering and SSE event names.
  Generation failures now expose no partial answer either. The page explains
  that it is preparing the answer and checking numerical source quotations.
- This is deliberately conservative: even a numbered list can be withheld.
  Source relevance, qualitative claims, spelled-out numbers and extraction
  errors are not validated. No new provider, dependency or infrastructure is added.

Verification:

- Latest offline suite: 144 passed, 38 skipped. Six skips are the opt-in live
  numerical cases; the previous corpus/Qdrant/performance skips remain.
- Ruff and `git diff --check` pass.
- Prompt-contract tests failed before the change. Temporarily disabling the raw
  digit guard made two API no-leak tests fail. Restoring it made all three pass,
  including the unknown-reference case. No mutation remains in the working tree.
- Live fixed-context run on NVIDIA `openai/gpt-oss-20b`: three passed and three
  timed out at the sixty-second per-case deadline. Total elapsed: 224.73 seconds.
  The English general and misleading questions, plus the Greek general question,
  passed. The two direct numerical questions and Greek misleading question timed out.
  All completed answers retained the expected nausea sentence. The English misleading
  question was explicitly rejected, with the unchanged source quotation as evidence.
- Before the source-reference approach, the original prompt met the new quotation
  contract in one of six cases; a prompt-only verbatim instruction met it in none
  of six. Those counts measure this quotation contract, not semantic answer accuracy.
- Full-retrieval NVIDIA API smoke: the original Greek Ozempic question and
  English Eliquis-use question produced `numerical_grounding_failed`, with zero
  answer tokens and no `done` event. This demonstrates fail-closed display, not
  successful question answering. Useful answer completion remains unresolved.
- Browser JavaScript syntax check: passed with `node --check`.
- The rebuilt API container's installed generation and API files have the same
  SHA-256 hashes as the working tree. `/readyz` reports 8,797 indexed points, and
  the served page contains the updated numerical-check progress message.

## NIM reliability follow-up

The restored default-effort model passed three of six live quote cases in
203.02 seconds. English numerical lookup timed out; both misleading questions
were withheld for generated digits. The guard worked, but these are not useful
answers. The other three cases passed.

Headless Edge exercised the actual Ask button and rendered page. English cited
answer: 106.92 seconds, including retrieval. Greek answer: safely withheld after
50.67 seconds. Personal-treatment redirect: 2.04 seconds. Emergency redirect:
0.53 seconds. This is partial browser acceptance, not a complete pass.

Two active, explicitly zero-priced NIM alternatives were screened on the English
and Greek misleading questions: `nvidia/nemotron-3.5-lightning-30b-a3b` and
`z-ai/glm-5.3-flash`. Both hit the thirty-second network timeout on both requests.
Neither replaced the runtime model.

NIM rejected an invalid `reasoning_effort` with HTTP 400 and accepted `low` with
HTTP 200. However, low effort failed all six quote cases in 24.82 seconds: every
answer contained generated figures outside source references. The experimental
setting and its tests were removed. Faster invalid answers are not an improvement.

NIM's native `response_format` JSON schema passed a forced-enum probe and rejected
an invalid format with HTTP 400. Its `nvext.guided_regex` field was ignored, so no
regex-based protection was installed.

The opt-in `MEDLEAF_LLM_SOURCE_QUOTES_ONLY=true` mode now restricts generation to
at most five source-line IDs. The server checks the complete JSON shape, types,
count and IDs, then renders the original lines through the existing numerical
and citation checks. Empty selection produces a fixed localized abstention.
Qualitative source lines can also be selected. Safety classification keeps its
original plain-label protocol. Other providers keep their default behavior.

This is extractive answering, not a generated medical explanation. It deliberately
reduces capability to avoid unverified clinical prose. It does not verify source
relevance, source extraction or clinical correctness.

The same six live quote assertions passed in 48.04 seconds in native selection
mode. The assertion still requires the original nausea sentence for numerical
and misleading questions. No source figures were reconstructed. Temporarily
removing local ID/type validation made two regression cases fail; restoring it
made all nine adversarial parser checks pass. No mutation remains.
The full offline suite passed 163 tests, with 38 skipped and one warning.
Ruff and `git diff --check` passed. A separate API wiring test checks that native
schema output applies to answers, not safety classification.

The first full-retrieval browser run completed eight paths, but strict sentence
and citation checks found only six useful results. English general, numerical
and misleading questions worked, as did Greek general information and both
safety redirects. Greek numerical lookup selected an unrelated overdose heading;
Greek misleading lookup returned different gastrointestinal outcomes, not the
requested nausea rates. The expected Greek chunk was absent from a repeat search's
top twenty candidates. These are not successful numerical answers.

The retained conservative selection prompt made the three Greek browser
questions abstain, including the general question that had worked before.
This is a known false-abstention cost, not successful multilingual acceptance.
A broader candidate with explicit outcome/dose/population instructions was
rejected: seven of eight live cases passed in 70.87 seconds, but the English
no-evidence case selected hypoglycaemia and discontinuation figures for a
nausea-rate question. The figures were copied correctly; the selection was wrong.
The conservative prompt was restored. No retrieval ranking, corpus or numerical
validation was changed.

Two opt-in live abstention tests now remove the nausea-rate sentence while
keeping the attributed hypoglycaemia and discontinuation distractors. English
abstained; Greek hit the sixty-second test deadline, then abstained in 10.01
seconds on one bounded retry. This was not a clean two-case pass.

The retained conservative prompt passed six quote-preservation cases in
42.82 seconds. Its eight-request browser run returned three English cited answers,
two safety redirects and three Greek abstentions. Strict checks count only five
of eight paths as useful. Timings: English general 50.17 seconds after a cold
start; English numerical 5.64; English misleading 14.89; Greek general/numerical/
misleading 16.91/8.20/7.71; personal 1.04; emergency 0.52 seconds.

All eight browser requests completed; no answer was shown during generation.
Abstentions are not counted as answered questions. The earlier cold-search
warm-up took 42.17 seconds and returned four hits. Container source hashes matched
local configuration, generation and API files. After restoring the conservative
runtime, all three installed source hashes matched again. Readiness reported
8,797 chunks. A fresh full-retrieval English numerical Ask completed in 49.39
seconds after a cold start and preserved the complete expected nausea sentence.
The rejected broader prompt is not running.

After adding the two opt-in abstention cases, the offline suite passed 163 tests,
with 40 skipped and one warning. Final verification after restoring the retained
prompt: 163 passed, 40 skipped, one warning in 14.99 seconds. Ruff and
`git diff --check` passed.

## Follow-up

The optional live quality gate and full browser acceptance are not green.
Investigate Greek retrieval misses and false abstentions before claiming a useful
bilingual demo. Evaluate retrieval changes against the existing development gate;
do not tune on the held-out test split or widen the corpus to hide this failure.
Provider latency and broader answer-quality evaluation also remain open.
The numerical safeguard is not a clinical fact checker. The README retains the
observed failure and these limitations instead of claiming medical reliability.

M3 still needs evidence of a deliberately broken retrieval PR failing the hosted
CI gate. No branch was pushed and no release was published. The pre-existing
untracked `docs/M2-verification-prompt.md` was left unchanged.
