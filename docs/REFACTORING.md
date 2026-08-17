# Refactoring log

Production hardening of the multi-agent support system.
Tooling: ruff 0.x, mypy 1.x, Python 3.12.

## Baseline (before refactoring)

Measured on commit `<вставь хеш>`, branch `refactor/tooling`.

### ruff — 126 issues

| Rule | Count | Description |
|---|---|---|
| T201 | 49 | print statements instead of logging |
| W292 | 45 | missing newline at end of file |
| ANN201 | 14 | missing return type annotation |
| I001 | 6 | unsorted imports |
| ANN001 | 3 | missing argument type annotation |
| RET503 | 3 | implicit return None |
| W293 | 3 | blank line with whitespace |
| ARG001 | 1 | unused function argument |
| B905 | 1 | zip() without explicit strict= |
| E501 | 1 | line too long |

### mypy — 33 errors in 9 files

| Area | Count | Nature |
|---|---|---|
| `config.py:42` | ~20 | untyped `**kwargs` passed to ChatAnthropic |
| `mcp_server/wp_server.py`, `wp_client.py` | 8 | unguarded `dict \| list` union from WP REST API |
| `agents/router.py`, `agents/bug_investigator.py` | 2 | `Any` returned where a Pydantic model is declared |
| `rag/retriever.py:27` | 1 | possible attribute typo (`float` vs `float_`) |
| `graph.py:42` | 1 | optional state key passed where `str` expected |

### Other

| Metric | Value |
|---|---|
| Logger instances | 0 |
| Points reading environment variables | 11 |
| Retry-wrapped external calls | 1 (of 4 dependencies) |
| Python modules | 28 |

## Findings

### Silent provider failure (found by ruff RET503)

`get_llm()` had no branch for the Gemini provider: with `LLM_PROVIDER=gemini`
— the default value in `.env.example` — the function fell through and returned
`None`, failing later with `AttributeError` on the first `.invoke()` call.
Static analysis surfaced it as an implicit-return warning before any test did.

Fixed in `fix: restore Gemini branch in get_llm()`. Development now runs on the
free tier; the paid provider is reserved for final metric runs.

## Progress

### Step 3 — configuration

| Metric | Before | After |
|---|---|---|
| mypy errors | 33 | 13 |
| `os.getenv` call sites | 11 | 8 |
| Hardcoded business parameters | 4 | 0 |

Introduced `src/settings.py`: a single `pydantic-settings` model validated at
import time. API keys are `SecretStr`, so they render as `**********` in logs
and tracebacks. A `model_validator` fails fast when the selected provider has
no credentials, instead of raising from inside a third-party validator on the
first model call.

Rewriting `get_llm()` with explicit keyword arguments — rather than
`**dict[str, object]` — removed 20 mypy errors from a single line.

| Metric | Before | After |
|---|---|---|
| mypy errors | 33 | 8 |
| ruff issues | 126 | 120 |
| `os.getenv` call sites | 11 | 3 (dev scripts only) |
| Hardcoded business parameters | 4 | 0 |
| Dead modules | 1 | 0 |

Also removed `src/wp_server.py`, an orphaned duplicate of
`src/mcp_server/wp_server.py` left behind when the module moved into a package;
nothing imported it.

Two latent bugs surfaced while migrating: `get_llm()` returning `None` for the
default provider, and an unguarded index into Cohere's `embeddings.float_`,
which the SDK types as optional. Both were fixed rather than silenced.

### Provider-dependent token budgets

With `LLM_PROVIDER=gemini`, the code generator returned 303-character answers —
truncated mid-sentence before any PHP appeared. `finish_reason` was `MAX_TOKENS`
with 1171 of 1996 output tokens spent on reasoning, leaving ~825 for the visible
answer.

Gemini counts reasoning tokens against `max_output_tokens`; Anthropic does not.
The limit was raised to 4000 in `Settings`, which is generous for Gemini and
inert for Anthropic. The bug was invisible until the Gemini branch was restored,
because all previous development had run on the paid provider.

### Step 4 — logging

| Metric | Before | After |
|---|---|---|
| ruff issues | 126 | 0 |
| mypy errors | 33 | 4 |
| `print()` in application code | 49 | 0 |
| Modules with a logger | 0 | 12 |
| Broken imports in `tests/` | 4 | 0 |

Introduced `src/logging_config.py`: a single setup point, `stderr` output so
CLI answers stay pipeable on `stdout`, third-party loggers silenced at WARNING,
and a `LoggerAdapter` that binds LangGraph's `thread_id` to every message for
correlated output.

Two bugs surfaced. The graph was compiled twice — once in `graph.py`, once in
`runner.py` — leaving two independent `InMemorySaver` instances, so a run
suspended on `interrupt()` could resume against a checkpointer that never saw
it. And

### Confidence calibration on short non-English messages

The router assigns high confidence to messages carrying no diagnostic
information at all, provided they are not in English. `"не работает"` scores
0.90 as `bug` on Gemini; the same class of message scored 0.75 on Anthropic in
the v1 metrics. The 0.6 escalation threshold is therefore bypassed exactly
where it matters most.

Reproduced on both providers, so it is a prompt issue rather than a
model-specific quirk. Deferred to the LangSmith stage (steps 7-8): the fix is a
prompt change, and a prompt change without a dataset to measure it against is a
guess.

### Console encoding on Windows

PowerShell mangles Cyrillic when piping a heredoc into `python -`. Set
`$env:PYTHONIOENCODING = "utf-8"` and a BOM-less `[Console]::OutputEncoding`
before any manual check involving non-English input, or the model will be
scored on corrupted text.

### Retry behaviour under a real 429

The first live failure was not simulated. Gemini's free tier returned
`RESOURCE_EXHAUSTED` mid-development, and the new machinery behaved as designed:
the provider error was translated into `LLMRateLimitError`, `with_retry` made
three attempts with growing back-off and logged each one, and the caller
received a domain exception instead of a provider-specific traceback.

It also exposed a configuration flaw. The free-tier quota is
`PerProjectPerModel`, and both tiers pointed at `gemini-2.5-flash` — so the
router and the three heavier agents shared a single 20-request daily budget.
Splitting `fast` onto `gemini-2.5-flash-lite` doubled the development budget and
matched the architecture's intent, where a cheap model does the triage.

### `Event loop is closed` on the second investigation round

`run_diagnostics` drove the ReAct agent through `asyncio.run`, which closes its
loop on exit. The Anthropic and MCP SDKs cache HTTP connections bound to the
loop that created them, so the second round landed on a closed loop and failed
as `APIConnectionError: Connection error` — an error message pointing at the
network rather than at the real cause.

The defect predates this refactor but was invisible: every per-node test and the
v1 latency benchmark measured the investigator up to the *first* clarifying
question, and a fresh process meant a fresh loop each time. It only surfaces on
a resumed run, which is the agent's entire reason to exist.

Fixed with `src/async_bridge.py`: one background event loop per process, started
on first use and never torn down, with `run_sync()` submitting coroutines to it
from synchronous code.
## Step 8 — Retrieval miss: diagnosis

**Symptom.** "How do I add a hidden field that stores the current user ID?"
returns five pages about user roles and accounts. The answer is correct, the
sources are not.

**Baseline** (`crocoblock-retrieval`, 12 examples): retrieval_hit 0.83,
retrieval_rank 0.53. Two cases score zero: `hidden-field` and `media-field`.

**Not a corpus gap.** Both pages are indexed (247 pages, 1836 chunks in
namespace `jfb`); `hidden-field` contributes 8 chunks.

**Not dense search.** Pinecone returns the correct page at candidate
position 2, with four of its chunks inside the top 20.

**Reranking drops it.** Cohere rerank-v3.5 promotes three user-role pages
(0.81, 0.81, 0.78) and removes every `hidden-field` chunk from the top 5.

**Prepending the page title does not help.** Reranking `title + text`
instead of `text` returns the same five pages, merely reordered.

**Root cause: chunk boundaries.** The chunk holding the literal answer
("Current User ID ㅡ an option that assigns the ID of the user that is
currently logged in") is the middle of a bulleted list. It opens mid-item,
ends mid-word, and never names the feature it describes. Judged in isolation
it reads as an unlabelled list of properties, while a user-role page reads as
coherent prose about users. The reranker behaves reasonably on the input it
is given.

**Not fixed here.** The fix is re-chunking at indexing time — carrying
section headings into every chunk and avoiding splits inside lists — which
means rebuilding the index and falls outside the scope of this refactor. The
low `retrieval_rank` on half the passing cases is likely the same mechanism.

## Step 8 — Router calibration differs by provider

The router assigns a confidence score and the graph escalates to a human below
0.6. The `crocoblock-calibration` dataset labels 16 queries with whether they
should escalate: 10 too vague to classify (including short non-English and
transliterated messages), 6 specific enough to route (4 of them non-English, so
that lowering confidence on everything non-English cannot pass as a fix).

Same prompt, same dataset, two providers:

| Metric | Gemini 2.5 Flash Lite | Claude Haiku 4.5 |
|---|---|---|
| escalation_correct | 0.47 | 0.93 |
| routed_type_correct | 1.00 | 1.00 |

Note: the Gemini figure is from a partial run — 3 of 16 examples failed on the
free-tier daily quota, so the average covers 13 cases. It is reported here as
indicative and will be re-measured on a full run.

**Classification is not the problem.** Both providers label every routable
query correctly. What differs is the confidence attached to a vague one:
Gemini scores `"не работает"` as `bug` at 0.90, sailing past the threshold,
while Haiku escalates the same message.

**One case fails on both.** `"форма не работает"` is not escalated. Naming a
subject ("форма") appears to read as sufficient

### The fix

Two changes to the confidence section of the router prompt:

1. Confidence was redefined as "does this message contain enough information
   to act on", not "how sure am I of the category". The old wording said to
   lower confidence when a message was "too vague to classify", which the model
   read literally — it could classify `"не работает"` as a bug, so it scored
   0.90. Naming a subject was called out explicitly as not being diagnostic
   information.
2. The threshold value was removed from the prompt in favour of bands (0.15–
   0.35 for vague, 0.75–0.95 for actionable). Naming 0.6 in the text anchored
   the model to it: an earlier revision scored `"форма не работает"` at exactly
   0.60, which the graph compares with `<` and therefore did not escalate. With
   bands the same message scores 0.25.

Measured on Claude Haiku 4.5:

| Metric | Before | After |
|---|---|---|
| escalation_correct (16 cases) | 0.93 | 1.00 |
| routed_type_correct (6 routable cases) | 1.00 | 1.00 |
| routing_accuracy (25 cases) | — | 1.00 |

The second and third rows are the guard rails: a prompt that simply lowered
confidence everywhere would score well on escalation while breaking these, and
four of the six routable cases are non-English precisely so that "distrust
anything not in English" cannot pass as a fix.

Gemini figures are pending a free-tier quota reset.

## Step 9 — Automated tests with stubbed providers

**Before:** `pytest` could not run at all. Sixteen manual probe scripts under
`tests/` carry a `*_test.py` suffix, which pytest collects by default. Collection
means import, and importing any of them reaches `src.graph`, which builds the
graph at module level and fails without API keys. A CI run would have been red
before a single assertion executed.

**After:** 129 tests, 1.4 seconds, no network, no keys, no quota. Verified by
renaming `.env` away and running the full suite.

### What was done

- Narrowed collection to `tests/unit` via `testpaths`, and `python_files` to
  `test_*.py`. The manual probes keep their names and their purpose: they are
  run by hand against live Pinecone, Cohere and WordPress.
- Renamed `scripts/test_embed.py` and `scripts/test_embed_cohere.py` to
  `*_probe.py`, so an explicit `pytest scripts/` cannot pick them up either.
- `tests/unit/conftest.py` injects stub credentials into `os.environ` at module
  scope, before pytest imports any test module. Environment variables outrank
  `.env` in pydantic-settings, so the suite behaves identically with or without
  a local `.env`.
- `requirements-dev.txt` split out from `requirements.txt`.
- `route_after_investigation` moved from inside `build_graph` to module level,
  so the loop condition is testable on plain dictionaries.

### Coverage by layer

| Module | What is pinned |
|---|---|
| `settings.py` | Validators reject a provider without its key, a retry ceiling below the initial wait, tracing without a key. Secrets stay out of `repr`. |
| `graph.py` | All four routing branches, the threshold boundary in both directions, the investigation loop, escalation. |
| `retry.py` | Retryable and non-retryable failures, attempt caps, async support, and the flags subclasses deliberately flip. |
| `llm_call.py` | String and block content shapes, empty answers, schema violations, provider-error mapping. |
| `retriever.py` | Rerank indices selecting the right metadata, stage isolation, per-stage exceptions. |
| Four agents | Model tier, prompt assembly, state keys, and every escalation path. |

### Defects found by writing the tests

1. **`requirements.txt` was unusable.** `tenacity==<9.1.4>` is not valid PEP 508
   syntax — `pip install -r` fails on that line. Never surfaced locally because
   packages had always been installed one at a time. `langchain-google-genai`
   was listed twice; `pydantic-settings` and `langsmith` were missing despite
   being imported.
2. **Two probe scripts sat under a name pytest collects.** Latent until the day
   someone ran `pytest` at the repository root.

### Deliberate omissions

- Retryable paths in `llm_call.py` are not exercised through the decorator:
  `@with_retry()` carries production waits, so those tests would really sleep.
  The policy itself is covered in `test_retry.py` with millisecond waits.
- One test reads the real `jfb_hooks.md` rather than a fixture. If that file is
  lost in a merge, the code generator starts inventing hook signatures — the
  exact failure its prompt exists to prevent.

  ## Step 10 — Continuous integration

Every push to any branch runs three checks on a clean Ubuntu runner: `ruff
check .`, `mypy src app.py`, `pytest`. First run passed in 1m 13s.

**No secrets are configured for the workflow.** That is the point rather than an
oversight: the suite injects stub credentials in `conftest.py` and replaces
every outbound call, so a green run without keys is evidence the tests are
genuinely isolated. Any real network call would fail on a missing key instead of
passing quietly on a developer machine that happens to have `.env` in place.

The pipeline was verified in both directions. A branch carrying one deliberately
failing assertion was pushed and the run went red with exit code 1; the branch
was then deleted. A CI that has never been seen to fail proves nothing.

**Scope decisions:**

- `ruff` runs on the whole repository. `mypy` runs on `src` and `app.py` only —
  annotations in `tests/` are switched off in `per-file-ignores`, and typing the
  stubs under `--strict` would be a lot of work for no defect-finding power.
- `app.py` was pulled into the type check rather than left out. It is the entry
  point every reader of the repository opens first, and it reaches into graph
  state. Three `type-arg` and `no-any-return` errors were fixed to get it in.

### Known limitation carried forward

`src/retry.py` backs off on a fixed exponential schedule and ignores the
`retryDelay` the provider returns in the response body. Under batch evaluation
runs the providers ask for 20–40 seconds while the policy waits 1–3. Left
unfixed deliberately: the two SDKs report the delay in different places —
Gemini in a structured error detail, Anthropic in a `retry-after` header — so a
correct fix means parsing both formats, and the failure only appears under batch
load, which the evaluation scripts already work around with explicit pauses.