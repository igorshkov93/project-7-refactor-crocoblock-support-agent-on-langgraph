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