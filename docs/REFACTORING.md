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