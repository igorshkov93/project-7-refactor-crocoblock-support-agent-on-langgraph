"""Shared fixtures for the automated unit tests.

Modules under ``src`` build their configuration at import time: ``settings.py``
ends with ``settings = Settings()``, and a validator refuses to construct the
object when the active provider has no API key. A plain ``import src.graph``
therefore fails on any machine without ``.env`` — exactly the situation in CI.

Stub credentials are injected into ``os.environ`` below, at module scope, so
they land before pytest imports a single test module. Environment variables
outrank ``.env`` in pydantic-settings, so these values win locally too and the
suite behaves identically on both machines. Nothing here reaches a live
provider: outbound calls are replaced by fakes in the tests themselves.
"""
from __future__ import annotations

import os

STUB_ENV = {
    "LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_API_KEY": "test-google-key",
    "PINECONE_API_KEY": "test-pinecone-key",
    "COHERE_API_KEY": "test-cohere-key",
    "WP_BASE_URL": "https://wordpress.test",
    "WP_USER": "test-user",
    "WP_APP_PASSWORD": "test-password",
    # Tracing during tests would pollute the LangSmith project with runs that
    # carry no real inputs.
    "LANGSMITH_TRACING": "false",
}

for _name, _value in STUB_ENV.items():
    os.environ[_name] = _value
