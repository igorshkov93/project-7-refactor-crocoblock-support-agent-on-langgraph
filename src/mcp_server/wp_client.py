"""Thin, typed client for the WordPress REST API.

Every response is narrowed to a concrete JSON shape before it leaves this
module. The sandbox endpoints answer with an object on success but with a list
on some error paths, and letting that union escape forced each caller to guess.
Callers now ask for the shape they expect — :func:`wp_get_object` or
:func:`wp_get_list` — and a mismatch surfaces as a
:class:`~src.exceptions.WordPressResponseError` at the boundary instead of an
``AttributeError`` three frames deeper.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.exceptions import (
    ConfigurationError,
    WordPressAuthError,
    WordPressError,
    WordPressRequestError,
    WordPressResponseError,
    WordPressTimeoutError,
)
from src.retry import with_retry
from src.settings import settings

logger = logging.getLogger(__name__)

JSONObject = dict[str, Any]
JSONArray = list[Any]

#: Statuses worth repeating: throttling and transient server-side failures.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _credentials() -> tuple[str, str, str]:
    """Return ``(base_url, user, app_password)``, validated.

    Read at call time rather than at import, so tests can override settings and
    so an incomplete ``.env`` fails with a clear message instead of an empty
    Authorization header.

    Raises:
        ConfigurationError: If any of the three values is missing.
    """
    base_url = settings.wp_base_url.rstrip("/")
    user = settings.wp_user
    password = (
        settings.wp_app_password.get_secret_value()
        if settings.wp_app_password is not None
        else ""
    )
    missing = [
        name
        for name, value in (
            ("WP_BASE_URL", base_url),
            ("WP_USER", user),
            ("WP_APP_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            f"WordPress credentials are incomplete: {', '.join(missing)} not set in .env"
        )
    return base_url, user, password


def _auth_header(user: str, password: str) -> str:
    """Build the Basic auth header for a WordPress application password."""
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def _classify_http_error(error: urllib.error.HTTPError, path: str) -> WordPressError:
    """Map an HTTP status onto the exception that describes it best."""
    body = error.read().decode(errors="replace")[:200]
    message = f"{error.reason} on {path}: {body}"
    if error.code in (401, 403):
        return WordPressAuthError(message, status_code=error.code)
    if error.code in RETRYABLE_STATUS:
        return WordPressError(message, status_code=error.code)
    return WordPressRequestError(message, status_code=error.code)


@with_retry()
def _wp_request(path: str, params: dict[str, Any] | None = None) -> Any:
    """Send a GET request to a WP REST endpoint and return the parsed JSON.

    The return type is deliberately ``Any``: narrowing happens in the public
    wrappers, which know what shape the caller expects.

    Args:
        path: Endpoint path after ``/wp-json``, e.g. ``"wp/v2/users/me"``.
        params: Optional query parameters.

    Raises:
        ConfigurationError: If credentials are missing.
        WordPressTimeoutError: If the site did not respond in time.
        WordPressAuthError: If the application password was rejected.
        WordPressRequestError: If the endpoint rejected the request.
        WordPressError: On a transient server-side or network failure.
        WordPressResponseError: If the body is not valid JSON.
    """
    base_url, user, password = _credentials()
    url = f"{base_url}/wp-json/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    request = urllib.request.Request(
        url, headers={"Authorization": _auth_header(user, password)}
    )
    logger.debug("WP request", extra={"wp_path": path, "wp_url": url})

    try:
        with urllib.request.urlopen(
            request, timeout=settings.wp_timeout_seconds
        ) as response:
            raw = response.read().decode()
    except TimeoutError as error:
        raise WordPressTimeoutError(
            f"No response from {base_url} within {settings.wp_timeout_seconds}s"
        ) from error
    except urllib.error.HTTPError as error:
        raise _classify_http_error(error, path) from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise WordPressTimeoutError(
                f"No response from {base_url} within {settings.wp_timeout_seconds}s"
            ) from error
        raise WordPressError(f"Cannot reach {base_url}: {error.reason}") from error

    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise WordPressResponseError(
            f"Response from {path} is not valid JSON: {raw[:200]}"
        ) from error


def wp_get_object(path: str, params: dict[str, Any] | None = None) -> JSONObject:
    """Call a WP REST endpoint that is expected to answer with a JSON object.

    Raises:
        WordPressResponseError: If the endpoint answered with something else.
    """
    payload = _wp_request(path, params)
    if not isinstance(payload, dict):
        raise WordPressResponseError(
            f"Expected a JSON object from {path}, got {type(payload).__name__}"
        )
    return payload


def wp_get_list(path: str, params: dict[str, Any] | None = None) -> JSONArray:
    """Call a WP REST endpoint that is expected to answer with a JSON array.

    Raises:
        WordPressResponseError: If the endpoint answered with something else.
    """
    payload = _wp_request(path, params)
    if not isinstance(payload, list):
        raise WordPressResponseError(
            f"Expected a JSON array from {path}, got {type(payload).__name__}"
        )
    return payload
