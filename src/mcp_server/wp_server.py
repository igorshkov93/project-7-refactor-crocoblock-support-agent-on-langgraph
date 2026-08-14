"""MCP server exposing read-only WordPress diagnostics to the support agents.

Failures are reported through the MCP error channel rather than returned as
data. A dict like ``{"error": "..."}`` is indistinguishable from a real payload
to the model, which would then reason about a site it never reached.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow running this file directly as an MCP server subprocess.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from src.exceptions import SupportAgentError
from src.mcp_server.wp_client import wp_get_object

logger = logging.getLogger(__name__)

mcp = FastMCP("wordpress-support")

#: Mirrors the cap enforced by the mu-plugin.
MAX_LOG_LINES = 200


def _call(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch a diagnostic endpoint, translating failures into MCP tool errors.

    Raises:
        ToolError: If the site is unreachable, refuses the request, or answers
            with an unexpected payload. The original exception is logged with
            its type and re-raised as a message the model can act on.
    """
    try:
        return wp_get_object(path, params)
    except SupportAgentError as error:
        logger.error(
            "WordPress diagnostic call failed",
            extra={"wp_path": path, "error_type": type(error).__name__},
        )
        raise ToolError(f"Could not read {path} from the site: {error}") from error


@mcp.tool()
def get_env_info() -> dict[str, Any]:
    """Get the WordPress site environment.

    Returns WordPress, PHP and MySQL versions, active theme, debug settings
    and PHP limits. Call this first when investigating any bug report, before
    asking the user for their setup details.
    """
    return _call("support-agent/v1/env")


@mcp.tool()
def list_plugins() -> dict[str, Any]:
    """List installed plugins with versions and activation status.

    Active plugins are listed first. Use this to check for outdated versions
    or plugin conflicts when a feature is reported as broken.
    """
    return _call("support-agent/v1/plugins")


@mcp.tool()
def get_error_log(lines: int = 50) -> dict[str, Any]:
    """Read the tail of the WordPress debug log.

    Args:
        lines: How many trailing lines to return (max 200, default 50).

    Use this when the user reports a fatal error, a white screen, or any
    failure with no visible message on the front end.
    """
    safe_lines = max(1, min(lines, MAX_LOG_LINES))
    return _call("support-agent/v1/error-log", {"lines": safe_lines})


if __name__ == "__main__":
    mcp.run()
