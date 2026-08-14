"""MCP server exposing WordPress diagnostics to support agents."""
import sys
from pathlib import Path

# Allow running this file directly as an MCP server subprocess.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastmcp import FastMCP

from src.mcp_server.wp_client import WPError, wp_get

mcp = FastMCP("wordpress-support")


@mcp.tool()
def get_env_info() -> dict:
    """Get the WordPress site environment.

    Returns WordPress, PHP and MySQL versions, active theme, debug settings
    and PHP limits. Call this first when investigating any bug report, before
    asking the user for their setup details.
    """
    try:
        return wp_get("support-agent/v1/env")
    except WPError as error:
        return {"error": str(error)}


@mcp.tool()
def list_plugins() -> dict:
    """List installed plugins with versions and activation status.

    Active plugins are listed first. Use this to check for outdated versions
    or plugin conflicts when a feature is reported as broken.
    """
    try:
        return wp_get("support-agent/v1/plugins")
    except WPError as error:
        return {"error": str(error)}


@mcp.tool()
def get_error_log(lines: int = 50) -> dict:
    """Read the tail of the WordPress debug log.

    Args:
        lines: How many trailing lines to return (max 200, default 50).

    Use this when the user reports a fatal error, a white screen, or any
    failure with no visible message on the front end.
    """
    try:
        return wp_get("support-agent/v1/error-log", {"lines": lines})
    except WPError as error:
        return {"error": str(error)}


if __name__ == "__main__":
    mcp.run()
