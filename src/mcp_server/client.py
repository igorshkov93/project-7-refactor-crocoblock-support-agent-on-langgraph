"""Bridge between the MCP server and LangChain tools.

The server runs as a stdio subprocess, so every failure mode of process
management applies: it may fail to launch, hang during the handshake, or die
mid-session. All three surface as :class:`~src.exceptions.MCPServerError`
rather than as an adapter-specific traceback.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.exceptions import MCPServerError
from src.logging_config import get_logger
from src.retry import with_retry
from src.settings import settings

logger = get_logger(__name__)

SERVER_PATH = Path(__file__).resolve().parent / "wp_server.py"

_tools_cache: list[BaseTool] | None = None


async def load_tools_async() -> list[BaseTool]:
    """Start the MCP server and expose its tools as LangChain tools.

    Raises:
        MCPServerError: If the server did not start or did not answer in time.
    """
    client = MultiServerMCPClient(
        {
            "wordpress": {
                "command": "python",
                "args": [str(SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )
    try:
        return await asyncio.wait_for(
            client.get_tools(), timeout=settings.wp_timeout_seconds
        )
    except TimeoutError as error:
        raise MCPServerError(
            f"MCP server did not respond within {settings.wp_timeout_seconds}s"
        ) from error
    except Exception as error:
        raise MCPServerError(f"Could not start the MCP server: {error}") from error


@with_retry()
def load_tools() -> list[BaseTool]:
    """Return the MCP tools, starting the server on first use.

    Cached for the lifetime of the process: the handshake costs a subprocess
    launch, and the tool list does not change while the server is running.

    Raises:
        MCPServerError: If the server could not be reached after retries.
    """
    global _tools_cache
    if _tools_cache is None:
        logger.info("Starting MCP server", extra={"server_path": str(SERVER_PATH)})
        _tools_cache = asyncio.run(load_tools_async())
        logger.info("MCP tools loaded", extra={"tool_count": len(_tools_cache)})
    return _tools_cache
