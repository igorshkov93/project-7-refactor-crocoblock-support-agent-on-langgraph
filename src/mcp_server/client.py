"""Bridge between the MCP server and LangChain tools."""
import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_PATH = Path(__file__).resolve().parent / "wp_server.py"

_tools_cache = None


async def load_tools_async() -> list:
    """Start the MCP server and expose its tools as LangChain tools."""
    client = MultiServerMCPClient(
        {
            "wordpress": {
                "command": "python",
                "args": [str(SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


def load_tools() -> list:
    """Synchronous wrapper with a process-level cache."""
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = asyncio.run(load_tools_async())
    return _tools_cache
