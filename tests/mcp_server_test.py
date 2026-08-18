"""List tools exposed by the MCP server."""
import asyncio

from src.mcp_server.wp_server import mcp


async def main():
    tools = await mcp.list_tools()
    print(f"Tools registered: {len(tools)}\n")
    for tool in tools:
        first_line = (tool.description or "").splitlines()[0]
        print(f"  - {tool.name}: {first_line}")


if __name__ == "__main__":
    asyncio.run(main())
