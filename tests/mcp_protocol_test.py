"""Talk to the MCP server the way a real client does: over stdio."""
import asyncio

from fastmcp import Client


async def main():
    async with Client("src/mcp_server/wp_server.py") as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools)} tools over stdio\n")

        result = await client.call_tool("get_env_info", {})
        data = result.data
        print(f"WP {data['wp_version']} / PHP {data['php_version']}")
        print(f"Theme: {data['active_theme']}")


if __name__ == "__main__":
    asyncio.run(main())
