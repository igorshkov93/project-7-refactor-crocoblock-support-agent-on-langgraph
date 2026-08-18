"""Verify MCP tools are usable as LangChain tools."""
import asyncio

from src.mcp_server.client import load_tools


def main():
    tools = load_tools()
    print(f"Tools loaded: {len(tools)}\n")

    for tool in tools:
        print(f"  {tool.name}")
        print(f"    args: {list(tool.args.keys())}")

    env_tool = next(t for t in tools if t.name == "get_env_info")
    result = asyncio.run(env_tool.ainvoke({}))
    print(f"\nCall result (truncated):\n  {str(result)[:200]}")


if __name__ == "__main__":
    main()
