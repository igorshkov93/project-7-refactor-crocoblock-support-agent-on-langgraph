"""Check the custom diagnostic endpoints added by the mu-plugin."""
from src.mcp_server.wp_client import WPError, wp_get


def main():
    try:
        env = wp_get("support-agent/v1/env")
        print(f"WP {env['wp_version']} / PHP {env['php_version']}")

        data = wp_get("support-agent/v1/plugins")
        print(f"\nPlugins: {data['active']} active of {data['total']}")
        for plugin in data["plugins"]:
            mark = "on " if plugin["active"] else "off"
            print(f"  [{mark}] {plugin['name']} {plugin['version']}")
    except WPError as error:
        print(f"Failed: {error}")


if __name__ == "__main__":
    main()
