"""Check the WordPress client against the live sandbox."""
from src.mcp_server.wp_client import WPError, wp_get


def main():
    try:
        me = wp_get("wp/v2/users/me")
        print(f"Authenticated as: {me['name']} (id {me['id']})")

        posts = wp_get("wp/v2/posts", {"per_page": 3})
        print(f"Posts returned: {len(posts)}")
    except WPError as error:
        print(f"Failed: {error}")


if __name__ == "__main__":
    main()
