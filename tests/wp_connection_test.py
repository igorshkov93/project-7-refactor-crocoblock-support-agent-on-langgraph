"""Verify WordPress REST API access via Application Password."""
import base64
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("WP_BASE_URL", "").rstrip("/")
USER = os.getenv("WP_USER", "")
APP_PASSWORD = os.getenv("WP_APP_PASSWORD", "")


def main():
    if not all([BASE_URL, USER, APP_PASSWORD]):
        print("Missing WP_* values in .env")
        return

    token = base64.b64encode(f"{USER}:{APP_PASSWORD}".encode()).decode()
    request = urllib.request.Request(
        f"{BASE_URL}/wp-json/wp/v2/users/me",
        headers={"Authorization": f"Basic {token}"},
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            print(f"Status: {response.status}")
            print(response.read().decode()[:300])
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode()[:300]}")
    except urllib.error.URLError as error:
        print(f"Connection failed: {error.reason}")


if __name__ == "__main__":
    main()
