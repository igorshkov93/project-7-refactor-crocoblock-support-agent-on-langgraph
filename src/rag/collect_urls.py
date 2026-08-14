"""Collect documentation URLs from JetFormBuilder sitemaps."""
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

SITEMAPS = [
    "https://jetformbuilder.com/post-sitemap.xml",
    "https://jetformbuilder.com/addons-sitemap.xml",
]
OUTPUT = Path("data/doc_urls.json")
HEADERS = {"User-Agent": "crocoblock-support-agent/1.0 (portfolio project)"}


def collect_from_sitemap(url: str) -> list[str]:
    """Return page URLs from a sitemap, ignoring image entries."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "xml")

    urls = []
    for entry in soup.find_all("url"):
        # Only direct <loc> children: <image:loc> tags must be skipped.
        loc = entry.find("loc", recursive=False)
        if loc:
            urls.append(loc.get_text(strip=True))
    return urls


def main() -> None:
    """Collect and store the documentation URL list."""
    setup_logging()

    all_urls: list[str] = []

    for sitemap in SITEMAPS:
        found = collect_from_sitemap(sitemap)
        logger.info("%s: %d URLs", sitemap.split("/")[-1], len(found))
        all_urls.extend(found)

    unique = sorted(set(all_urls))
    logger.info("Total unique URLs: %d", len(unique))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(unique, indent=2), encoding="utf-8")
    logger.info("Saved to %s", OUTPUT)

    for url in unique[:5]:
        logger.debug("Sample: %s", url)


if __name__ == "__main__":
    main()


