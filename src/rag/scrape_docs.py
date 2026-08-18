"""Download documentation pages and extract clean text."""

import json
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

URLS_FILE = Path("data/doc_urls.json")
OUTPUT = Path("data/docs.json")
HEADERS = {"User-Agent": "crocoblock-support-agent/1.0 (portfolio project)"}
DELAY = 0.5


def extract_page(url: str) -> dict[str, Any] | None:
    """Fetch a page and return its title and main text."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    heading = soup.find("h1")
    title = heading.get_text(strip=True) if heading else url.rstrip("/").split("/")[-1]

    # Select the content container FIRST, then clean inside it.
    main = soup.find("main") or soup.find("div", class_="single-addon")
    if main is None:
        return None

    for tag in main.find_all(["script", "style", "nav", "noscript"]):
        tag.decompose()

    # Remove the "related docs" navigation block found on feature pages.
    for tag in main.find_all(class_="related-docs"):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) < 200:
        return None

    return {"url": url, "title": title, "text": text, "chars": len(text)}


def main() -> None:
    """Fetch every documentation page and store the extracted text."""
    setup_logging()

    urls = json.loads(URLS_FILE.read_text(encoding="utf-8"))
    logger.info("Pages to fetch: %d", len(urls))

    docs: list[dict[str, Any]] = []
    skipped: list[str] = []

    for index, url in enumerate(urls, 1):
        try:
            page = extract_page(url)
        except requests.HTTPError as error:
            skipped.append(url)
            logger.warning("[%d/%d] HTTP %s: %s", index, len(urls), error.response.status_code, url)
            continue
        except requests.RequestException as error:
            skipped.append(url)
            logger.warning("[%d/%d] %s: %s", index, len(urls), type(error).__name__, url)
            continue

        if page is None:
            skipped.append(url)
            logger.debug("[%d/%d] too short, skipped: %s", index, len(urls), url)
        else:
            docs.append(page)
            logger.debug(
                "[%d/%d] %d chars: %s",
                index,
                len(urls),
                page["chars"],
                page["title"][:50],
            )

        time.sleep(DELAY)

    OUTPUT.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")

    total_chars = sum(d["chars"] for d in docs)
    logger.info("Collected %d pages, %d characters", len(docs), total_chars)
    logger.info("Skipped %d pages", len(skipped))
    logger.info("Saved to %s", OUTPUT)

if __name__ == "__main__":
    main()
