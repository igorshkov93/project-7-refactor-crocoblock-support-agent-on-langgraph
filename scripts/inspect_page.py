"""Inspect the HTML structure of a single documentation page."""
import requests
from bs4 import BeautifulSoup

URL = "https://jetformbuilder.com/addons/address-autocomplete/"
HEADERS = {"User-Agent": "crocoblock-support-agent/1.0 (portfolio project)"}


def main():
    response = requests.get(URL, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(response.text, "lxml")

    print(f"Raw HTML length: {len(response.text):,}")

    for tag_name in ("main", "article", "body"):
        found = soup.find(tag_name)
        if found:
            print(f"  <{tag_name}> found, text: {len(found.get_text()):,} chars")
        else:
            print(f"  <{tag_name}> not found")

    h1 = soup.find("h1")
    print(f"\nH1: {h1.get_text(strip=True) if h1 else 'none'}")

    print("\nTop-level containers with most text:")
    candidates = []
    for div in soup.find_all(["div", "section", "article"]):
        text_len = len(div.get_text(strip=True))
        classes = " ".join(div.get("class") or [])
        candidates.append((text_len, div.name, classes[:70]))

    candidates.sort(reverse=True)
    for text_len, name, classes in candidates[:12]:
        print(f"  {text_len:>7} chars  <{name}> {classes}")


if __name__ == "__main__":
    main()