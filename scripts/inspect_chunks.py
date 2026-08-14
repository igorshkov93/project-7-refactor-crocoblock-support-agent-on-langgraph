"""Look for navigation-like chunks that should be filtered out."""
import json
from pathlib import Path

CHUNKS = Path("data/chunks.json")


def link_list_ratio(text: str) -> float:
    """Share of lines that look like article titles rather than prose."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    # Title-like lines are short and end without punctuation.
    title_like = [
        line for line in lines
        if len(line) < 90 and not line.rstrip().endswith((".", ":", ";", "!", "?"))
    ]
    return len(title_like) / len(lines)


def main():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))

    suspicious = [c for c in chunks if link_list_ratio(c["text"]) > 0.8]
    print(f"Total chunks:      {len(chunks)}")
    print(f"Navigation-like:   {len(suspicious)} ({len(suspicious)/len(chunks):.0%})")

    print("\nExamples:")
    for chunk in suspicious[:3]:
        preview = chunk["text"][:120].replace("\n", " | ")
        print(f"\n  {chunk['id']}")
        print(f"  {preview}...")


if __name__ == "__main__":
    main()