"""Render the compiled graph to a PNG file."""
from pathlib import Path

from src.graph import graph

OUTPUT = Path("docs/diagrams/graph.png")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
