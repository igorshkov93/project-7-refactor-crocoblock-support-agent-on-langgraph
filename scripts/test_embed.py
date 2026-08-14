"""Verify the embedding model and its output dimension."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.rag.index_chunks import DIMENSION, EMBED_MODEL


def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, output_dimensionality=DIMENSION
    )
    vector = embeddings.embed_query("How do I save form records?")
    print(f"Model:     {EMBED_MODEL}")
    print(f"Dimension: {len(vector)} (expected {DIMENSION})")
    print(f"Sample:    {vector[:5]}")


if __name__ == "__main__":
    main()