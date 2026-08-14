"""Check the Cohere embedding model and its output dimension."""
import os

import cohere
from dotenv import load_dotenv

load_dotenv()

MODEL = "embed-v4.0"


def main():
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    response = client.embed(
        texts=["How do I save form records?"],
        model=MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    vector = response.embeddings.float[0]
    print(f"Model:     {MODEL}")
    print(f"Dimension: {len(vector)}")
    print(f"Sample:    {vector[:5]}")


if __name__ == "__main__":
    main()
    