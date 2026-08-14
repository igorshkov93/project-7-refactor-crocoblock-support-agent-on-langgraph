"""Delete the Pinecone index so it can be recreated with a new dimension."""
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.rag.index_chunks import INDEX_NAME

load_dotenv()


def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME in existing:
        pc.delete_index(INDEX_NAME)
        print(f"Deleted index: {INDEX_NAME}")
    else:
        print(f"Index not found: {INDEX_NAME}")


if __name__ == "__main__":
    main()