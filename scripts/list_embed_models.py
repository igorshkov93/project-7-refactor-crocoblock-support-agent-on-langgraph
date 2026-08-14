"""List Gemini models that support embeddings."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "embedContent" in actions:
            print(f"  {model.name}")


if __name__ == "__main__":
    main()