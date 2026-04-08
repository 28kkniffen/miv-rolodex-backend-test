"""
ingest.py — Read contact text files, generate embeddings, and upsert to Pinecone.

Usage:
    python scripts/ingest.py contacts/                  # Ingest all .txt files in folder
    python scripts/ingest.py contacts/sarah_chen.txt    # Ingest a single file

Currently using Ollama (local) for embeddings.
To switch to OpenAI, change EMBEDDING_PROVIDER to "openai".
"""

import os
import sys
import glob
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── EMBEDDING PROVIDER TOGGLE ──────────────────────────────────────────────
# Change this to "openai" when you have API credits
EMBEDDING_PROVIDER = "ollama"  # "ollama" or "openai"
# ───────────────────────────────────────────────────────────────────────────

if EMBEDDING_PROVIDER == "openai":
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMBEDDING_MODEL = "text-embedding-3-small"
else:
    import ollama
    EMBEDDING_MODEL = "nomic-embed-text"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "miv-contacts"))


def parse_contact_file(filepath: str) -> dict:
    """Parse a structured contact text file into a dictionary."""
    contact = {}
    current_key = None
    current_value = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if ": " in line and line.split(": ")[0] in ["Name", "Title", "Company", "Email", "Expertise", "Bio"]:
                if current_key:
                    contact[current_key] = " ".join(current_value).strip()

                key, value = line.split(": ", 1)
                current_key = key.lower()
                current_value = [value]
            else:
                if current_key:
                    current_value.append(line)

        if current_key:
            contact[current_key] = " ".join(current_value).strip()

    return contact


def build_embedding_text(contact: dict) -> str:
    """Combine contact fields into a single string for embedding."""
    parts = []
    if contact.get("name"):
        parts.append(f"Name: {contact['name']}")
    if contact.get("title"):
        parts.append(f"Title: {contact['title']}")
    if contact.get("company"):
        parts.append(f"Company: {contact['company']}")
    if contact.get("expertise"):
        parts.append(f"Expertise: {contact['expertise']}")
    if contact.get("bio"):
        parts.append(f"Bio: {contact['bio']}")
    return "\n".join(parts)


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector using the configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    else:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response["embeddings"][0]


def create_contact_id(contact: dict) -> str:
    """Create a unique ID for a contact based on name and company."""
    name = contact.get("name", "unknown").lower().replace(" ", "_")
    company = contact.get("company", "unknown").lower().replace(" ", "_")
    return f"{name}_{company}"


def upsert_contact(contact: dict):
    """Embed a contact and upsert it to Pinecone."""
    embedding_text = build_embedding_text(contact)

    print(f"  Generating embedding for {contact.get('name', 'Unknown')}...")
    vector = generate_embedding(embedding_text)

    contact_id = create_contact_id(contact)
    metadata = {
        "name": contact.get("name", ""),
        "title": contact.get("title", ""),
        "company": contact.get("company", ""),
        "email": contact.get("email", ""),
        "expertise": contact.get("expertise", ""),
        "bio": contact.get("bio", ""),
    }

    print(f"  Upserting to Pinecone (id: {contact_id})...")
    index.upsert(vectors=[(contact_id, vector, metadata)])
    print(f"  Done!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <path-to-contacts-folder-or-file>")
        print("  python scripts/ingest.py contacts/")
        print("  python scripts/ingest.py contacts/sarah_chen.txt")
        sys.exit(1)

    path = sys.argv[1]

    print(f"Using embedding provider: {EMBEDDING_PROVIDER} ({EMBEDDING_MODEL})\n")

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
        if not files:
            print(f"No .txt files found in {path}")
            sys.exit(1)
        print(f"Found {len(files)} contact file(s) in {path}\n")
    elif os.path.isfile(path):
        files = [path]
        print(f"Processing single file: {path}\n")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    success_count = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"[{filename}]")

        try:
            contact = parse_contact_file(filepath)

            if not contact.get("name"):
                print(f"  Skipping — no Name field found\n")
                continue

            upsert_contact(contact)
            success_count += 1
            print()

        except Exception as e:
            print(f"  Error: {e}\n")

    print(f"=== Ingested {success_count}/{len(files)} contacts ===")


if __name__ == "__main__":
    main()