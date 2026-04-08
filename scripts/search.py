"""
search.py — Search contacts using natural language queries.

Usage:
    python scripts/search.py "find someone who can assess an AI chip design"
    python scripts/search.py "who can help a log management company increase sales"

Currently using Ollama (local) for embeddings and LLM.
To switch to OpenAI, change EMBEDDING_PROVIDER and LLM_PROVIDER to "openai".
"""

import os
import sys
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── PROVIDER TOGGLES ───────────────────────────────────────────────────────
# Change these to "openai" when you have API credits
EMBEDDING_PROVIDER = "ollama"  # "ollama" or "openai"
LLM_PROVIDER = "ollama"        # "ollama" or "openai"
# ───────────────────────────────────────────────────────────────────────────

if EMBEDDING_PROVIDER == "openai" or LLM_PROVIDER == "openai":
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if EMBEDDING_PROVIDER == "ollama" or LLM_PROVIDER == "ollama":
    import ollama

EMBEDDING_MODEL = "nomic-embed-text" if EMBEDDING_PROVIDER == "ollama" else "text-embedding-3-small"
LLM_MODEL = "llama3" if LLM_PROVIDER == "ollama" else "gpt-4o"
TOP_K = 5

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "miv-contacts"))


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


def search_contacts(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and search Pinecone for matching contacts."""
    print(f"Embedding query...")
    query_vector = generate_embedding(query)

    print(f"Searching Pinecone for top {top_k} matches...")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    contacts = []
    for match in results.matches:
        contact = {
            "id": match.id,
            "score": round(match.score, 4),
            "name": match.metadata.get("name", "Unknown"),
            "title": match.metadata.get("title", ""),
            "company": match.metadata.get("company", ""),
            "email": match.metadata.get("email", ""),
            "expertise": match.metadata.get("expertise", ""),
            "bio": match.metadata.get("bio", ""),
        }
        contacts.append(contact)

    return contacts


def format_contacts_for_llm(contacts: list[dict]) -> str:
    """Format retrieved contacts into a string for the LLM prompt."""
    parts = []
    for i, c in enumerate(contacts, 1):
        parts.append(
            f"Contact {i} (relevance score: {c['score']}):\n"
            f"  Name: {c['name']}\n"
            f"  Title: {c['title']}\n"
            f"  Company: {c['company']}\n"
            f"  Email: {c['email']}\n"
            f"  Expertise: {c['expertise']}\n"
            f"  Bio: {c['bio']}"
        )
    return "\n\n".join(parts)


def generate_response(query: str, contacts: list[dict]) -> str:
    """Use an LLM to generate a natural language response based on retrieved contacts."""
    contacts_text = format_contacts_for_llm(contacts)

    system_prompt = """You are an AI assistant for a venture capital fund. Your job is to help 
fund members find the right person in their professional network for a specific need.

You will be given a query and a list of contacts retrieved from the fund's database, ranked by 
relevance score (0 to 1, higher = more relevant).

Your response should:
1. Present the most relevant contacts (skip any that are clearly irrelevant, score below 0.3)
2. For each relevant contact, explain WHY they are a good match for the specific query
3. Be concise and actionable — the user wants to know who to reach out to and why
4. If no contacts are a strong match, say so honestly

IMPORTANT: Only reference contacts that were provided to you. Never make up people or information."""

    user_prompt = f"""Query: "{query}"

Retrieved contacts:
{contacts_text}

Based on these contacts, who is the best match for this query and why?"""

    print("Generating AI response...\n")

    if LLM_PROVIDER == "openai":
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    else:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.3}
        )
        return response["message"]["content"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/search.py \"your natural language query\"")
        print("")
        print("Examples:")
        print('  python scripts/search.py "find someone who can assess an AI chip design"')
        print('  python scripts/search.py "who can help a log management company increase sales"')
        print('  python scripts/search.py "find a marketing expert in the gaming industry"')
        sys.exit(1)

    query = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Providers: embeddings={EMBEDDING_PROVIDER}, llm={LLM_PROVIDER}")
    print(f"{'='*60}\n")

    # Step 1: Search Pinecone for relevant contacts
    contacts = search_contacts(query)

    if not contacts:
        print("No contacts found in the database.")
        sys.exit(0)

    # Step 2: Print raw results
    print(f"\n--- Raw Results ({len(contacts)} matches) ---\n")
    for i, c in enumerate(contacts, 1):
        print(f"  {i}. {c['name']} — {c['title']} at {c['company']} (score: {c['score']})")
    print()

    # Step 3: Generate LLM response
    print(f"{'='*60}")
    print("AI RECOMMENDATION")
    print(f"{'='*60}\n")
    response = generate_response(query, contacts)
    print(response)
    print()


if __name__ == "__main__":
    main()