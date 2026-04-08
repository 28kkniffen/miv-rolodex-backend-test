# Rolodex — VC Productivity App

AI-powered semantic contact search for venture capital teams.

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and add your real keys:
- `OPENAI_API_KEY` — from platform.openai.com → API Keys
- `PINECONE_API_KEY` — from pinecone.io → API Keys

### 3. Ingest contacts

Add contact text files to the `contacts/` folder (see existing samples), then run:

```bash
python scripts/ingest.py contacts/
```

This reads each contact file, generates an embedding via OpenAI, and upserts it to Pinecone.

### 4. Search contacts

```bash
python scripts/search.py "find someone who can assess an AI chip design"
```

Returns ranked contacts with relevance scores and an AI-generated explanation.

## Project Structure

```
Rolodex/
├── contacts/          # Contact text files (one per person)
├── scripts/
│   ├── ingest.py      # Embed + upsert contacts to Pinecone
│   └── search.py      # Query contacts via natural language
├── .env.example       # Template for API keys
├── requirements.txt   # Python dependencies
└── README.md
```

## Tech Stack

- **Embeddings:** OpenAI text-embedding-3-small (1536 dimensions)
- **LLM:** OpenAI GPT-4o
- **Vector DB:** Pinecone (free starter tier)
- **Language:** Python 3.10+

## Team

- **Kade** — Project Manager + AI Engineer
- **Duncan** — AI Engineer
- **Mike** — UI Engineer
- **Ryan** — Database Engineer
- **Mateo** — Database Engineer
