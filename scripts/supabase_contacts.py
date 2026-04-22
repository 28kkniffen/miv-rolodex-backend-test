"""
supabase_contacts.py — Sync contact text files into Supabase, scoped to a user.

Runs alongside ingest.py. Reads the same contacts/ folder, produces the same
contact_ids (matching Pinecone vector ids), but writes to Supabase instead of
Pinecone. Your existing ingest.py and search.py are untouched.

Commands:
    sync <user_id> <folder-or-file>    Insert/update contacts for a user
    list <user_id>                     List all contacts for a user
    delete <user_id> <contact_id>      Delete a single contact
    diff <user_id> <folder>            Show what's in the folder vs Supabase

Usage:
    python scripts/supabase_contacts.py sync 00000000-0000-0000-0000-000000000000 contacts/
    python scripts/supabase_contacts.py list 00000000-0000-0000-0000-000000000000
    python scripts/supabase_contacts.py delete 00000000-... sarah_chen_acme_corp
    python scripts/supabase_contacts.py diff 00000000-... contacts/
"""

import os
import sys
import glob
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


# ── Supabase client (lazy init) ────────────────────────────────────────────

_supabase: Client = None
def get_supabase() -> Client:
    """
    Returns a Supabase client using the service_role key.
    This BYPASSES Row Level Security — backend-only, never expose the service
    key to a browser.
    """
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError(
                "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment. "
                "Check your .env file."
            )
        _supabase = create_client(url, key)
    return _supabase


# ── Contact parsing (same scheme as ingest.py) ─────────────────────────────
# These helpers intentionally mirror ingest.py so contact_ids match Pinecone
# vector ids exactly. If ingest.py ever changes its parsing or id scheme,
# this file needs the same update.

CONTACT_FIELDS = ["Name", "Title", "Company", "Email", "Expertise", "Bio"]


def parse_contact_file(filepath: str) -> dict:
    """Parse a structured contact text file into a dictionary. Same as ingest.py."""
    contact = {}
    current_key = None
    current_value = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if ": " in line and line.split(": ")[0] in CONTACT_FIELDS:
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


def create_contact_id(contact: dict) -> str:
    """Same id scheme as ingest.py so Supabase contact_id == Pinecone vector id."""
    name = contact.get("name", "unknown").lower().replace(" ", "_")
    company = contact.get("company", "unknown").lower().replace(" ", "_")
    return f"{name}_{company}"


# ── Commands ───────────────────────────────────────────────────────────────

def upsert_contact(user_id: str, contact: dict) -> tuple[str, str]:
    """
    Insert or update a contact row, scoped to user_id.
    Returns (contact_id, action) where action is 'inserted' or 'updated'.
    """
    supabase = get_supabase()
    contact_id = create_contact_id(contact)

    row = {
        "user_id": user_id,
        "contact_id": contact_id,
        "name": contact.get("name", ""),
        "title": contact.get("title") or None,
        "company": contact.get("company") or None,
        "email": contact.get("email") or None,
        "expertise": contact.get("expertise") or None,
        "bio": contact.get("bio") or None,
    }

    existing = (
        supabase.table("contacts")
        .select("id")
        .eq("user_id", user_id)
        .eq("contact_id", contact_id)
        .execute()
    )

    if existing.data:
        row_id = existing.data[0]["id"]
        supabase.table("contacts").update(row).eq("id", row_id).execute()
        return contact_id, "updated"
    else:
        supabase.table("contacts").insert(row).execute()
        return contact_id, "inserted"


def cmd_sync(user_id: str, path: str):
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

    inserted = updated = skipped = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"[{filename}]")

        try:
            contact = parse_contact_file(filepath)

            if not contact.get("name"):
                print(f"  Skipping — no Name field found\n")
                skipped += 1
                continue

            contact_id, action = upsert_contact(user_id, contact)
            if action == "inserted":
                inserted += 1
                print(f"  ✓ Inserted ({contact_id})\n")
            else:
                updated += 1
                print(f"  ✓ Updated ({contact_id})\n")

        except Exception as e:
            print(f"  Error: {e}\n")
            skipped += 1

    total = len(files)
    print(f"=== Synced {inserted + updated}/{total} contacts "
          f"(inserted: {inserted}, updated: {updated}, skipped: {skipped}) ===")


def cmd_list(user_id: str):
    supabase = get_supabase()
    result = (
        supabase.table("contacts")
        .select("contact_id, name, title, company, email, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )
    rows = result.data or []
    if not rows:
        print(f"No contacts found for user {user_id}")
        return

    print(f"\n{len(rows)} contact(s) for user {user_id}:\n")
    for r in rows:
        print(f"  [{r['contact_id']}]")
        print(f"    {r.get('name')} — {r.get('title') or '—'} at {r.get('company') or '—'}")
        print(f"    {r.get('email') or '(no email)'}")
        print()


def cmd_delete(user_id: str, contact_id: str):
    supabase = get_supabase()
    result = (
        supabase.table("contacts")
        .select("id, name")
        .eq("user_id", user_id)
        .eq("contact_id", contact_id)
        .execute()
    )
    if not result.data:
        print(f"Contact '{contact_id}' not found for user {user_id}.")
        return

    name = result.data[0].get("name", "Unknown")
    supabase.table("contacts").delete().eq("user_id", user_id).eq("contact_id", contact_id).execute()
    print(f"✓ Deleted '{name}' ({contact_id}) from Supabase.")
    print(f"  Note: the Pinecone vector still exists. Remove it separately if needed.")


def cmd_diff(user_id: str, folder: str):
    """Show which contacts exist in the folder vs in Supabase for this user."""
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}")
        sys.exit(1)

    # Build set of contact_ids from folder
    folder_ids = {}
    for filepath in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        try:
            contact = parse_contact_file(filepath)
            if contact.get("name"):
                cid = create_contact_id(contact)
                folder_ids[cid] = os.path.basename(filepath)
        except Exception as e:
            print(f"  ⚠ Couldn't parse {filepath}: {e}")

    # Get contact_ids from Supabase for this user
    supabase = get_supabase()
    result = supabase.table("contacts").select("contact_id").eq("user_id", user_id).execute()
    db_ids = {r["contact_id"] for r in (result.data or [])}

    only_in_folder = set(folder_ids.keys()) - db_ids
    only_in_db = db_ids - set(folder_ids.keys())
    in_both = set(folder_ids.keys()) & db_ids

    print(f"\nDiff for user {user_id}:")
    print(f"  In both:       {len(in_both)}")
    print(f"  Only in folder (not yet synced to Supabase): {len(only_in_folder)}")
    for cid in sorted(only_in_folder):
        print(f"    - {cid} ({folder_ids[cid]})")
    print(f"  Only in Supabase (file removed or renamed):  {len(only_in_db)}")
    for cid in sorted(only_in_db):
        print(f"    - {cid}")

    if only_in_folder:
        print(f"\n  → run: python scripts/supabase_contacts.py sync {user_id} {folder}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    user_id = sys.argv[2]

    if command == "sync":
        if len(sys.argv) < 4:
            print("Usage: python scripts/supabase_contacts.py sync <user_id> <folder-or-file>")
            sys.exit(1)
        cmd_sync(user_id, sys.argv[3])
    elif command == "list":
        cmd_list(user_id)
    elif command == "delete":
        if len(sys.argv) < 4:
            print("Usage: python scripts/supabase_contacts.py delete <user_id> <contact_id>")
            sys.exit(1)
        cmd_delete(user_id, sys.argv[3])
    elif command == "diff":
        if len(sys.argv) < 4:
            print("Usage: python scripts/supabase_contacts.py diff <user_id> <folder>")
            sys.exit(1)
        cmd_diff(user_id, sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()