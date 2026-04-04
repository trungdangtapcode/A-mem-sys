"""Test persistence: save notes, reload from disk, verify search works."""
import os
import shutil
from agentic_memory.memory_system import AgenticMemorySystem

PERSIST_DIR = ".memory"

# Clean up from previous runs
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

# === Phase 1: Create and save memories ===
print("=" * 60)
print("PHASE 1: Creating memories with persistence")
print("=" * 60)

ms = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR
)

notes = [
    "Docker containers provide lightweight virtualization by sharing the host OS kernel.",
    "PostgreSQL supports JSONB columns for storing and querying semi-structured data.",
    "Redis is an in-memory data store used as cache, message broker, and session store.",
    "Kubernetes orchestrates container deployments with auto-scaling and self-healing.",
    "Git rebase rewrites commit history by replaying commits on top of another branch.",
]

ids = []
for note in notes:
    mid = ms.add_note(note)
    ids.append(mid)
    print(f"  Added: {note[:60]}...")

print(f"\nTotal memories: {len(ms.memories)}")
print(f"Saved IDs: {ids}")

# Show what's on disk
print(f"\n--- Files in {PERSIST_DIR}/notes/ ---")
for f in sorted(os.listdir(os.path.join(PERSIST_DIR, "notes"))):
    print(f"  {f}")

# Show a sample markdown file
sample_file = os.path.join(PERSIST_DIR, "notes", f"{ids[0]}.md")
print(f"\n--- Sample note ({ids[0]}.md) ---")
with open(sample_file, "r") as f:
    print(f.read())

# Delete the memory system object (simulate app exit)
del ms

# === Phase 2: Reload from disk ===
print("=" * 60)
print("PHASE 2: Reloading from disk (new instance)")
print("=" * 60)

ms2 = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR
)

print(f"Loaded memories: {len(ms2.memories)}")

# Verify all notes loaded correctly
for mid in ids:
    mem = ms2.read(mid)
    if mem:
        print(f"  [{mid[:8]}] {mem.content[:60]}... tags={mem.tags[:3]}")
    else:
        print(f"  [{mid[:8]}] MISSING!")

# === Phase 3: Search on reloaded data ===
print(f"\n--- Search tests ---")
queries = ["container orchestration", "database storage", "version control"]
for q in queries:
    print(f"\nQuery: \"{q}\"")
    results = ms2.search(q, k=2)
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['content'][:70]}...")

# === Phase 4: Add more, delete one, verify ===
print(f"\n--- Add + Delete test ---")
new_id = ms2.add_note("Nginx reverse proxy handles SSL termination and load balancing.")
print(f"Added new note: {new_id[:8]}...")

ms2.delete(ids[0])
print(f"Deleted note: {ids[0][:8]}...")

print(f"Total memories after add+delete: {len(ms2.memories)}")
print(f"Deleted note file exists: {os.path.exists(os.path.join(PERSIST_DIR, 'notes', f'{ids[0]}.md'))}")
print(f"New note file exists: {os.path.exists(os.path.join(PERSIST_DIR, 'notes', f'{new_id}.md'))}")

# === Phase 5: Final reload to verify add+delete persisted ===
del ms2

ms3 = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR
)

print(f"\n--- Final reload ---")
print(f"Total memories: {len(ms3.memories)}")
print(f"Deleted note still exists: {ms3.read(ids[0]) is not None}")
print(f"New note exists: {ms3.read(new_id) is not None}")

print("\nDone!")
