"""Test zvec backend: add, search, persistence, tree."""
import os
import shutil
from agentic_memory.memory_system import AgenticMemorySystem

PERSIST_DIR = ".memory"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

ms = AgenticMemorySystem(
    model_name='gemini-embedding-001',
    embedding_backend='gemini',
    vector_backend='zvec',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR,
    context_aware_analysis=True,
    max_links=2,
)

# === Short notes ===
short_notes = [
    "Docker containers share the host OS kernel for lightweight virtualization.",
    "Kubernetes orchestrates container deployments with auto-scaling and self-healing.",
    "PostgreSQL supports JSONB columns for semi-structured data storage.",
    "Redis is an in-memory key-value store used for caching.",
    "OAuth 2.0 middleware handles token validation and user authentication.",
]

# === Long note ===
long_note = """Transformer architecture introduced in "Attention Is All You Need" by Vaswani et al.
revolutionized NLP by replacing RNNs with self-attention mechanisms. The architecture uses
encoder-decoder structure with multi-head attention and position-wise feed-forward networks.
Self-attention computes Query, Key, Value matrices allowing each token to attend to all others.
Multi-head attention runs parallel attention ops to capture different relationship types.
Positional encoding is added since the architecture has no inherent sequence order notion.
Key variants include BERT (encoder-only, bidirectional), GPT (decoder-only, autoregressive),
T5 (encoder-decoder, text-to-text), and ViT (vision, image patches as tokens)."""

print("=" * 60)
print("ADDING NOTES (zvec backend)")
print("=" * 60)

ids = []
for n in short_notes:
    mid = ms.add_note(n)
    ids.append(mid)
    m = ms.read(mid)
    print(f"  {m.filepath:<55} links={len(m.links)} backlinks={len(m.backlinks)}")

mid_long = ms.add_note(long_note.strip())
ids.append(mid_long)
m = ms.read(mid_long)
print(f"  {m.filepath:<55} links={len(m.links)} summary={'yes' if m.summary else 'no'}")

print(f"\n{'=' * 60}")
print("TREE")
print("=" * 60)
print(ms.tree())

print(f"\n{'=' * 60}")
print("SEARCH")
print("=" * 60)
queries = [
    "container orchestration",
    "database storage and caching",
    "attention mechanism neural networks",
    "authentication security",
]
for q in queries:
    results = ms.search(q, k=2)
    print(f'\n  "{q}"')
    for r in results:
        note = ms.read(r["id"])
        print(f"    -> {note.name} ({note.path})")

print(f"\n{'=' * 60}")
print("GRAPH")
print("=" * 60)
for mid in ids:
    m = ms.read(mid)
    link_names = [ms.read(lid).name for lid in m.links if ms.read(lid)]
    backlink_names = [ms.read(lid).name for lid in m.backlinks if ms.read(lid)]
    print(f"  {m.name}")
    if link_names:
        print(f"    -> {link_names}")
    if backlink_names:
        print(f"    <- {backlink_names}")

# === Persistence ===
count_before = len(ms.memories)
del ms

print(f"\n{'=' * 60}")
print("PERSISTENCE (reload)")
print("=" * 60)

ms2 = AgenticMemorySystem(
    model_name='gemini-embedding-001',
    embedding_backend='gemini',
    vector_backend='zvec',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR,
    context_aware_analysis=True,
)
print(f"  Before: {count_before} memories")
print(f"  After reload: {len(ms2.memories)} memories")
assert count_before == len(ms2.memories), "Memory count mismatch!"

results = ms2.search("container orchestration", k=1)
print(f"  Search after reload: {ms2.read(results[0]['id']).name}")

# === Vectordb files ===
print(f"\n{'=' * 60}")
print("VECTORDB FILES")
print("=" * 60)
vdb_path = os.path.join(PERSIST_DIR, "vectordb", "memories")
if os.path.exists(vdb_path):
    for f in sorted(os.listdir(vdb_path)):
        size = os.path.getsize(os.path.join(vdb_path, f))
        print(f"  {f:<30} {size:>8} bytes")

print("\nDone!")
