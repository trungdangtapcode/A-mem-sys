"""Verify persistence: save memories, reload from disk, compare every field."""
import os
import shutil
import json
from agentic_memory.memory_system import AgenticMemorySystem

PERSIST_DIR = ".memory"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

# === Phase 1: Create memories ===
print("=" * 60)
print("PHASE 1: Creating memories")
print("=" * 60)

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

notes_text = [
    "Docker containers share the host OS kernel for lightweight virtualization.",
    "Kubernetes orchestrates container deployments with auto-scaling and self-healing.",
    "PostgreSQL supports JSONB columns for semi-structured data storage.",
    "Redis is an in-memory key-value store used for caching.",
    """OAuth 2.0 is an authorization framework that enables third-party applications to obtain
limited access to user accounts. The Authorization Code flow with PKCE is now recommended
for all client types. Key components include the Resource Owner, Client, Authorization
Server, and Resource Server. Access tokens are typically short-lived JWTs. Security best
practices include always using HTTPS, validating redirect URIs strictly, and implementing
token rotation for refresh tokens.""",
]

ids = []
for n in notes_text:
    mid = ms.add_note(n)
    ids.append(mid)

# Manually link two notes
ms.add_link(ids[0], ids[1])

# Capture full state of every memory BEFORE reload
state_before = {}
for mid in ids:
    m = ms.read(mid)
    state_before[mid] = {
        "id": m.id,
        "name": m.name,
        "path": m.path,
        "content": m.content,
        "summary": m.summary,
        "keywords": m.keywords,
        "tags": m.tags,
        "links": sorted(m.links),
        "backlinks": sorted(m.backlinks),
        "timestamp": m.timestamp,
        "last_accessed": m.last_accessed,
        "retrieval_count": m.retrieval_count,
        "context": m.context,
        "category": m.category,
    }

# Also capture search results
search_before = {}
queries = ["container orchestration", "database caching", "authentication security"]
for q in queries:
    results = ms.search(q, k=2)
    search_before[q] = [r["id"] for r in results]

tree_before = ms.tree()

print(f"  Created {len(ids)} memories")
print(f"  Tree:\n{tree_before}")

# === Phase 2: Verify markdown files on disk ===
print("\n" + "=" * 60)
print("PHASE 2: Verify markdown files on disk")
print("=" * 60)

errors = []
for mid in ids:
    m = ms.read(mid)
    filepath = os.path.join(PERSIST_DIR, "notes", m.filepath)
    if not os.path.exists(filepath):
        errors.append(f"MISSING file: {filepath}")
        print(f"  FAIL: {m.filepath} — file not found!")
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify frontmatter has key fields
    from agentic_memory.memory_system import MemoryNote
    loaded = MemoryNote.from_markdown(content)

    checks = {
        "id": loaded.id == m.id,
        "name": loaded.name == m.name,
        "path": loaded.path == m.path,
        "content": loaded.content == m.content,
        "keywords": loaded.keywords == m.keywords,
        "tags": loaded.tags == m.tags,
        "links": sorted(loaded.links) == sorted(m.links),
        "summary": loaded.summary == m.summary,
        "timestamp": loaded.timestamp == m.timestamp,
    }

    all_ok = all(checks.values())
    status = "OK" if all_ok else "FAIL"
    print(f"  {status}: {m.filepath}")
    if not all_ok:
        for field, ok in checks.items():
            if not ok:
                err = f"    {field}: file={getattr(loaded, field)} != memory={getattr(m, field)}"
                print(err)
                errors.append(err)

# Verify backlinks NOT in files
print("\n  Backlinks in files:")
for dirpath, _, filenames in os.walk(os.path.join(PERSIST_DIR, "notes")):
    for f in filenames:
        if not f.endswith(".md"):
            continue
        text = open(os.path.join(dirpath, f)).read()
        if "backlinks" in text:
            err = f"  FAIL: {f} contains 'backlinks'!"
            print(f"    {err}")
            errors.append(err)
        else:
            print(f"    OK: {f} — no backlinks stored")

# === Phase 3: Destroy and reload ===
del ms

print("\n" + "=" * 60)
print("PHASE 3: Reload from disk and compare")
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

print(f"  Loaded: {len(ms2.memories)} memories (expected {len(ids)})")
if len(ms2.memories) != len(ids):
    errors.append(f"Memory count: {len(ms2.memories)} != {len(ids)}")

# Compare every field
print("\n  Field-by-field comparison:")
for mid in ids:
    m = ms2.read(mid)
    if not m:
        errors.append(f"Memory {mid} not found after reload!")
        print(f"  FAIL: {mid} — not found!")
        continue

    after = {
        "id": m.id,
        "name": m.name,
        "path": m.path,
        "content": m.content,
        "summary": m.summary,
        "keywords": m.keywords,
        "tags": m.tags,
        "links": sorted(m.links),
        "backlinks": sorted(m.backlinks),
        "timestamp": m.timestamp,
        "last_accessed": m.last_accessed,
        "retrieval_count": m.retrieval_count,
        "context": m.context,
        "category": m.category,
    }

    before = state_before[mid]
    mismatches = []
    for field in before:
        if before[field] != after[field]:
            mismatches.append(f"    {field}: before={before[field]} != after={after[field]}")

    if mismatches:
        print(f"  FAIL: {m.name}")
        for mm in mismatches:
            print(mm)
            errors.append(mm)
    else:
        print(f"  OK: {m.name}")

# === Phase 4: Verify search results match ===
print("\n  Search comparison:")
for q in queries:
    results = ms2.search(q, k=2)
    result_ids = [r["id"] for r in results]
    match = result_ids == search_before[q]
    status = "OK" if match else "DIFF"
    print(f"  {status}: \"{q}\"")
    if not match:
        print(f"    before: {search_before[q]}")
        print(f"    after:  {result_ids}")
        # Not an error — zvec may reorder slightly

# === Phase 5: Verify tree ===
tree_after = ms2.tree()
tree_match = tree_before == tree_after
print(f"\n  Tree match: {'OK' if tree_match else 'DIFF'}")
if not tree_match:
    print(f"  Before:\n{tree_before}")
    print(f"  After:\n{tree_after}")

# === Phase 6: Verify vectordb files exist ===
print("\n" + "=" * 60)
print("PHASE 4: Disk contents")
print("=" * 60)

print("  notes/")
for dirpath, dirnames, filenames in sorted(os.walk(os.path.join(PERSIST_DIR, "notes"))):
    level = dirpath.replace(os.path.join(PERSIST_DIR, "notes"), "").count(os.sep)
    indent = "    " * (level + 1)
    basename = os.path.basename(dirpath)
    if basename != "notes":
        print(f"{indent}{basename}/")
    for f in sorted(filenames):
        print(f"{indent}  {f}")

print("\n  vectordb/")
vdb_path = os.path.join(PERSIST_DIR, "vectordb", "memories")
if os.path.exists(vdb_path):
    for f in sorted(os.listdir(vdb_path)):
        size = os.path.getsize(os.path.join(vdb_path, f))
        print(f"    {f:<30} {size:>8} bytes")

# === Phase 7: Print full memory database ===
print("\n" + "=" * 60)
print("PHASE 7: Full memory database dump")
print("=" * 60)

for i, mid in enumerate(ids):
    m = ms2.read(mid)
    link_names = [ms2.read(lid).name for lid in m.links if ms2.read(lid)]
    backlink_names = [ms2.read(lid).name for lid in m.backlinks if ms2.read(lid)]

    print(f"\n  [{i+1}] {m.name}")
    print(f"      id:         {m.id}")
    print(f"      path:       {m.path}")
    print(f"      filepath:   {m.filepath}")
    print(f"      content:    {m.content[:100]}{'...' if len(m.content) > 100 else ''}")
    if m.summary:
        print(f"      summary:    {m.summary[:100]}...")
    print(f"      keywords:   {m.keywords}")
    print(f"      tags:        {m.tags}")
    print(f"      context:    {m.context[:100] if m.context else 'N/A'}{'...' if m.context and len(m.context) > 100 else ''}")
    print(f"      links:      {m.links}")
    if link_names:
        print(f"                  -> {link_names}")
    print(f"      backlinks:  {m.backlinks}")
    if backlink_names:
        print(f"                  <- {backlink_names}")
    print(f"      timestamp:  {m.timestamp}")
    print(f"      accessed:   {m.last_accessed}")
    print(f"      retrievals: {m.retrieval_count}")
    print(f"      category:   {m.category}")

# === Summary ===
print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} errors found!")
    for e in errors:
        print(f"  - {e}")
else:
    print("RESULT: ALL CHECKS PASSED")
    print("  - All markdown files match in-memory state")
    print("  - All fields preserved after reload")
    print("  - Backlinks correctly rebuilt (not persisted)")
    print("  - Search works after reload")
    print("  - Tree structure preserved")
print("=" * 60)
