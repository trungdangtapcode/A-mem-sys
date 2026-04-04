"""Full test: short notes, long notes, links, search, persistence, update, delete."""
import os
import shutil
from agentic_memory.memory_system import AgenticMemorySystem

PERSIST_DIR = ".memory"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

ms = AgenticMemorySystem(
    model_name='gemini-embedding-001',
    embedding_backend='gemini',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR
)

# === Short notes ===
short_notes = [
    "Docker containers share the host OS kernel for lightweight virtualization.",
    "Kubernetes orchestrates container deployments with auto-scaling and self-healing.",
    "Redis is an in-memory data store used as cache, message broker, and session store.",
    "Nginx reverse proxy handles SSL termination and load balancing for microservices.",
    "Git rebase rewrites commit history by replaying commits on top of another branch.",
]

# === Long notes ===
long_notes = [
    """PostgreSQL is a powerful open-source relational database management system known for
its robustness, extensibility, and standards compliance. It supports a wide range of data
types including JSON, JSONB, arrays, hstore, and custom types. PostgreSQL's MVCC
(Multi-Version Concurrency Control) architecture provides excellent concurrent access
without read locks. Key features include table partitioning for managing large datasets,
full-text search with tsvector and tsquery, window functions for complex analytical queries,
Common Table Expressions (CTEs) for readable recursive queries, and materialized views for
caching expensive query results. The Write-Ahead Logging (WAL) system ensures data integrity
and enables point-in-time recovery. PostgreSQL also supports logical replication for
selective data distribution, foreign data wrappers for querying external data sources,
and extensions like PostGIS for geospatial data, pg_trgm for fuzzy text matching,
and TimescaleDB for time-series workloads. Its advanced indexing strategies include
B-tree, Hash, GIN (Generalized Inverted Index), GiST (Generalized Search Tree),
SP-GiST, and BRIN (Block Range Index) for different query patterns.""",

    """OAuth 2.0 is an authorization framework that enables third-party applications to obtain
limited access to user accounts on an HTTP service. The framework defines four grant types:
Authorization Code (for server-side apps), Implicit (deprecated, was for SPAs), Resource
Owner Password Credentials (for trusted apps), and Client Credentials (for machine-to-machine).
The Authorization Code flow with PKCE (Proof Key for Code Exchange) is now recommended for
all client types including single-page applications and mobile apps. Key components include
the Resource Owner (user), Client (application), Authorization Server (issues tokens), and
Resource Server (hosts protected resources). Access tokens are typically short-lived JWTs
containing claims about the user and their permissions, while refresh tokens are long-lived
and used to obtain new access tokens without user interaction. OpenID Connect (OIDC) extends
OAuth 2.0 by adding an identity layer, introducing the ID token which contains user profile
information. Security best practices include always using HTTPS, validating redirect URIs
strictly, implementing token rotation for refresh tokens, using short-lived access tokens,
storing tokens securely (HttpOnly cookies for web apps, secure storage for mobile),
and implementing proper CORS policies. Common vulnerabilities include token leakage through
referrer headers, CSRF attacks on the authorization endpoint, and authorization code
injection attacks.""",

    """The Transformer architecture, introduced in the paper "Attention Is All You Need" by
Vaswani et al. in 2017, revolutionized natural language processing by replacing recurrent
neural networks with self-attention mechanisms. The architecture consists of an encoder-decoder
structure where both components are built from stacked layers of multi-head self-attention
and position-wise feed-forward networks. The self-attention mechanism computes attention
scores using Query, Key, and Value matrices derived from the input embeddings, allowing
each token to attend to all other tokens in the sequence simultaneously. Multi-head attention
runs multiple attention operations in parallel, enabling the model to capture different types
of relationships. Positional encoding is added to input embeddings since the architecture
has no inherent notion of sequence order. The encoder processes the entire input sequence
in parallel, while the decoder generates output tokens autoregressively with masked
self-attention to prevent attending to future tokens. Key variants include BERT (encoder-only,
bidirectional, used for understanding tasks), GPT (decoder-only, autoregressive, used for
generation), T5 (encoder-decoder, text-to-text framework), and Vision Transformer (ViT)
which adapts the architecture for image classification by treating image patches as tokens.
Recent developments include sparse attention patterns for longer sequences, mixture of experts
for scaling model capacity, and architectural innovations like rotary position embeddings
(RoPE) and grouped query attention (GQA) for efficiency.""",
]

print("=" * 60)
print("ADDING SHORT NOTES")
print("=" * 60)
short_ids = []
for note in short_notes:
    mid = ms.add_note(note)
    short_ids.append(mid)
    m = ms.read(mid)
    print(f"  [{m.filepath}]")
    print(f"    summary: {m.summary}")
    print(f"    links: {len(m.links)}  backlinks: {len(m.backlinks)}")

print()
print("=" * 60)
print("ADDING LONG NOTES")
print("=" * 60)
long_ids = []
for note in long_notes:
    mid = ms.add_note(note)
    long_ids.append(mid)
    m = ms.read(mid)
    print(f"  [{m.filepath}]")
    print(f"    content: {len(m.content.split())} words")
    print(f"    summary: {m.summary[:100] if m.summary else 'None'}...")
    print(f"    links: {len(m.links)}  backlinks: {len(m.backlinks)}")

# === Graph ===
print()
print("=" * 60)
print("GRAPH (links + backlinks)")
print("=" * 60)
all_ids = short_ids + long_ids
for mid in all_ids:
    m = ms.read(mid)
    link_names = [ms.read(lid).name for lid in m.links if ms.read(lid)]
    backlink_names = [ms.read(lid).name for lid in m.backlinks if ms.read(lid)]
    print(f"  {m.name}")
    if link_names:
        print(f"    -> links to: {link_names}")
    if backlink_names:
        print(f"    <- linked from: {backlink_names}")

# === Search ===
print()
print("=" * 60)
print("SEARCH TESTS")
print("=" * 60)
queries = [
    "database indexing strategies",
    "token authentication security",
    "attention mechanism in neural networks",
    "container orchestration",
    "version control",
    "caching and performance",
    "SSL and load balancing",
]
for q in queries:
    results = ms.search(q, k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results):
        m = ms.read(r['id'])
        has_summary = "summary" if m.summary else "direct"
        print(f"    [{i+1}] {m.name} ({has_summary}) - {m.filepath}")

# === Update long note ===
print()
print("=" * 60)
print("UPDATE TEST")
print("=" * 60)
old = ms.read(long_ids[0])
print(f"  BEFORE: name={old.name}  path={old.path}")
ms.update(long_ids[0], content="MongoDB is a NoSQL document database that stores data in flexible JSON-like BSON format with dynamic schemas.")
new = ms.read(long_ids[0])
print(f"  AFTER:  name={new.name}  path={new.path}  summary={new.summary}")

# === Persistence ===
print()
print("=" * 60)
print("PERSISTENCE TEST")
print("=" * 60)
count_before = len(ms.memories)
del ms

ms2 = AgenticMemorySystem(
    model_name='gemini-embedding-001',
    embedding_backend='gemini',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview',
    persist_dir=PERSIST_DIR
)
print(f"  Before: {count_before} memories")
print(f"  After reload: {len(ms2.memories)} memories")
assert count_before == len(ms2.memories), "Memory count mismatch!"

# Search after reload
results = ms2.search("attention mechanism", k=1)
print(f"  Search after reload: {ms2.read(results[0]['id']).name}")

# === Directory tree ===
print()
print("=" * 60)
print("DIRECTORY TREE")
print("=" * 60)
for dirpath, dirnames, filenames in sorted(os.walk(os.path.join(PERSIST_DIR, 'notes'))):
    level = dirpath.replace(os.path.join(PERSIST_DIR, 'notes'), '').count(os.sep)
    indent = '  ' * level
    basename = os.path.basename(dirpath)
    if basename != 'notes':
        print(f'{indent}{basename}/')
    for f in sorted(filenames):
        print(f'{indent}  {f}')

print("\nDone!")
