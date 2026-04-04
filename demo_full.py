"""Full architecture test: diverse memories across many domains,
short + long notes, links, search, update, delete, persistence."""
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

# =====================================================================
# DIVERSE MEMORY NOTES — covering many domains, lengths, and languages
# =====================================================================

short_notes = [
    # DevOps / Infrastructure
    "Docker containers share the host OS kernel for lightweight virtualization.",
    "Kubernetes uses etcd as its distributed key-value store for cluster state.",
    "Terraform manages infrastructure as code using HCL declarative configuration files.",
    "GitHub Actions workflows are triggered by events like push, pull_request, or schedule.",

    # Backend / Database
    "Redis supports data structures like strings, hashes, lists, sets, and sorted sets.",
    "SQLite is a serverless embedded database that stores the entire DB in a single file.",
    "GraphQL lets clients request exactly the data they need, avoiding over-fetching.",

    # Frontend
    "React uses a virtual DOM to efficiently update only the changed parts of the UI.",
    "CSS Grid and Flexbox serve different layout purposes: Grid for 2D, Flexbox for 1D.",
    "Web Vitals metrics (LCP, FID, CLS) measure real-world user experience performance.",

    # Security
    "bcrypt hashes passwords with a configurable work factor to resist brute-force attacks.",
    "CORS (Cross-Origin Resource Sharing) controls which domains can access your API.",

    # Networking
    "DNS resolves domain names to IP addresses using a hierarchical caching system.",
    "TCP guarantees ordered delivery via sequence numbers and acknowledgments.",
    "WebSocket provides full-duplex communication over a single persistent TCP connection.",

    # Data Science / ML
    "Pandas DataFrame is the primary data structure for tabular data manipulation in Python.",
    "Gradient descent minimizes loss by iteratively adjusting model parameters.",
    "Feature engineering transforms raw data into meaningful inputs for ML models.",

    # Personal / Productivity
    "Pomodoro technique uses 25-minute focused work intervals with 5-minute breaks.",
    "Zettelkasten method links atomic notes to build a personal knowledge graph.",
]

long_notes = [
    # PostgreSQL deep dive
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

    # OAuth 2.0 deep dive
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
and implementing proper CORS policies.""",

    # Transformer architecture
    """The Transformer architecture, introduced in the paper "Attention Is All You Need" by
Vaswani et al. in 2017, revolutionized natural language processing by replacing recurrent
neural networks with self-attention mechanisms. The architecture consists of an encoder-decoder
structure where both components are built from stacked layers of multi-head self-attention
and position-wise feed-forward networks. The self-attention mechanism computes attention
scores using Query, Key, and Value matrices derived from the input embeddings, allowing
each token to attend to all other tokens in the sequence simultaneously. Multi-head attention
runs multiple attention operations in parallel, enabling the model to capture different types
of relationships. Positional encoding is added to input embeddings since the architecture
has no inherent notion of sequence order. Key variants include BERT (encoder-only,
bidirectional, used for understanding tasks), GPT (decoder-only, autoregressive, used for
generation), T5 (encoder-decoder, text-to-text framework), and Vision Transformer (ViT)
which adapts the architecture for image classification by treating image patches as tokens.
Recent developments include sparse attention patterns for longer sequences, mixture of experts
for scaling model capacity, and rotary position embeddings (RoPE) for efficiency.""",

    # Microservices architecture
    """Microservices architecture decomposes a monolithic application into small, independently
deployable services that communicate via APIs. Each service owns its data store (database per
service pattern), enabling polyglot persistence where different services use different databases
best suited to their needs. Service discovery mechanisms like Consul or Kubernetes DNS allow
services to find each other dynamically. API gateways (Kong, AWS API Gateway) handle cross-
cutting concerns like rate limiting, authentication, and request routing. Circuit breaker
pattern (implemented via libraries like Resilience4j or Hystrix) prevents cascade failures
when a downstream service is unavailable. Saga pattern manages distributed transactions
across services using either choreography (event-driven) or orchestration (central coordinator).
Observability is critical: distributed tracing (Jaeger, Zipkin), centralized logging (ELK
stack), and metrics (Prometheus + Grafana) provide visibility into system behavior. Event-
driven communication via message brokers (Kafka, RabbitMQ) enables loose coupling and
eventual consistency. Key challenges include network latency, data consistency, debugging
distributed systems, and operational complexity of managing many deployments.""",

    # RAG systems
    """Retrieval-Augmented Generation (RAG) combines vector search with LLM generation to
produce grounded, factual responses. The pipeline consists of three stages: indexing (chunk
documents, generate embeddings, store in vector DB), retrieval (embed the query, find top-k
similar chunks), and generation (feed retrieved context + query to LLM). Chunking strategies
significantly impact quality: fixed-size chunks are simple but may split sentences, semantic
chunking preserves meaning but is computationally expensive, and recursive character splitting
offers a balance. Embedding models like OpenAI text-embedding-3, Cohere embed-v3, or open-
source alternatives like BGE and E5 convert text to dense vectors. Vector databases (Pinecone,
Weaviate, Qdrant, ChromaDB, Zvec) store and index these vectors for fast similarity search.
Advanced techniques include hybrid search (combining dense vectors with sparse BM25), re-
ranking retrieved results with cross-encoder models, query decomposition for complex questions,
HyDE (Hypothetical Document Embeddings) for improved retrieval, and multi-hop retrieval for
questions requiring reasoning across multiple documents. Evaluation metrics include faithfulness
(is the answer supported by retrieved context?), relevance (are the retrieved documents
useful?), and answer correctness. Common failure modes include retrieval missing relevant
chunks, LLM hallucinating despite correct context, and context window overflow.""",
]

# =====================================================================
# PHASE 1: Add all notes
# =====================================================================
print("=" * 70)
print("PHASE 1: ADDING SHORT NOTES (20 notes)")
print("=" * 70)

short_ids = []
for note in short_notes:
    mid = ms.add_note(note)
    short_ids.append(mid)
    m = ms.read(mid)
    print(f"  {m.path:<35} {m.name}")

print(f"\n{'=' * 70}")
print(f"PHASE 2: ADDING LONG NOTES (5 notes)")
print("=" * 70)

long_ids = []
for note in long_notes:
    mid = ms.add_note(note)
    long_ids.append(mid)
    m = ms.read(mid)
    words = len(m.content.split())
    has_sum = "summary" if m.summary else "no summary"
    print(f"  {m.path:<35} {m.name} ({words}w, {has_sum})")

all_ids = short_ids + long_ids

# =====================================================================
# PHASE 3: Directory tree
# =====================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: DIRECTORY TREE")
print("=" * 70)
print(ms.tree())

# =====================================================================
# PHASE 4: Graph
# =====================================================================
print(f"\n{'=' * 70}")
print("PHASE 4: KNOWLEDGE GRAPH")
print("=" * 70)

total_links = 0
total_backlinks = 0
for mid in all_ids:
    m = ms.read(mid)
    total_links += len(m.links)
    total_backlinks += len(m.backlinks)
    link_names = [ms.read(lid).name for lid in m.links if ms.read(lid)]
    backlink_names = [ms.read(lid).name for lid in m.backlinks if ms.read(lid)]
    if link_names or backlink_names:
        print(f"  {m.name}")
        if link_names:
            print(f"    -> {link_names}")
        if backlink_names:
            print(f"    <- {backlink_names}")

print(f"\n  Total: {total_links} forward links, {total_backlinks} backlinks")

# =====================================================================
# PHASE 5: Search across domains
# =====================================================================
print(f"\n{'=' * 70}")
print("PHASE 5: CROSS-DOMAIN SEARCH")
print("=" * 70)

queries = [
    # Should find database notes
    "database indexing and query optimization",
    # Should find security notes
    "authentication tokens and password security",
    # Should find ML notes
    "neural network attention mechanism",
    # Should find DevOps notes
    "container orchestration and deployment pipeline",
    # Should find frontend notes
    "UI rendering performance metrics",
    # Should find networking notes
    "network protocols and DNS resolution",
    # Should find RAG note
    "vector search embedding retrieval augmented generation",
    # Should find microservices note
    "distributed systems circuit breaker saga pattern",
    # Should find productivity notes
    "personal knowledge management and focus techniques",
    # Cross-domain: should find relevant from multiple domains
    "how to build a scalable web application",
]

for q in queries:
    results = ms.search(q, k=3)
    print(f'\n  "{q}"')
    for i, r in enumerate(results):
        m = ms.read(r['id'])
        kind = "summary" if m.summary else "direct"
        print(f"    [{i+1}] {m.name} ({m.path}, {kind})")

# =====================================================================
# PHASE 6: Update and Delete
# =====================================================================
print(f"\n{'=' * 70}")
print("PHASE 6: UPDATE AND DELETE")
print("=" * 70)

# Update a short note to a completely different topic
target_id = short_ids[5]  # SQLite note
old = ms.read(target_id)
print(f"  UPDATE: '{old.name}' ({old.path})")
ms.update(target_id, content="DynamoDB is a fully managed NoSQL database by AWS with single-digit millisecond latency at any scale, using partition keys for data distribution.")
new = ms.read(target_id)
print(f"    -> '{new.name}' ({new.path})")
print(f"    Old file exists: {os.path.exists(os.path.join(PERSIST_DIR, 'notes', old.filepath))}")
print(f"    New file exists: {os.path.exists(os.path.join(PERSIST_DIR, 'notes', new.filepath))}")

# Delete a note
del_id = short_ids[10]  # bcrypt note
del_note = ms.read(del_id)
print(f"\n  DELETE: '{del_note.name}' ({del_note.path})")
ms.delete(del_id)
print(f"    File exists: {os.path.exists(os.path.join(PERSIST_DIR, 'notes', del_note.filepath))}")
print(f"    Memory exists: {ms.read(del_id) is not None}")
# Verify no dangling references
for mid in all_ids:
    m = ms.read(mid)
    if m:
        assert del_id not in m.links, f"Dangling link in {m.name}"
        assert del_id not in m.backlinks, f"Dangling backlink in {m.name}"
print(f"    No dangling links: OK")

# =====================================================================
# PHASE 7: Persistence
# =====================================================================
count_before = len(ms.memories)
del ms

print(f"\n{'=' * 70}")
print("PHASE 7: PERSISTENCE")
print("=" * 70)

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

# Verify search still works across all domains
spot_checks = [
    ("transformer attention", "ml"),
    ("kubernetes cluster", "devops"),
    ("OAuth tokens", "security"),
    ("React virtual DOM", "frontend"),
    ("vector database RAG", "rag"),
]
print("\n  Spot checks after reload:")
for query, domain in spot_checks:
    results = ms2.search(query, k=1)
    name = ms2.read(results[0]['id']).name if results else "NO RESULT"
    print(f"    [{domain}] \"{query}\" -> {name}")

# =====================================================================
# PHASE 8: Final tree
# =====================================================================
print(f"\n{'=' * 70}")
print("PHASE 8: FINAL TREE ({} memories)".format(len(ms2.memories)))
print("=" * 70)
print(ms2.tree())

# =====================================================================
# PHASE 9: Stats
# =====================================================================
print(f"\n{'=' * 70}")
print("PHASE 9: STATS")
print("=" * 70)
paths = sorted(set(m.path for m in ms2.memories.values() if m.path))
total_links = sum(len(m.links) for m in ms2.memories.values())
total_backlinks = sum(len(m.backlinks) for m in ms2.memories.values())
notes_with_summary = sum(1 for m in ms2.memories.values() if m.summary)
notes_without_summary = sum(1 for m in ms2.memories.values() if not m.summary)

print(f"  Total memories:       {len(ms2.memories)}")
print(f"  Unique paths:         {len(paths)}")
print(f"  Total forward links:  {total_links}")
print(f"  Total backlinks:      {total_backlinks}")
print(f"  Notes with summary:   {notes_with_summary}")
print(f"  Notes without:        {notes_without_summary}")
print(f"  Top-level categories: {sorted(set(p.split('/')[0] for p in paths))}")

print("\nDone!")
