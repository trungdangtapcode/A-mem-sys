"""MCP Server for Agentic Memory System.

Exposes the memory system as MCP tools for LLM agents.
Run with: python mcp_server.py
"""
import json
import logging
import os
import re
import subprocess
import threading
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

from agentic_memory.memory_system import AgenticMemorySystem

# --- Configuration via environment variables ---
PERSIST_DIR = os.getenv("MEMORY_PERSIST_DIR", ".memory")
LOG_DIR = os.getenv("MEMORY_LOG_DIR", PERSIST_DIR)

# --- Logging setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mcp_server.log")),
    ],
)
logger = logging.getLogger(__name__)
LLM_BACKEND = os.getenv("MEMORY_LLM_BACKEND", "gemini")
LLM_MODEL = os.getenv("MEMORY_LLM_MODEL", "gemini-3-flash-preview")
EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_BACKEND = os.getenv("MEMORY_EMBEDDING_BACKEND", "gemini")
VECTOR_BACKEND = os.getenv("MEMORY_VECTOR_BACKEND", "zvec")
CONTEXT_AWARE = os.getenv("MEMORY_CONTEXT_AWARE", "true").lower() == "true"
CONTEXT_AWARE_TREE = os.getenv("MEMORY_CONTEXT_AWARE_TREE", "false").lower() == "true"
MAX_LINKS = int(os.getenv("MEMORY_MAX_LINKS", "3"))
AUTO_SYNC_ENABLED = os.getenv("MEMORY_AUTO_SYNC", "true").lower() == "true"
AUTO_SYNC_INTERVAL = int(os.getenv("MEMORY_AUTO_SYNC_INTERVAL", "60"))  # seconds

# Comma-separated list of tools to disable (hide from MCP clients).
# Example: MEMORY_DISABLED_TOOLS="unlink_memories,sync_to_disk"
# To re-enable all: MEMORY_DISABLED_TOOLS=""
DISABLED_TOOLS = set(
    t.strip() for t in os.getenv(
        "MEMORY_DISABLED_TOOLS",
        "read_memory,update_memory,delete_memory,link_memories,unlink_memories,memory_stats,sync_from_disk,sync_to_disk,graph_snapshot"
    ).split(",") if t.strip()
)


def tool_enabled(name: str) -> bool:
    """Check if a tool is enabled (not in the disabled list)."""
    return name not in DISABLED_TOOLS


def optional_tool(name: str):
    """Register an MCP tool only if it's not in DISABLED_TOOLS."""
    if tool_enabled(name):
        return mcp.tool()
    # Return a no-op decorator that keeps the function but doesn't register it
    def noop(func):
        return func
    return noop

logger.info("=" * 60)
logger.info("MCP Server starting up")
logger.info("persist_dir=%s  llm=%s/%s  embedding=%s/%s  vector=%s",
            PERSIST_DIR, LLM_BACKEND, LLM_MODEL, EMBEDDING_BACKEND, EMBEDDING_MODEL, VECTOR_BACKEND)
logger.info("auto_sync=%s  interval=%ds", AUTO_SYNC_ENABLED, AUTO_SYNC_INTERVAL)

# --- Initialize memory system ---
memory = AgenticMemorySystem(
    model_name=EMBEDDING_MODEL,
    embedding_backend=EMBEDDING_BACKEND,
    vector_backend=VECTOR_BACKEND,
    llm_backend=LLM_BACKEND,
    llm_model=LLM_MODEL,
    persist_dir=PERSIST_DIR,
    context_aware_analysis=CONTEXT_AWARE,
    context_aware_tree=CONTEXT_AWARE_TREE,
    max_links=MAX_LINKS,
)
logger.info("Memory system initialized: %d memories loaded", len(memory.memories))

GRAPH_GROUP_COLORS = [
    "#8b5cf6",
    "#facc15",
    "#2563eb",
    "#d4d4d8",
    "#16a34a",
    "#ff5d5d",
    "#14b8a6",
    "#f97316",
]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", value.lower().strip())
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-") or "unfiled"


def _note_title(note) -> str:
    if note.name:
        return note.name
    return note.filename.replace("-", " ")


def _note_excerpt(note, max_length: int = 220) -> str:
    source = (note.summary or note.content or "").strip().replace("\n", " ")
    source = re.sub(r"\s+", " ", source)
    if len(source) <= max_length:
        return source
    return source[: max_length - 1].rstrip() + "…"


def _note_group_key(note) -> str:
    if note.path:
        return note.path.split("/")[0]
    if note.tags:
        return note.tags[0].lstrip("#")
    return "unfiled"


def _group_query(note, group_key: str) -> str:
    if note.path:
        return f'path:"{group_key}/"'
    if note.tags:
        return f"tag:{note.tags[0]}"
    return 'path:"unfiled/"'


def _build_graph_snapshot() -> dict[str, Any]:
    notes = sorted(
        memory.memories.values(),
        key=lambda note: ((note.path or ""), _note_title(note), note.id),
    )

    if not notes:
        return {
            "groups": [],
            "nodes": [],
            "links": [],
            "stats": {
                "total_memories": 0,
                "total_links": 0,
                "persist_dir": PERSIST_DIR,
                "transport": "stdio",
            },
        }

    group_order: list[str] = []
    group_meta: dict[str, dict[str, Any]] = {}

    for note in notes:
        raw_group_key = _note_group_key(note)
        group_id = _slugify(raw_group_key)
        if group_id not in group_meta:
            color = GRAPH_GROUP_COLORS[len(group_meta) % len(GRAPH_GROUP_COLORS)]
            label = raw_group_key.replace("-", " ").replace("_", " ").title()
            group_meta[group_id] = {
                "id": group_id,
                "label": label,
                "query": _group_query(note, raw_group_key),
                "color": color,
                "pathPrefix": raw_group_key,
                "tag": note.tags[0] if note.tags else f"#{group_id}",
                "description": f"Notes grouped under {label.lower()}",
                "_count": 0,
            }
            group_order.append(group_id)
        group_meta[group_id]["_count"] += 1

    links: list[dict[str, Any]] = []
    link_pairs: set[tuple[str, str]] = set()
    total_forward_links = 0

    for note in notes:
        for target_id in note.links:
            if target_id not in memory.memories:
                continue
            total_forward_links += 1
            pair = (note.id, target_id)
            if pair in link_pairs:
                continue
            link_pairs.add(pair)
            target = memory.memories[target_id]
            overlap = len(set(note.tags) & set(target.tags))
            strength = 0.38 + min(0.5, overlap * 0.08 + len(target.backlinks) * 0.02)
            links.append({
                "source": note.id,
                "target": target_id,
                "strength": round(strength, 2),
            })

    nodes: list[dict[str, Any]] = []
    for note in notes:
        group_id = _slugify(_note_group_key(note))
        color = group_meta[group_id]["color"]
        connections = len(set(note.links + note.backlinks))
        weight = round(1.0 + min(2.4, connections * 0.25 + note.retrieval_count * 0.03), 2)
        nodes.append({
            "id": note.id,
            "title": _note_title(note),
            "path": note.filepath,
            "pathLabel": note.path,
            "excerpt": _note_excerpt(note),
            "content": note.content,
            "summary": note.summary,
            "keywords": note.keywords,
            "tags": note.tags,
            "groupId": group_id,
            "color": color,
            "weight": weight,
            "connections": connections,
            "timestamp": note.timestamp,
            "retrievalCount": note.retrieval_count,
        })

    groups = []
    for group_id in group_order:
        group = dict(group_meta[group_id])
        group["description"] = f"{group['_count']} notes in {group['label'].lower()}"
        del group["_count"]
        groups.append(group)

    return {
        "groups": groups,
        "nodes": nodes,
        "links": links,
        "stats": {
            "total_memories": len(nodes),
            "total_links": total_forward_links,
            "persist_dir": PERSIST_DIR,
            "transport": "stdio",
        },
    }

# --- MCP Server ---
mcp = FastMCP(
    "Agentic Memory",
    instructions="""You have access to a persistent memory system that stores knowledge as
interconnected notes organized in a directory tree. Use it to remember important information,
decisions, context, and learnings across conversations.

IMPORTANT GUIDELINES FOR WRITING MEMORIES:
- Write DETAILED, RICH memories. Don't just save a one-liner — include context, reasoning,
  examples, and nuance. A good memory is 3-10 sentences that capture the full picture.
- Think of each memory as a knowledge article that your future self will read. Include
  the WHY, not just the WHAT.
- Good memory: "PostgreSQL's JSONB type stores semi-structured data with full indexing
  support via GIN indexes. We chose it over MongoDB because our data has relational
  aspects (user->orders->items) but product attributes vary per category. The GIN index
  on product.attributes reduced our catalog search from 800ms to 12ms. Key gotcha:
  JSONB equality checks are exact-match, so normalize data before insertion."
- Bad memory: "Use PostgreSQL JSONB for product data."

The system automatically generates keywords, tags, directory paths, and links between
related memories. You can search by natural language — the richer your memories, the
better search results you'll get."""
)


@optional_tool("save_memory")
def save_memory(
    content: str,
    name: Optional[str] = None,
    path: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """Save a new memory note to the knowledge base.

    Write detailed, comprehensive memories — not just brief notes. Include context,
    reasoning, examples, trade-offs, and lessons learned. The more detail you provide,
    the more useful the memory will be when retrieved later.

    Good memories are 3-10 sentences and capture:
    - WHAT happened or was decided
    - WHY it matters or was chosen over alternatives
    - HOW it works in practice, with specific details
    - GOTCHAS or edge cases discovered

    The system will automatically:
    - Generate a name and directory path if not provided
    - Extract keywords and tags for semantic search
    - Find and link related existing memories
    - Create a summary for long content (>150 words)

    Args:
        content: The memory content. Be detailed and thorough — include context,
            reasoning, specific examples, and lessons learned. Aim for 3-10 sentences
            minimum. Raw facts without context are far less useful when retrieved later.
        name: Optional human-readable name (2-5 words). Auto-generated if not provided.
        path: Optional directory path (e.g. "backend/database", "devops/ci-cd").
            Auto-generated if not provided.
        tags: Optional list of tags. Auto-generated if not provided.

    Returns:
        JSON with the saved memory's id, name, path, and generated metadata.
    """
    kwargs = {}
    if name:
        kwargs["name"] = name
    if path:
        kwargs["path"] = path
    if tags:
        kwargs["tags"] = tags

    logger.info("save_memory: name=%s path=%s tags=%s content_len=%d", name, path, tags, len(content))
    mid = memory.add_note(content, **kwargs)
    note = memory.read(mid)
    logger.info("save_memory: saved id=%s name=%s path=%s", note.id, note.name, note.path)

    return json.dumps({
        "id": note.id,
        "name": note.name,
        "path": note.path,
        "filepath": note.filepath,
        "keywords": note.keywords,
        "tags": note.tags,
        "links": note.links,
        "has_summary": note.summary is not None,
    }, ensure_ascii=False)


@optional_tool("search_memory")
def search_memory(query: str, k: int = 5) -> str:
    """Search the knowledge base using natural language.

    Returns the most semantically relevant memories for the query. Results include
    full content, metadata, and relationship information.

    Use specific, descriptive queries for best results:
    - Good: "PostgreSQL indexing strategies for JSON data"
    - Bad: "database"

    Args:
        query: Natural language search query. Be specific and descriptive.
        k: Maximum number of results to return (default: 5).

    Returns:
        JSON array of matching memories with content, tags, path, and links.
    """
    logger.info("search_memory: query=%r k=%d", query, k)
    results = memory.search(query, k=k)
    logger.info("search_memory: returned %d results", len(results))
    output = []
    for r in results:
        note = memory.read(r["id"])
        entry = {
            "id": r["id"],
            "name": note.name if note else None,
            "path": note.path if note else None,
            "content": r["content"],
            "tags": r.get("tags", []),
            "keywords": r.get("keywords", []),
            "links": note.links if note else [],
            "backlinks": note.backlinks if note else [],
        }
        output.append(entry)

    return json.dumps(output, ensure_ascii=False)


@optional_tool("read_memory")
def read_memory(memory_id: str) -> str:
    """Read a specific memory note by its ID.

    Returns the full content and all metadata for a single memory.

    Args:
        memory_id: The UUID of the memory to read.

    Returns:
        JSON with full memory content and metadata, or error if not found.
    """
    logger.info("read_memory: id=%s", memory_id)
    note = memory.read(memory_id)
    if not note:
        logger.warning("read_memory: not found id=%s", memory_id)
        return json.dumps({"error": f"Memory {memory_id} not found"})

    logger.info("read_memory: found name=%s", note.name)
    return json.dumps({
        "id": note.id,
        "name": note.name,
        "path": note.path,
        "content": note.content,
        "summary": note.summary,
        "keywords": note.keywords,
        "tags": note.tags,
        "links": note.links,
        "backlinks": note.backlinks,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "retrieval_count": note.retrieval_count,
    }, ensure_ascii=False)


@optional_tool("update_memory")
def update_memory(
    memory_id: str,
    content: Optional[str] = None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """Update an existing memory note.

    When content changes, all metadata (name, path, keywords, tags, summary)
    is automatically re-generated. The old file is cleaned up if the path changes.

    Use this to enrich existing memories with new information, corrections,
    or additional context discovered later.

    Args:
        memory_id: The UUID of the memory to update.
        content: New content (triggers full re-analysis if changed).
        name: New name override.
        path: New path override.
        tags: New tags override.

    Returns:
        JSON with updated memory metadata, or error if not found.
    """
    kwargs = {}
    if content is not None:
        kwargs["content"] = content
    if name is not None:
        kwargs["name"] = name
    if path is not None:
        kwargs["path"] = path
    if tags is not None:
        kwargs["tags"] = tags

    if not kwargs:
        return json.dumps({"error": "No fields to update"})

    logger.info("update_memory: id=%s fields=%s", memory_id, list(kwargs.keys()))
    success = memory.update(memory_id, **kwargs)
    if not success:
        logger.warning("update_memory: not found id=%s", memory_id)
        return json.dumps({"error": f"Memory {memory_id} not found"})

    note = memory.read(memory_id)
    logger.info("update_memory: updated id=%s name=%s", note.id, note.name)
    return json.dumps({
        "id": note.id,
        "name": note.name,
        "path": note.path,
        "filepath": note.filepath,
        "tags": note.tags,
        "has_summary": note.summary is not None,
    }, ensure_ascii=False)


@optional_tool("delete_memory")
def delete_memory(memory_id: str) -> str:
    """Delete a memory note and clean up all its links.

    This removes the memory from the knowledge base, deletes its markdown file,
    and cleans up all forward links and backlinks in related memories.

    Args:
        memory_id: The UUID of the memory to delete.

    Returns:
        JSON confirmation or error.
    """
    logger.info("delete_memory: id=%s", memory_id)
    success = memory.delete(memory_id)
    if not success:
        logger.warning("delete_memory: not found id=%s", memory_id)
        return json.dumps({"error": f"Memory {memory_id} not found"})
    logger.info("delete_memory: deleted id=%s", memory_id)
    return json.dumps({"deleted": memory_id})


@optional_tool("link_memories")
def link_memories(from_id: str, to_id: str) -> str:
    """Create a directional link between two memories.

    Links represent active, intentional connections. Backlinks are auto-maintained.
    Use this when you discover a relationship between two memories that the
    automatic evolution didn't catch.

    Args:
        from_id: Source memory ID (the one that "points to" the other).
        to_id: Target memory ID (the one being "pointed at").

    Returns:
        JSON confirmation with updated link info.
    """
    logger.info("link_memories: %s -> %s", from_id, to_id)
    memory.add_link(from_id, to_id)
    from_note = memory.read(from_id)
    to_note = memory.read(to_id)
    if not from_note or not to_note:
        return json.dumps({"error": "One or both memories not found"})

    return json.dumps({
        "linked": f"{from_note.name} -> {to_note.name}",
        "from_links": from_note.links,
        "to_backlinks": to_note.backlinks,
    }, ensure_ascii=False)


@optional_tool("unlink_memories")
def unlink_memories(from_id: str, to_id: str) -> str:
    """Remove a link between two memories. Backlink is auto-removed.

    Args:
        from_id: Source memory ID.
        to_id: Target memory ID.

    Returns:
        JSON confirmation.
    """
    logger.info("unlink_memories: %s -> %s", from_id, to_id)
    memory.remove_link(from_id, to_id)
    return json.dumps({"unlinked": f"{from_id} -> {to_id}"})


@optional_tool("memory_tree")
def memory_tree() -> str:
    """Show the full directory tree of all memories.

    Returns a tree-like visualization of how memories are organized,
    similar to the `tree` command. Useful for understanding the knowledge
    structure and finding where to place new memories.

    Returns:
        Tree-formatted string of the memory directory structure.
    """
    return memory.tree()


@optional_tool("memory_stats")
def memory_stats() -> str:
    """Get statistics about the memory system.

    Returns:
        JSON with total count, directory paths, and link statistics.
    """
    total = len(memory.memories)
    paths = sorted(set(m.path for m in memory.memories.values() if m.path))
    total_links = sum(len(m.links) for m in memory.memories.values())
    total_backlinks = sum(len(m.backlinks) for m in memory.memories.values())

    return json.dumps({
        "total_memories": total,
        "unique_paths": len(paths),
        "paths": paths,
        "total_links": total_links,
        "total_backlinks": total_backlinks,
        "persist_dir": PERSIST_DIR,
    }, ensure_ascii=False)


@optional_tool("sync_from_disk")
def sync_from_disk() -> str:
    """Sync: reload memories from disk files into the running system.

    Reads all markdown files from the persistent directory and updates the
    in-memory state and vector index to match what's on disk.

    Use this when:
    - You manually added/edited/deleted .md files in the notes folder
    - Another process modified the memory files
    - You suspect the in-memory state is stale

    Caveats:
    - Disk wins: if a file was edited, the disk version overwrites memory.
    - Notes deleted from disk are removed from memory.
    - New .md files on disk are added to memory.
    - Vector index is fully rebuilt (may take a moment for large databases).

    Returns:
        JSON with counts of added, updated, and removed notes.
    """
    logger.info("sync_from_disk: starting")
    result = memory.sync_from_disk()
    logger.info("sync_from_disk: %s", result)
    return json.dumps(result, ensure_ascii=False)


@optional_tool("sync_to_disk")
def sync_to_disk() -> str:
    """Sync: write current in-memory state to disk files.

    Saves all memories as markdown files and removes any orphan files
    on disk that don't correspond to a memory in the running system.

    Use this when:
    - You want to ensure disk matches the current state exactly
    - You suspect some file writes may have failed
    - You want to clean up stale files after bulk operations

    Returns:
        JSON with counts of written files and orphans removed.
    """
    logger.info("sync_to_disk: starting")
    result = memory.sync_to_disk()
    logger.info("sync_to_disk: %s", result)
    return json.dumps(result, ensure_ascii=False)


@optional_tool("graph_snapshot")
def graph_snapshot() -> str:
    """Return graph-ready memory data for UI clients.

    This is intended for visual frontends that need the whole vault graph:
    groups, nodes, links, and basic note metadata. It avoids forcing the UI
    to call `read_memory` repeatedly just to render the graph.

    Returns:
        JSON with groups, nodes, links, and graph stats.
    """
    return json.dumps(_build_graph_snapshot(), ensure_ascii=False)


def _auto_sync_loop(interval: int):
    """Background thread that periodically syncs memory to disk."""
    while True:
        _sync_stop_event.wait(interval)
        if _sync_stop_event.is_set():
            break
        try:
            result = memory.sync_to_disk()
            logger.info("Auto-sync to disk: %s", result)
        except Exception:
            logger.exception("Auto-sync to disk failed")


_sync_stop_event = threading.Event()
_sync_thread: threading.Thread | None = None

if AUTO_SYNC_ENABLED:
    _sync_thread = threading.Thread(
        target=_auto_sync_loop,
        args=(AUTO_SYNC_INTERVAL,),
        daemon=True,
        name="memory-auto-sync",
    )
    _sync_thread.start()
    logger.info("Auto-sync enabled: every %ds", AUTO_SYNC_INTERVAL)


def _get_notes_dir() -> str:
    notes_dir = os.path.join(os.path.abspath(PERSIST_DIR), "notes")
    os.makedirs(notes_dir, exist_ok=True)
    return notes_dir


@optional_tool("grep_memory")
def grep_memory(pattern: str, flags: Optional[str] = None) -> str:
    """Search memory files using grep (full CLI grep).

    Runs grep on all markdown files in the memory notes directory.

    Examples:
        grep_memory("PostgreSQL")                    -- basic search
        grep_memory("oauth.*token", "-i")            -- case-insensitive regex
        grep_memory("TODO", "-l")                    -- list filenames only
        grep_memory("error", "-c")                   -- count matches per file
        grep_memory("BEGIN", "-A 3")                 -- show 3 lines after match
        grep_memory("docker|kubernetes", "-E")       -- extended regex (OR)

    Args:
        pattern: Search pattern (string or regex depending on flags).
        flags: Optional grep flags as a single string (-i, -n, -l, -c, -w, -v, -E, -P, -A N, -B N, -C N).

    Returns:
        Grep output with paths relative to notes directory.
    """
    notes_dir = _get_notes_dir()
    cmd = ["grep", "-r", "--include=*.md"]
    if flags:
        cmd.extend(flags.split())
    cmd.extend([pattern, notes_dir])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if not result.stdout and result.returncode == 1:
            return "No matches found."
        if result.returncode > 1:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.replace(notes_dir + "/", "")
    except subprocess.TimeoutExpired:
        return "Error: grep timed out after 30 seconds."
    except FileNotFoundError:
        return "Error: grep command not found on this system."


@optional_tool("find_memory")
def find_memory(args: Optional[str] = None) -> str:
    """Search memory files using find (full CLI find).

    Runs find on the memory notes directory.

    Examples:
        find_memory()                                -- list all files
        find_memory("-name '*.md'")                  -- find by name pattern
        find_memory("-type d")                       -- list directories only
        find_memory("-name '*database*'")            -- find files matching pattern
        find_memory("-size +1k")                     -- files larger than 1KB
        find_memory("-mmin -60")                     -- modified in last 60 minutes
        find_memory("-maxdepth 2 -type d")           -- directories, max 2 levels deep
        find_memory("-path '*/database/*'")          -- files under database/ path

    Args:
        args: Optional find arguments as a single string (-name, -type, -size, -mtime, -mmin, -maxdepth, -empty, -path, -iname).

    Returns:
        Find output with paths relative to notes directory.
    """
    notes_dir = _get_notes_dir()
    cmd = ["find", notes_dir]
    if args:
        import shlex
        cmd.extend(shlex.split(args))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 and result.stderr:
            return f"Error: {result.stderr.strip()}"
        if not result.stdout.strip():
            return "No results found."
        return result.stdout.replace(notes_dir + "/", "").replace(notes_dir, ".")
    except subprocess.TimeoutExpired:
        return "Error: find timed out after 30 seconds."
    except FileNotFoundError:
        return "Error: find command not found on this system."


if __name__ == "__main__":
    mcp.run(transport="stdio")
