import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever, ZvecRetriever
import json
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 name: Optional[str] = None,
                 path: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[List[str]] = None,
                 backlinks: Optional[List[str]] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 summary: Optional[str] = None):
        """Initialize a new memory note with its associated metadata.

        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            name (Optional[str]): Human-readable name for the memory (used as filename)
            path (Optional[str]): Directory path for organizing memory in a tree
                (e.g. "devops/kubernetes", "backend/database")
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[List[str]]): Active references to other memories (can be added/removed)
            backlinks (Optional[List[str]]): Passive references from other memories (auto-maintained)
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
            summary (Optional[str]): Short summary for embedding when content exceeds token limit
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.path = path

        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.backlinks = backlinks or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

        # Summary for long content embedding
        self.summary = summary

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filesystem-safe slug."""
        import re
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        return slug.strip('-')

    @property
    def filename(self) -> str:
        """Generate a filesystem-safe filename from the name, falling back to id."""
        if not self.name:
            return self.id
        return self._slugify(self.name) or self.id

    @property
    def filepath(self) -> str:
        """Generate the relative file path including directory tree.

        Returns path like 'devops/kubernetes/container-orchestration.md'
        or just 'container-orchestration.md' if no path is set.
        """
        name_slug = self.filename
        if self.path:
            # Slugify each segment of the path
            segments = [self._slugify(s) for s in self.path.strip("/").split("/") if s.strip()]
            segments = [s for s in segments if s]  # remove empty
            if segments:
                return os.path.join(*segments, f"{name_slug}.md")
        return f"{name_slug}.md"

    def to_markdown(self) -> str:
        """Serialize this note to a markdown string with YAML frontmatter."""
        frontmatter = {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "keywords": self.keywords,
            "links": self.links,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "evolution_history": self.evolution_history,
            "category": self.category,
            "tags": self.tags,
        }
        if self.summary:
            frontmatter["summary"] = self.summary

        lines = ["---"]
        for key, value in frontmatter.items():
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> "MemoryNote":
        """Deserialize a MemoryNote from a markdown string with YAML frontmatter."""
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid markdown format: missing frontmatter")

        frontmatter_str = parts[1].strip()
        content = parts[2].strip()

        metadata = {}
        for line in frontmatter_str.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                metadata[key.strip()] = json.loads(value)

        return cls(
            content=content,
            id=metadata.get("id"),
            name=metadata.get("name"),
            path=metadata.get("path"),
            keywords=metadata.get("keywords"),
            links=metadata.get("links"),
            retrieval_count=metadata.get("retrieval_count"),
            timestamp=metadata.get("timestamp"),
            last_accessed=metadata.get("last_accessed"),
            context=metadata.get("context"),
            evolution_history=metadata.get("evolution_history"),
            category=metadata.get("category"),
            tags=metadata.get("tags"),
            summary=metadata.get("summary"),
        )


class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 persist_dir: Optional[str] = None,
                 embedding_backend: str = "sentence-transformer",
                 vector_backend: str = "chroma",
                 context_aware_analysis: bool = False,
                 context_aware_tree: bool = False,
                 max_links: Optional[int] = None):
        """Initialize the memory system.

        Args:
            model_name: Name of the embedding model
            llm_backend: LLM backend to use (openai/ollama/sglang/gemini)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            sglang_host: Host URL for SGLang server (default: http://localhost)
            sglang_port: Port for SGLang server (default: 30000)
            persist_dir: Directory for persistent storage. If None, uses in-memory mode.
            embedding_backend: Embedding backend ("sentence-transformer" or "gemini")
            vector_backend: Vector database backend ("chroma" or "zvec")
            context_aware_analysis: When True, analyze_content sees similar memories
                and directory paths to keep naming/categorization consistent.
            context_aware_tree: When True (requires context_aware_analysis=True),
                include full directory tree in analysis context so LLM can see
                the complete knowledge structure for better linking and placement.
            max_links: Maximum number of links created per note during evolution.
                If None, no limit (LLM decides freely).
        """
        self.memories = {}
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.embedding_backend = embedding_backend
        self.vector_backend = vector_backend
        self.context_aware_analysis = context_aware_analysis
        self.context_aware_tree = context_aware_tree
        self.max_links = max_links
        self._initialize_evolution_prompt()

        # Set up subdirectories for persistence
        self._notes_dir = None
        self._vector_dir = None
        if self.persist_dir:
            self._notes_dir = os.path.join(self.persist_dir, "notes")
            self._vector_dir = os.path.join(self.persist_dir, "vectordb")
            os.makedirs(self._notes_dir, exist_ok=True)
            os.makedirs(self._vector_dir, exist_ok=True)

        if not self.persist_dir and self.vector_backend == "chroma":
            # In-memory mode: reset old ChromaDB collection
            try:
                temp = ChromaRetriever(
                    collection_name="memories", model_name=self.model_name,
                    embedding_backend=self.embedding_backend
                )
                temp.client.reset()
            except Exception as e:
                logger.warning(f"Could not reset ChromaDB collection: {e}")

        self.retriever = self._create_retriever()

        if self.persist_dir:
            self._load_notes()

        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, sglang_host, sglang_port)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def _create_retriever(self):
        """Create the vector retriever based on configured backend."""
        if self.vector_backend == "zvec":
            return ZvecRetriever(
                collection_name="memories", model_name=self.model_name,
                persist_dir=self._vector_dir, embedding_backend=self.embedding_backend
            )
        else:
            return ChromaRetriever(
                collection_name="memories", model_name=self.model_name,
                persist_dir=self._vector_dir, embedding_backend=self.embedding_backend
            )

    # --- Persistence helpers ---

    def _save_note(self, note: MemoryNote):
        """Save a single MemoryNote as a markdown file in its directory tree."""
        if not self._notes_dir:
            return
        filepath = os.path.join(self._notes_dir, note.filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(note.to_markdown())

    def _delete_note_file(self, memory_id: str):
        """Delete a MemoryNote's markdown file and clean up empty parent dirs."""
        if not self._notes_dir:
            return
        note = self.memories.get(memory_id)
        if note:
            filepath = os.path.join(self._notes_dir, note.filepath)
            if os.path.exists(filepath):
                os.remove(filepath)
                # Clean up empty parent directories
                parent = os.path.dirname(filepath)
                while parent != self._notes_dir:
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                        parent = os.path.dirname(parent)
                    else:
                        break

    def _load_notes(self):
        """Load all MemoryNotes from markdown files in the notes directory tree."""
        if not self._notes_dir:
            return
        for dirpath, _dirnames, filenames in os.walk(self._notes_dir):
            for filename in filenames:
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                try:
                    note = MemoryNote.from_markdown(text)
                    self.memories[note.id] = note
                except Exception as e:
                    logger.warning(f"Could not load note {filepath}: {e}")

        # Rebuild backlinks from links (backlinks are never persisted)
        self._rebuild_backlinks()

    def _rebuild_backlinks(self):
        """Derive all backlinks from forward links. Also prunes dead links."""
        # Clear all backlinks
        for note in self.memories.values():
            note.backlinks = []

        # Rebuild: if A.links contains B, then B.backlinks gets A
        for note in self.memories.values():
            valid_links = []
            for linked_id in note.links:
                if linked_id in self.memories:
                    valid_links.append(linked_id)
                    self.memories[linked_id].backlinks.append(note.id)
            # Prune links to notes that no longer exist
            if len(valid_links) != len(note.links):
                note.links = valid_links
                self._save_note(note)

        self._initialize_evolution_prompt()

    def _initialize_evolution_prompt(self):
        """Initialize the prompt used by the memory evolution flow."""
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories (each line starts with memory_id):
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Use the memory_id from the neighbors above. Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["memory_id_1", "memory_id_2", ...],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''

    # Approximate word count threshold for generating summary.
    # all-MiniLM-L6-v2 supports 256 tokens; enhanced_document appends
    # gemini embedding truncates to 512 tokens. To leave room for metadata like
    # context/keywords/tags, so we reserve ~100 tokens for metadata
    # and use ~250 words as the content threshold.
    SUMMARY_WORD_THRESHOLD = 250

    def tree(self) -> str:
        """Generate a tree-like string of the memory directory structure."""
        if not self.memories:
            return "(empty)"

        # Build nested dict from all note filepaths
        root = {}
        for mem in sorted(self.memories.values(), key=lambda m: m.filepath):
            parts = mem.filepath.split(os.sep)
            node = root
            for part in parts:
                node = node.setdefault(part, {})

        def _render(node, prefix="", is_last=True):
            lines = []
            items = list(node.items())
            for i, (name, children) in enumerate(items):
                last = (i == len(items) - 1)
                connector = "└── " if last else "├── "
                lines.append(f"{prefix}{connector}{name}")
                if children:
                    extension = "    " if last else "│   "
                    lines.extend(_render(children, prefix + extension, last))
            return lines

        return "\n".join(_render(root))

    def _get_existing_context(self, content: str, include_tree: bool = False) -> str:
        """Collect context from similar memories and existing directory structure.

        Args:
            content: The note content to find similar memories for.
            include_tree: If True, include full directory tree in context.
        """
        if not self.memories:
            return ""

        lines = []

        # 1. Find similar memories via vector search
        try:
            results = self.retriever.search(content, k=5)
            if results.get('ids') and results['ids'][0]:
                similar = []
                similar_tags = set()
                for doc_id in results['ids'][0]:
                    mem = self.memories.get(doc_id)
                    if mem:
                        similar.append(f"{mem.name} (path: {mem.path})")
                        similar_tags.update(mem.tags)
                if similar:
                    lines.append(f"Similar memories: {', '.join(similar)}")
                if similar_tags:
                    lines.append(f"Their tags: {', '.join(sorted(similar_tags)[:15])}")
        except Exception:
            pass

        # 2. Directory structure
        if include_tree:
            lines.append(f"Full memory tree:\n{self.tree()}")
        else:
            all_paths = sorted(set(m.path for m in self.memories.values() if m.path))
            if all_paths:
                tree_paths = sorted(set(
                    "/".join(p.split("/")[:2]) for p in all_paths
                ))
                lines.append(f"Existing directory tree: {', '.join(tree_paths)}")

        return "\n            ".join(lines)

    def analyze_content(self, content: str) -> Dict:
        """Analyze content using LLM to extract semantic metadata.

        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        - Summary: A concise summary when content exceeds the embedding token limit

        Args:
            content (str): The text content to analyze

        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
                - summary: Optional[str] (only when content is long)
        """
        needs_summary = len(content.split()) > self.SUMMARY_WORD_THRESHOLD

        summary_instruction = ""
        summary_schema = {}
        if needs_summary:
            summary_instruction = """
            4. Writing a concise summary (2-3 sentences, under 100 words) that captures
               the key information. This summary will be used for semantic search embedding,
               so it must preserve the most important concepts and terms."""
            summary_schema = {
                "summary": {
                    "type": "string",
                }
            }

        context_section = ""
        if self.context_aware_analysis:
            existing = self._get_existing_context(content, include_tree=self.context_aware_tree)
            if existing:
                context_section = f"""
            IMPORTANT - Existing knowledge base context:
            {existing}
            Reuse existing paths and tags when the content fits. Only create new ones
            if no existing option is appropriate. Keep naming consistent."""

        prompt = f"""Generate a structured analysis of the following content by:
            1. Creating a short, descriptive name (2-5 words, lowercase, like a file name)
            2. Creating a directory path that categorizes this content in a knowledge tree
            3. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            4. Extracting core themes and contextual elements
            5. Creating relevant categorical tags
            {summary_instruction}
            {context_section}

            Format the response as a JSON object:
            {{
                "name":
                    // a short descriptive name for this memory (2-5 words, lowercase)
                    // e.g. "docker container basics", "postgresql jsonb indexing"
                ,
                "path":
                    // a directory path (2-4 levels, lowercase) that places this memory
                    // in a logical knowledge tree. Use broad domain first, then narrow topic.
                    // e.g. "devops/containerization", "backend/database",
                    //      "backend/middleware/auth", "ml/nlp", "devops/ci-cd"
                    // Keep segments short (1-2 words each), consistent, and reusable.
                ,
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context":
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]{', "summary": "..."' if needs_summary else ''}
            }}

            Content for analysis:
            {content}"""

        schema_properties = {
            "name": {
                "type": "string",
            },
            "path": {
                "type": "string",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            },
            "context": {
                "type": "string",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            },
        }
        schema_properties.update(summary_schema)

        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": schema_properties
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)
        
        # 🔧 LLM Analysis Enhancement: Auto-generate attributes using LLM if they are empty or default values
        needs_analysis = (
            not note.keywords or  # keywords is empty list
            note.context == "General" or  # context is default value
            not note.tags  # tags is empty list
        )
        
        if needs_analysis:
            analysis = self.analyze_content(content)

            # Only update attributes that are not provided or have default values
            if note.name is None:
                note.name = analysis.get("name")
            if note.path is None:
                note.path = analysis.get("path")
            if not note.keywords:
                note.keywords = analysis.get("keywords", [])
            if note.context == "General":
                note.context = analysis.get("context", "General")
            if not note.tags:
                note.tags = analysis.get("tags", [])
            if note.summary is None:
                note.summary = analysis.get("summary")
        
        # Add to memories before evolution so add_link can find it
        self.memories[note.id] = note
        evo_label, note = self.process_memory(note)
        self._save_note(note)

        # Add to ChromaDB with complete metadata
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
            "summary": note.summary
        }
        self.retriever.add_document(note.content, metadata, note.id)
        
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def sync_from_disk(self) -> Dict:
        """Sync: read current persistent files → update in-memory state + vectordb.

        Loads all markdown files from disk, detects added/modified/deleted notes
        compared to current in-memory state, and rebuilds the vector index.

        Caveats:
        - Notes that exist on disk but not in memory are ADDED.
        - Notes that exist in memory but not on disk are REMOVED.
        - Notes whose content differs on disk are UPDATED (disk wins).
        - Vector index is fully rebuilt after sync.
        - Backlinks are rebuilt from forward links.

        Returns:
            Dict with counts of added, updated, removed notes.
        """
        if not self._notes_dir:
            return {"error": "No persist_dir configured"}

        # Load all notes from disk
        disk_notes = {}
        for dirpath, _dirnames, filenames in os.walk(self._notes_dir):
            for filename in filenames:
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                try:
                    note = MemoryNote.from_markdown(text)
                    disk_notes[note.id] = note
                except Exception as e:
                    logger.warning(f"Could not load note {filepath}: {e}")

        added = 0
        updated = 0
        removed = 0

        # Detect added and updated
        for nid, disk_note in disk_notes.items():
            if nid not in self.memories:
                added += 1
            elif disk_note.content != self.memories[nid].content:
                updated += 1
            self.memories[nid] = disk_note

        # Detect removed (in memory but not on disk)
        removed_ids = [nid for nid in self.memories if nid not in disk_notes]
        for nid in removed_ids:
            del self.memories[nid]
            removed += 1

        # Rebuild backlinks and vector index
        self._rebuild_backlinks()
        self.consolidate_memories()

        return {"added": added, "updated": updated, "removed": removed}

    def sync_to_disk(self) -> Dict:
        """Sync: write current in-memory state → persistent files + vectordb.

        Saves all in-memory notes as markdown files, removes orphan files
        that don't correspond to any in-memory note, and rebuilds the vector index.

        Returns:
            Dict with counts of written and cleaned files.
        """
        if not self._notes_dir:
            return {"error": "No persist_dir configured"}

        # Collect all existing .md filepaths on disk
        existing_files = set()
        for dirpath, _dirnames, filenames in os.walk(self._notes_dir):
            for filename in filenames:
                if filename.endswith(".md"):
                    existing_files.add(os.path.join(dirpath, filename))

        # Write all in-memory notes to disk
        written_files = set()
        for note in self.memories.values():
            filepath = os.path.join(self._notes_dir, note.filepath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(note.to_markdown())
            written_files.add(filepath)

        # Remove orphan files (on disk but not in memory)
        orphans = existing_files - written_files
        for orphan_path in orphans:
            os.remove(orphan_path)
            # Clean empty parent dirs
            parent = os.path.dirname(orphan_path)
            while parent != self._notes_dir:
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)
                    parent = os.path.dirname(parent)
                else:
                    break

        # Rebuild vector index from memory
        self.consolidate_memories()

        return {"written": len(written_files), "orphans_removed": len(orphans)}

    def consolidate_memories(self):
        """Consolidate memories: rebuild the vector index from current in-memory state."""
        self.retriever.clear()

        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags,
                "summary": memory.summary
            }
            self.retriever.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Find related memories using ChromaDB retrieval

        Returns:
            Tuple[str, List[str]]: (formatted_memory_string, list_of_memory_ids)
        """
        if not self.memories:
            return "", []

        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)

            # Convert to list of memories
            memory_str = ""
            memory_ids = []

            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Get metadata from ChromaDB results
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string with actual memory ID
                        memory_str += f"memory_id:{doc_id}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        memory_ids.append(doc_id)

            return memory_str, memory_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.

        When content changes, all LLM-derived metadata (name, keywords, context,
        tags, summary) is re-generated. The old markdown file is removed if the
        filename changes.

        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update

        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False

        note = self.memories[memory_id]
        old_filepath = note.filepath

        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)

        # Re-analyze all metadata when content changes
        if 'content' in kwargs:
            analysis = self.analyze_content(note.content)
            # Only overwrite fields that were NOT explicitly provided in kwargs
            if 'name' not in kwargs:
                note.name = analysis.get("name", note.name)
            if 'path' not in kwargs:
                note.path = analysis.get("path", note.path)
            if 'keywords' not in kwargs:
                note.keywords = analysis.get("keywords", note.keywords)
            if 'context' not in kwargs:
                note.context = analysis.get("context", note.context)
            if 'tags' not in kwargs:
                note.tags = analysis.get("tags", note.tags)
            if 'summary' not in kwargs:
                note.summary = analysis.get("summary")

        # Delete old markdown file if filepath changed
        if self._notes_dir and old_filepath != note.filepath:
            old_full_path = os.path.join(self._notes_dir, old_filepath)
            if os.path.exists(old_full_path):
                os.remove(old_full_path)
                # Clean up empty parent dirs
                parent = os.path.dirname(old_full_path)
                while parent != self._notes_dir:
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                        parent = os.path.dirname(parent)
                    else:
                        break

        # Update in ChromaDB
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
            "summary": note.summary
        }

        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        self._save_note(note)

        return True
    
    def add_link(self, from_id: str, to_id: str):
        """Create a forward link from one note to another. Backlink is auto-created."""
        if from_id not in self.memories or to_id not in self.memories:
            return
        from_note = self.memories[from_id]
        to_note = self.memories[to_id]
        if to_id not in from_note.links:
            from_note.links.append(to_id)
            self._save_note(from_note)
        if from_id not in to_note.backlinks:
            to_note.backlinks.append(from_id)

    def remove_link(self, from_id: str, to_id: str):
        """Remove a forward link. Backlink is auto-removed."""
        if from_id not in self.memories or to_id not in self.memories:
            return
        from_note = self.memories[from_id]
        to_note = self.memories[to_id]
        if to_id in from_note.links:
            from_note.links.remove(to_id)
            self._save_note(from_note)
        if from_id in to_note.backlinks:
            to_note.backlinks.remove(from_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            note = self.memories[memory_id]

            # Clean up all links/backlinks via interface
            for linked_id in list(note.links):
                self.remove_link(memory_id, linked_id)
            for backlinked_id in list(note.backlinks):
                self.remove_link(backlinked_id, memory_id)

            # Delete markdown file first (needs note for filename)
            self._delete_note_file(memory_id)
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []

        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'tags': memory.tags,
                    'score': search_results['distances'][0][i]
                })

        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.

        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)

        The results are deduplicated and ranked by relevance.

        Args:
            query (str): The search query text
            k (int): Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - context: Memory context
                - keywords: Memory keywords
                - tags: Memory tags
                - score: Similarity score
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []

        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'tags': memory.tags,
                    'score': chroma_results['distances'][0][i]
                })

        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)

        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'tags': memory.tags,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)

        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        if not self.memories:
            return []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break
                    
                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.

        Args:
            note: The memory note to process

        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note

        try:
            # Get nearest neighbors - now returns actual memory IDs
            neighbors_text, memory_ids = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not memory_ids:
                return False, note

            # Format neighbors for LLM - in this case, neighbors_text is already formatted

            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(memory_ids)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json["should_evolve"]
                
                if should_evolve:
                    actions = response_json["actions"]
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            if self.max_links is not None:
                                suggest_connections = suggest_connections[:self.max_links]
                            for conn_id in suggest_connections:
                                self.add_link(note.id, conn_id)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]

                            # Update each neighbor memory using its actual ID
                            for i in range(min(len(memory_ids), len(new_tags_neighborhood))):
                                memory_id = memory_ids[i]

                                # Skip if memory doesn't exist
                                if memory_id not in self.memories:
                                    continue

                                # Get the memory to update
                                neighbor_memory = self.memories[memory_id]

                                # Update tags
                                if i < len(new_tags_neighborhood):
                                    neighbor_memory.tags = new_tags_neighborhood[i]

                                # Update context
                                if i < len(new_context_neighborhood):
                                    neighbor_memory.context = new_context_neighborhood[i]

                                # Save the updated memory back
                                self.memories[memory_id] = neighbor_memory
                                self._save_note(neighbor_memory)

                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
