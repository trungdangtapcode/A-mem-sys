from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import litellm


def simple_tokenize(text):
    return word_tokenize(text)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function using Gemini embedding models via litellm."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = f"gemini/{model_name}" if not model_name.startswith("gemini/") else model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = litellm.embedding(model=self.model_name, input=input)
        return [item["embedding"] for item in response.data]


def _create_embedding_function(embedding_backend: str, model_name: str):
    """Create an embedding function based on the backend type."""
    if embedding_backend == "gemini":
        return GeminiEmbeddingFunction(model_name=model_name)
    else:
        return SentenceTransformerEmbeddingFunction(model_name=model_name)


class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2",
                 persist_dir: str = None, embedding_backend: str = "sentence-transformer"):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model
            persist_dir: Directory for persistent storage. If None, uses in-memory mode.
            embedding_backend: "sentence-transformer" or "gemini"
        """
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client(Settings(allow_reset=True))
        self.persist_dir = persist_dir

        self.embedding_function = _create_embedding_function(embedding_backend, model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB with enhanced embedding using metadata.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata including keywords, tags, context
            doc_id: Unique identifier for the document
        """
        # Use summary for embedding when available (long content),
        # otherwise use original content
        summary = metadata.get('summary')
        enhanced_document = summary if summary else document
        
        # Add context information
        if 'context' in metadata and metadata['context'] != "General":
            enhanced_document += f" context: {metadata['context']}"
        
        # Add keywords information    
        if 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords'] if isinstance(metadata['keywords'], list) else json.loads(metadata['keywords'])
            if keywords:
                enhanced_document += f" keywords: {', '.join(keywords)}"
        
        # Add tags information
        if 'tags' in metadata and metadata['tags']:
            tags = metadata['tags'] if isinstance(metadata['tags'], list) else json.loads(metadata['tags'])
            if tags:
                enhanced_document += f" tags: {', '.join(tags)}"
        
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # Store enhanced document content for better embedding
        processed_metadata['enhanced_content'] = enhanced_document
                
        # Use enhanced document content for embedding generation
        self.collection.add(
            documents=[enhanced_document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
    def clear(self):
        """Delete all documents and recreate the collection."""
        self.client.delete_collection("memories")
        self.collection = self.client.get_or_create_collection(
            name="memories", embedding_function=self.embedding_function
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass

        return results


class ZvecRetriever:
    """Vector database retrieval using Zvec."""

    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2",
                 persist_dir: str = None, embedding_backend: str = "sentence-transformer"):
        """Initialize Zvec retriever.

        Args:
            collection_name: Name of the Zvec collection
            model_name: Name of the embedding model
            persist_dir: Directory for persistent storage. Required for Zvec.
            embedding_backend: "sentence-transformer" or "gemini"
        """
        import zvec as _zvec
        self._zvec = _zvec

        self.embedding_function = _create_embedding_function(embedding_backend, model_name)
        self.persist_dir = persist_dir

        # Determine embedding dimension by encoding a test string
        test_embedding = self.embedding_function(["test"])
        self._dimension = len(test_embedding[0])

        collection_path = os.path.join(persist_dir, collection_name) if persist_dir else os.path.join("/tmp/zvec", collection_name)
        self._collection_path = collection_path

        if os.path.exists(collection_path):
            self.collection = _zvec.open(path=collection_path)
        else:
            os.makedirs(os.path.dirname(collection_path), exist_ok=True)
            schema = _zvec.CollectionSchema(
                name=collection_name,
                fields=[
                    _zvec.FieldSchema(name="metadata_json", data_type=_zvec.DataType.STRING),
                ],
                vectors=[
                    _zvec.VectorSchema(
                        name="embedding",
                        data_type=_zvec.DataType.VECTOR_FP32,
                        dimension=self._dimension,
                        index_param=_zvec.HnswIndexParam(metric_type=_zvec.MetricType.COSINE),
                    ),
                ],
            )
            self.collection = _zvec.create_and_open(path=collection_path, schema=schema)

    def clear(self):
        """Delete all documents and recreate the collection."""
        self.collection.destroy()
        schema = self._zvec.CollectionSchema(
            name="memories",
            fields=[
                self._zvec.FieldSchema(name="metadata_json", data_type=self._zvec.DataType.STRING),
            ],
            vectors=[
                self._zvec.VectorSchema(
                    name="embedding",
                    data_type=self._zvec.DataType.VECTOR_FP32,
                    dimension=self._dimension,
                    index_param=self._zvec.HnswIndexParam(metric_type=self._zvec.MetricType.COSINE),
                ),
            ],
        )
        self.collection = self._zvec.create_and_open(path=self._collection_path, schema=schema)

    def add_document(self, document: str, metadata: dict, doc_id: str):
        """Add a document to Zvec with enhanced embedding using metadata."""
        summary = metadata.get('summary')
        enhanced_document = summary if summary else document

        if 'context' in metadata and metadata['context'] != "General":
            enhanced_document += f" context: {metadata['context']}"
        if 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords'] if isinstance(metadata['keywords'], list) else json.loads(metadata['keywords'])
            if keywords:
                enhanced_document += f" keywords: {', '.join(keywords)}"
        if 'tags' in metadata and metadata['tags']:
            tags = metadata['tags'] if isinstance(metadata['tags'], list) else json.loads(metadata['tags'])
            if tags:
                enhanced_document += f" tags: {', '.join(tags)}"

        # Generate embedding
        embedding = self.embedding_function([enhanced_document])[0]

        # Serialize metadata to JSON string
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed_metadata[key] = value
            elif value is None:
                processed_metadata[key] = None
            else:
                processed_metadata[key] = str(value)

        doc = self._zvec.Doc(
            id=doc_id,
            vectors={"embedding": embedding},
            fields={"metadata_json": json.dumps(processed_metadata, ensure_ascii=False)},
        )
        self.collection.insert(doc)

    def delete_document(self, doc_id: str):
        """Delete a document from Zvec."""
        self.collection.delete(ids=doc_id)

    def search(self, query: str, k: int = 5):
        """Search for similar documents. Returns ChromaDB-compatible result format."""
        embedding = self.embedding_function([query])[0]

        results = self.collection.query(
            vectors=self._zvec.VectorQuery(field_name="embedding", vector=embedding),
            topk=k,
        )

        ids = []
        metadatas = []
        distances = []

        for doc in results:
            ids.append(doc.id)
            distances.append(doc.score)
            # Deserialize metadata
            meta_str = doc.fields.get("metadata_json", "{}")
            meta = json.loads(meta_str) if isinstance(meta_str, str) else {}
            # Parse list/dict values
            for key, value in meta.items():
                if isinstance(value, str):
                    try:
                        if value.startswith('[') or value.startswith('{'):
                            meta[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
            metadatas.append(meta)

        return {
            'ids': [ids],
            'metadatas': [metadatas],
            'distances': [distances],
        }
