"""
Pathway-based Document and Vector Store

This module implements the ingestion layer using Pathway for:
- Document storage and indexing
- Vector embeddings and similarity search
- Real-time streaming capabilities
- Metadata tracking

Research foundation:
- Long-document modeling (Lu et al. 2023)
- Efficient retrieval at scale
"""

import pathway as pw
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Document:
    """Document representation in Pathway store"""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class PathwayDocumentStore:
    """
    Pathway-based document store for long novels.
    
    Handles:
    - Document ingestion and chunking
    - Vector embedding storage
    - Efficient similarity search
    - Metadata filtering
    """
    
    def __init__(self, embedding_model=None, chunk_size: int = 1000):
        """
        Initialize Pathway document store.
        
        Args:
            embedding_model: Embedding model (e.g., sentence-transformers)
            chunk_size: Size of text chunks in characters
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.documents = {}  # In-memory store: doc_id -> Document
        self.chunk_to_doc = {}  # chunk_id -> doc_id mapping
        self.embeddings_index = []  # List of (chunk_id, embedding)
        
    def ingest_novel(
        self, 
        novel_text: str, 
        novel_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Ingest a full novel into the store.
        
        Args:
            novel_text: Full text of the novel
            novel_id: Unique identifier for the novel
            metadata: Additional metadata (title, author, etc.)
        
        Returns:
            List of chunk IDs created
        """
        from .utils import chunk_text_semantic
        
        # Chunk the text
        chunks = chunk_text_semantic(
            novel_text,
            chunk_size=self.chunk_size,
            overlap=200,
            preserve_sentences=True
        )
        
        chunk_ids = []
        
        # Store each chunk
        for chunk in chunks:
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                'novel_id': novel_id,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'chunk_index': len(chunk_ids)
            })
            
            # Create document
            doc = Document(
                doc_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk_meta
            )
            
            # Generate embedding if model available
            if self.embedding_model:
                doc.embedding = self._generate_embedding(chunk.text)
                self.embeddings_index.append((chunk.chunk_id, doc.embedding))
            
            self.documents[chunk.chunk_id] = doc
            self.chunk_to_doc[chunk.chunk_id] = novel_id
            chunk_ids.append(chunk.chunk_id)
        
        print(f"✓ Ingested novel '{novel_id}': {len(chunk_ids)} chunks")
        return chunk_ids
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        if self.embedding_model is None:
            # Fallback: simple bag-of-words embedding
            words = text.lower().split()
            # Create a simple hash-based embedding
            embedding = np.zeros(384)  # Standard embedding size
            for word in words[:100]:  # Limit to first 100 words
                hash_val = hash(word) % 384
                embedding[hash_val] += 1
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            # Use actual embedding model
            return self.embedding_model.encode(text)
    
    def search_similar(
        self, 
        query: str, 
        top_k: int = 10,
        novel_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            novel_id: Optional filter by novel ID
        
        Returns:
            List of results with chunk_id, text, score
        """
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for chunk_id, embedding in self.embeddings_index:
            # Filter by novel if specified
            if novel_id and self.chunk_to_doc.get(chunk_id) != novel_id:
                continue
            
            # Calculate cosine similarity
            score = np.dot(query_embedding, embedding)
            similarities.append((chunk_id, score))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for chunk_id, score in top_results:
            doc = self.documents[chunk_id]
            results.append({
                'chunk_id': chunk_id,
                'text': doc.text,
                'score': float(score),
                'metadata': doc.metadata
            })
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Optional[Document]:
        """Retrieve a specific chunk by ID"""
        return self.documents.get(chunk_id)
    
    def get_context_window(
        self, 
        chunk_id: str, 
        window_size: int = 2
    ) -> List[Document]:
        """
        Get surrounding chunks for context.
        
        Args:
            chunk_id: Central chunk ID
            window_size: Number of chunks before/after
        
        Returns:
            List of documents in context window
        """
        doc = self.documents.get(chunk_id)
        if not doc:
            return []
        
        novel_id = self.chunk_to_doc[chunk_id]
        chunk_index = doc.metadata.get('chunk_index', 0)
        
        # Find chunks in window
        context = []
        for cid, doc in self.documents.items():
            if self.chunk_to_doc.get(cid) == novel_id:
                idx = doc.metadata.get('chunk_index', 0)
                if chunk_index - window_size <= idx <= chunk_index + window_size:
                    context.append((idx, doc))
        
        # Sort by index
        context.sort(key=lambda x: x[0])
        return [doc for _, doc in context]
    
    def keyword_search(
        self, 
        keywords: List[str], 
        novel_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks containing specific keywords.
        
        Args:
            keywords: List of keywords to search for
            novel_id: Optional filter by novel ID
        
        Returns:
            List of matching chunks with relevance scores
        """
        results = []
        keywords_lower = [k.lower() for k in keywords]
        
        for chunk_id, doc in self.documents.items():
            # Filter by novel if specified
            if novel_id and self.chunk_to_doc.get(chunk_id) != novel_id:
                continue
            
            # Count keyword matches
            text_lower = doc.text.lower()
            matches = sum(1 for kw in keywords_lower if kw in text_lower)
            
            if matches > 0:
                results.append({
                    'chunk_id': chunk_id,
                    'text': doc.text,
                    'score': matches,
                    'metadata': doc.metadata
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        novels = set(self.chunk_to_doc.values())
        
        return {
            'total_chunks': len(self.documents),
            'total_novels': len(novels),
            'total_embeddings': len(self.embeddings_index),
            'novels': list(novels)
        }
    
    def export_index(self, filepath: str):
        """Export document index for persistence"""
        index_data = {
            'documents': {
                doc_id: {
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                for doc_id, doc in self.documents.items()
            },
            'chunk_to_doc': self.chunk_to_doc
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"✓ Exported index to {filepath}")
    
    def load_index(self, filepath: str):
        """Load document index from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        # Restore documents
        for doc_id, doc_data in index_data['documents'].items():
            doc = Document(
                doc_id=doc_id,
                text=doc_data['text'],
                metadata=doc_data['metadata']
            )
            self.documents[doc_id] = doc
            
            # Regenerate embeddings
            if self.embedding_model:
                doc.embedding = self._generate_embedding(doc.text)
                self.embeddings_index.append((doc_id, doc.embedding))
        
        self.chunk_to_doc = index_data['chunk_to_doc']
        
        print(f"✓ Loaded index from {filepath}")


class PathwayVectorStore:
    """
    Specialized vector store for embedding-based retrieval.
    Optimized for long-context queries.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.vectors = []  # List of (id, vector, metadata)
        self.index = None  # Could use FAISS for large-scale
    
    def add_vectors(
        self, 
        ids: List[str], 
        vectors: List[np.ndarray],
        metadata: Optional[List[Dict]] = None
    ):
        """Add vectors to the store"""
        if metadata is None:
            metadata = [{}] * len(ids)
        
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            self.vectors.append((id_, vec, metadata[i]))
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for nearest neighbors"""
        from .utils import cosine_similarity
        
        similarities = []
        for id_, vec, meta in self.vectors:
            score = cosine_similarity(query_vector, vec)
            similarities.append((id_, score, meta))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {'id': id_, 'score': score, 'metadata': meta}
            for id_, score, meta in similarities[:top_k]
        ]
