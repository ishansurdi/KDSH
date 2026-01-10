"""
Multi-Hop Evidence Retrieval

Implements sophisticated retrieval that:
- Performs multi-hop reasoning across document
- Maintains evidence provenance
- Combines vector and keyword search
- Reranks results

Research foundation:
- NovelHopQA 2025 (Multi-hop failure analysis)
- Rashkin et al. 2021 (Evidence attribution)
- Lu et al. 2023 (Long-document retrieval)
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RetrievalResult:
    """Result from evidence retrieval"""
    chunk_id: str
    text: str
    score: float
    hop: int  # Which hop this was retrieved in
    source_claim: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'hop': self.hop,
            'source_claim': self.source_claim,
            'metadata': self.metadata
        }


class MultiHopRetriever:
    """
    Multi-hop evidence retrieval system.
    
    Performs iterative retrieval:
    1. Retrieve evidence for initial claim
    2. Extract entities/concepts from retrieved evidence
    3. Retrieve additional evidence for those concepts
    4. Repeat for multiple hops
    5. Aggregate and rerank all evidence
    """
    
    def __init__(self, document_store, max_hops: int = 3):
        """
        Initialize retriever.
        
        Args:
            document_store: PathwayDocumentStore instance
            max_hops: Maximum number of retrieval hops
        """
        self.document_store = document_store
        self.max_hops = max_hops
    
    def retrieve_evidence(
        self,
        query: str,
        novel_id: str,
        top_k_per_hop: int = 5,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Perform multi-hop evidence retrieval.
        
        Args:
            query: Initial query (e.g., claim text)
            novel_id: Novel to search in
            top_k_per_hop: Results to retrieve per hop
            rerank: Whether to rerank final results
        
        Returns:
            List of RetrievalResult objects
        """
        all_results = []
        seen_chunks = set()
        current_queries = [query]
        
        for hop in range(self.max_hops):
            hop_results = []
            
            for hop_query in current_queries:
                # Vector search
                vector_results = self.document_store.search_similar(
                    query=hop_query,
                    top_k=top_k_per_hop,
                    novel_id=novel_id
                )
                
                # Convert to RetrievalResult
                for result in vector_results:
                    chunk_id = result['chunk_id']
                    if chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        hop_results.append(RetrievalResult(
                            chunk_id=chunk_id,
                            text=result['text'],
                            score=result['score'],
                            hop=hop,
                            source_claim=query if hop == 0 else None,
                            metadata=result.get('metadata', {})
                        ))
            
            all_results.extend(hop_results)
            
            # Prepare queries for next hop
            if hop < self.max_hops - 1:
                current_queries = self._extract_expansion_queries(hop_results)
                if not current_queries:
                    break  # No more expansion possible
        
        # Rerank if requested
        if rerank and all_results:
            all_results = self._rerank_results(all_results, query)
            # IMPROVED: Ensure diversity in final results
            all_results = self._ensure_diversity(all_results)
        
        # IMPROVED: Return more results for better coverage
        return all_results[:top_k_per_hop * 2]
    
    def retrieve_for_claims(
        self,
        claims: List,
        novel_id: str,
        top_k_per_claim: int = 5
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve evidence for multiple claims.
        
        Args:
            claims: List of Claim objects
            novel_id: Novel to search in
            top_k_per_claim: Results per claim
        
        Returns:
            Dict mapping claim_id to list of results
        """
        evidence_map = {}
        
        for claim in claims:
            results = self.retrieve_evidence(
                query=claim.text,
                novel_id=novel_id,
                top_k_per_hop=top_k_per_claim,
                rerank=True
            )
            evidence_map[claim.claim_id] = results
        
        print(f"✓ Retrieved evidence for {len(claims)} claims")
        
        return evidence_map
    
    def retrieve_with_context(
        self,
        chunk_id: str,
        window_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a chunk with surrounding context.
        
        Args:
            chunk_id: Target chunk ID
            window_size: Number of chunks before/after
        
        Returns:
            List of chunks with context
        """
        context_docs = self.document_store.get_context_window(
            chunk_id=chunk_id,
            window_size=window_size
        )
        
        return [
            {
                'chunk_id': doc.doc_id,
                'text': doc.text,
                'metadata': doc.metadata
            }
            for doc in context_docs
        ]
    
    def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        novel_id: str,
        top_k: int = 10,
        vector_weight: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining vector and keyword retrieval.
        
        Args:
            query: Query text for vector search
            keywords: Keywords for lexical search
            novel_id: Novel to search
            top_k: Number of results
            vector_weight: Weight for vector scores (0-1)
        
        Returns:
            Combined and reranked results
        """
        # Vector search
        vector_results = self.document_store.search_similar(
            query=query,
            top_k=top_k * 2,
            novel_id=novel_id
        )
        
        # Keyword search
        keyword_results = self.document_store.keyword_search(
            keywords=keywords,
            novel_id=novel_id
        )
        
        # Combine scores
        chunk_scores = {}
        chunk_texts = {}
        chunk_metadata = {}
        
        # Add vector scores
        for result in vector_results:
            chunk_id = result['chunk_id']
            chunk_scores[chunk_id] = result['score'] * vector_weight
            chunk_texts[chunk_id] = result['text']
            chunk_metadata[chunk_id] = result.get('metadata', {})
        
        # Add keyword scores
        keyword_weight = 1.0 - vector_weight
        max_keyword_score = max([r['score'] for r in keyword_results]) if keyword_results else 1.0
        
        for result in keyword_results:
            chunk_id = result['chunk_id']
            normalized_score = result['score'] / max_keyword_score
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += normalized_score * keyword_weight
            else:
                chunk_scores[chunk_id] = normalized_score * keyword_weight
                chunk_texts[chunk_id] = result['text']
                chunk_metadata[chunk_id] = result.get('metadata', {})
        
        # Create results
        combined_results = []
        for chunk_id, score in chunk_scores.items():
            combined_results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=chunk_texts[chunk_id],
                score=score,
                hop=0,
                metadata=chunk_metadata[chunk_id]
            ))
        
        # Sort by score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results[:top_k]
    
    def _extract_expansion_queries(
        self,
        results: List[RetrievalResult]
    ) -> List[str]:
        """
        Extract queries for next hop from current results.
        Focuses on entities and key concepts.
        """
        from .utils import extract_entities
        
        queries = []
        
        for result in results[:3]:  # Use top 3 results
            # Extract entities
            entities = extract_entities(result.text)
            for entity in entities[:2]:  # Top 2 entities per result
                queries.append(entity['text'])
            
            # Extract key phrases (simple: noun phrases)
            # In production, use spaCy for better phrase extraction
            words = result.text.split()
            if len(words) > 10:
                # Take middle section as potential key phrase
                mid_start = len(words) // 3
                mid_end = 2 * len(words) // 3
                key_phrase = ' '.join(words[mid_start:mid_end])
                if len(key_phrase) > 20:
                    queries.append(key_phrase[:100])
        
        # Deduplicate
        return list(set(queries))[:5]  # Max 5 expansion queries
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        original_query: str
    ) -> List[RetrievalResult]:
        """
        Rerank results based on relevance to original query.
        
        Uses a combination of:
        - Original similarity score
        - Hop penalty (earlier hops weighted higher)
        - Diversity bonus
        """
        if not results:
            return results
        
        # Apply hop penalty
        hop_weights = {0: 1.0, 1: 0.8, 2: 0.6}
        
        for result in results:
            hop_penalty = hop_weights.get(result.hop, 0.5)
            result.score = result.score * hop_penalty
        
        # Apply diversity bonus (penalize very similar results)
        for i, result1 in enumerate(results):
            for result2 in results[i+1:]:
                # Simple diversity: check text overlap
                words1 = set(result1.text.lower().split())
                words2 = set(result2.text.lower().split())
                overlap = len(words1 & words2) / max(len(words1), len(words2), 1)
                
                if overlap > 0.7:  # High overlap
                    result2.score *= 0.9  # Slight penalty
        
        # Sort by adjusted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def retrieve_chain(
        self,
        start_claim: str,
        end_claim: str,
        novel_id: str,
        max_chain_length: int = 5
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve evidence chain connecting two claims.
        Useful for causal and temporal reasoning.
        
        Args:
            start_claim: Starting claim text
            end_claim: Ending claim text
            novel_id: Novel to search
            max_chain_length: Maximum chain length
        
        Returns:
            List of evidence chains (each chain is list of results)
        """
        # Retrieve for start
        start_evidence = self.retrieve_evidence(
            query=start_claim,
            novel_id=novel_id,
            top_k_per_hop=3,
            rerank=True
        )
        
        # Retrieve for end
        end_evidence = self.retrieve_evidence(
            query=end_claim,
            novel_id=novel_id,
            top_k_per_hop=3,
            rerank=True
        )
        
        # Find connecting evidence
        chains = []
        
        for start_ev in start_evidence[:2]:
            for end_ev in end_evidence[:2]:
                # Try to find intermediate evidence
                intermediate_query = f"{start_ev.text[:100]} ... {end_ev.text[:100]}"
                intermediate_evidence = self.retrieve_evidence(
                    query=intermediate_query,
                    novel_id=novel_id,
                    top_k_per_hop=2,
                    rerank=True
                )
                
                # Build chain
                chain = [start_ev] + intermediate_evidence[:1] + [end_ev]
                chains.append(chain)
        
        return chains[:3]  # Return top 3 chains
    
    def _ensure_diversity(
        self,
        results: List[RetrievalResult],
        similarity_threshold: float = 0.75
    ) -> List[RetrievalResult]:
        \"\"\"Ensure diversity in results by removing very similar chunks\"\"\"
        if not results:
            return results
        
        diverse_results = [results[0]]  # Keep first result
        
        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            result_words = set(result.text.lower().split())
            
            for selected in diverse_results:
                selected_words = set(selected.text.lower().split())
                overlap = len(result_words & selected_words) / max(len(result_words), len(selected_words), 1)
                
                if overlap > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
