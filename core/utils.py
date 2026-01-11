"""
Utility Functions for Long-Context Narrative Consistency System

Research foundations:
- Text chunking strategies from Lu et al. 2023 (Long-document modeling)
- Semantic similarity from Rashkin et al. 2021 (Evidence attribution)
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import numpy as np


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    chapter: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """Evidence snippet with provenance"""
    text: str
    source: str
    chunk_id: str
    relevance_score: float
    location: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_text_semantic(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200,
    preserve_sentences: bool = True
) -> List[TextChunk]:
    """
    Chunk text with semantic boundaries preservation.
    
    Based on Lu et al. 2023 - maintains sentence/paragraph boundaries
    for better semantic coherence.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size in characters
        overlap: Overlap between chunks
        preserve_sentences: Whether to preserve sentence boundaries
    
    Returns:
        List of TextChunk objects
    """
    if preserve_sentences:
        # Split into sentences
        sentences = re.split(r'([.!?]+[\s\n]+)', text)
        # Recombine sentence with its punctuation
        sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
        if len(sentences) % 2 == 1:
            sentences.append(sentences[-1])
    else:
        sentences = [text]
    
    chunks = []
    current_chunk = ""
    current_start = 0
    chunk_idx = 0
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            # Create chunk
            chunk_id = generate_chunk_id(current_chunk, chunk_idx)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + sentence
            current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence)
            chunk_idx += 1
        else:
            current_chunk += sentence
    
    # Add final chunk
    if current_chunk:
        chunk_id = generate_chunk_id(current_chunk, chunk_idx)
        chunks.append(TextChunk(
            chunk_id=chunk_id,
            text=current_chunk.strip(),
            start_char=current_start,
            end_char=current_start + len(current_chunk)
        ))
    
    return chunks


def generate_chunk_id(text: str, idx: int) -> str:
    """Generate unique chunk ID from content and index"""
    hash_obj = hashlib.md5(text.encode())
    return f"chunk_{idx}_{hash_obj.hexdigest()[:8]}"


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract named entities from text.
    Simple rule-based extraction for demonstration.
    In production, use spaCy or similar NLP library.
    
    Returns:
        List of entities with type and text
    """
    entities = []
    
    # Capitalized words (potential names)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for name in set(names):
        if len(name.split()) <= 3:  # Reasonable name length
            entities.append({
                'text': name,
                'type': 'PERSON',
                'start': text.find(name)
            })
    
    # Quoted locations (common in novels)
    locations = re.findall(r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
    for loc in set(locations):
        entities.append({
            'text': loc,
            'type': 'LOCATION',
            'start': text.find(loc)
        })
    
    return entities


def extract_temporal_markers(text: str) -> List[Dict[str, Any]]:
    """
    Extract temporal expressions from text.
    Based on Sun et al. 2013 - Temporal constraint tracking.
    
    Returns:
        List of temporal markers with text and type
    """
    markers = []
    
    # Absolute time patterns
    absolute_patterns = [
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'ABSOLUTE_DATE'),
        (r'\b\d{4}\b', 'YEAR'),
        (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'DAY_OF_WEEK'),
    ]
    
    # Relative time patterns
    relative_patterns = [
        (r'\b(before|after|during|since|until|while)\b', 'RELATIVE'),
        (r'\b(\d+\s+(?:years?|months?|weeks?|days?|hours?|minutes?)\s+(?:ago|later|earlier))\b', 'DURATION'),
        (r'\b(yesterday|today|tomorrow|now|then|previously|subsequently)\b', 'DEICTIC'),
    ]
    
    for pattern, marker_type in absolute_patterns + relative_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            markers.append({
                'text': match.group(0),
                'type': marker_type,
                'start': match.start(),
                'end': match.end()
            })
    
    return markers


def extract_causal_markers(text: str) -> List[Dict[str, Any]]:
    """
    Extract causal connectives and markers.
    Based on Feder et al. 2022 - Causal inference in NLP.
    
    Returns:
        List of causal markers with text and type
    """
    markers = []
    
    causal_patterns = [
        (r'\b(because|since|as|due to|owing to)\b', 'CAUSE'),
        (r'\b(therefore|thus|hence|consequently|as a result|so)\b', 'EFFECT'),
        (r'\b(if|when|unless|provided that)\b', 'CONDITIONAL'),
        (r'\b(caused|led to|resulted in|brought about|triggered)\b', 'CAUSAL_VERB'),
    ]
    
    for pattern, marker_type in causal_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            markers.append({
                'text': match.group(0),
                'type': marker_type,
                'start': match.start(),
                'end': match.end()
            })
    
    return markers


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def load_csv_data(filepath: str) -> List[Dict[str, Any]]:
    """Load CSV data (train or test set)"""
    import csv
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    
    return data


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to CSV format for submission"""
    import csv
    
    if not results:
        return
    
    # Only output story_id and prediction columns
    fieldnames = ['story_id', 'prediction']
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def format_rationale(
    evidence: List[Evidence], 
    reasoning: Dict[str, Any],
    max_length: int = 500
) -> str:
    """
    Format explanation rationale for human judges.
    Based on Rashkin et al. 2021 - Evidence attribution.
    
    Args:
        evidence: List of evidence snippets
        reasoning: Dict with reasoning components
        max_length: Maximum rationale length
    
    Returns:
        Formatted rationale string
    """
    parts = []
    
    # Add key evidence
    if evidence:
        parts.append("Evidence:")
        for i, ev in enumerate(evidence[:3], 1):  # Top 3 pieces
            excerpt = ev.text[:100] + "..." if len(ev.text) > 100 else ev.text
            parts.append(f"  {i}. {excerpt} [{ev.location}]")
    
    # Add reasoning summary
    if 'temporal_conflicts' in reasoning and reasoning['temporal_conflicts']:
        parts.append(f"Temporal issues: {len(reasoning['temporal_conflicts'])} conflicts")
    
    if 'causal_conflicts' in reasoning and reasoning['causal_conflicts']:
        parts.append(f"Causal issues: {len(reasoning['causal_conflicts'])} conflicts")
    
    if 'entity_conflicts' in reasoning and reasoning['entity_conflicts']:
        parts.append(f"Entity conflicts: {len(reasoning['entity_conflicts'])} mismatches")
    
    rationale = " | ".join(parts)
    
    # Truncate if needed
    if len(rationale) > max_length:
        rationale = rationale[:max_length-3] + "..."
    
    return rationale


def log_experiment(
    experiment_name: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    output_dir: str = "results"
):
    """Log experiment configuration and results"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
    
    log_data = {
        'experiment': experiment_name,
        'timestamp': timestamp,
        'config': config,
        'results': results
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_file


def calculate_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }


def print_section(title: str, char: str = "="):
    """Print formatted section header"""
    print(f"\n{char * 60}")
    print(f"{title.center(60)}")
    print(f"{char * 60}\n")
