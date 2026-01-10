"""
Core Package for Narrative Consistency System

This package implements a complete long-context reasoning system
for checking backstory consistency with novels.
"""

from .utils import *
from .pathway_store import PathwayDocumentStore, PathwayVectorStore
from .memory import HierarchicalNarrativeMemory, Scene, Episode, CharacterState
from .claims import ClaimExtractor, Claim
from .constraints import ConstraintBuilder, ConstraintGraph, Constraint
from .retriever import MultiHopRetriever, RetrievalResult
from .causal import CausalReasoningEngine, CausalLink, CausalConflict
from .temporal import TemporalReasoningEngine, TemporalEvent, TemporalConflict
from .scorer import InconsistencyScorer
from .classifier import ConsistencyClassifier

__version__ = "1.0.0"

__all__ = [
    # Utils
    'TextChunk',
    'Evidence',
    'chunk_text_semantic',
    'extract_entities',
    'extract_temporal_markers',
    'extract_causal_markers',
    'load_csv_data',
    'save_results',
    'format_rationale',
    
    # Pathway Store
    'PathwayDocumentStore',
    'PathwayVectorStore',
    
    # Memory
    'HierarchicalNarrativeMemory',
    'Scene',
    'Episode',
    'CharacterState',
    
    # Claims
    'ClaimExtractor',
    'Claim',
    
    # Constraints
    'ConstraintBuilder',
    'ConstraintGraph',
    'Constraint',
    
    # Retriever
    'MultiHopRetriever',
    'RetrievalResult',
    
    # Reasoning Engines
    'CausalReasoningEngine',
    'CausalLink',
    'CausalConflict',
    'TemporalReasoningEngine',
    'TemporalEvent',
    'TemporalConflict',
    
    # Scorer
    'InconsistencyScorer',
    
    # Classifier
    'ConsistencyClassifier',
]
