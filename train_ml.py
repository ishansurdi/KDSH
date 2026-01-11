"""
Train Machine Learning Classifier on Training Data

This script:
1. Loads training data
2. Runs full pipeline to extract features
3. Trains ML ensemble
4. Saves trained models
"""

import sys
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from core import (
    PathwayDocumentStore, HierarchicalNarrativeMemory,
    ClaimExtractor, ConstraintBuilder, MultiHopRetriever,
    CausalReasoningEngine, TemporalReasoningEngine,
    InconsistencyScorer, load_csv_data
)
from core.ml_classifier import MLConsistencyClassifier


def main():
    """Train ML classifier on training data"""
    
    print("=" * 80)
    print("TRAINING ML CLASSIFIER")
    print("=" * 80)
    
    # Load training data
    print("\n[1] Loading training data...")
    train_data = load_csv_data('data/train.csv')
    print(f"Loaded {len(train_data)} training examples")
    
    # Load novels
    print("\n[2] Loading novels...")
    novels_dir = Path('data/novels')
    documents = []
    for novel_file in novels_dir.glob('*.txt'):
        with open(novel_file, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'id': novel_file.stem,
                'title': novel_file.stem,
                'content': content
            })
    print(f"Loaded {len(documents)} novels")
    
    # Initialize pipeline components
    print("\n[3] Initializing pipeline...")
    doc_store = PathwayDocumentStore(embedding_model=None, chunk_size=1000)
    memory = HierarchicalNarrativeMemory()
    claim_extractor = ClaimExtractor()
    constraint_builder = ConstraintBuilder()
    retriever = MultiHopRetriever(doc_store, max_hops=3, top_k=5)
    temporal_engine = TemporalReasoningEngine()
    causal_engine = CausalReasoningEngine()
    scorer = InconsistencyScorer()
    
    # Index documents
    print("\n[4] Indexing documents...")
    for doc in tqdm(documents, desc="Indexing"):
        chunks = doc_store.add_document(doc['content'], doc['id'])
        memory.add_narrative(doc['content'], doc['id'], doc['title'])
    
    # Extract features from training data
    print("\n[5] Extracting features from training data...")
    X_train = []
    y_train = []
    
    ml_classifier = MLConsistencyClassifier()
    
    for idx, example in enumerate(tqdm(train_data, desc="Processing training examples")):
        try:
            # Extract claims
            claims = claim_extractor.extract_claims(
                example['content'],
                example['char'],
                example['book_name']
            )
            
            # Build constraints
            constraints = constraint_builder.build_constraints(
                claims,
                example['book_name']
            )
            
            # Retrieve evidence
            evidence_map = {}
            for claim in claims:
                evidence = retriever.retrieve(
                    claim.text,
                    book_name=example['book_name']
                )
                evidence_map[claim.id] = evidence
            
            # Temporal reasoning
            temporal_conflicts = temporal_engine.check_conflicts(
                claims,
                constraints,
                example['book_name']
            )
            
            # Causal reasoning
            causal_conflicts = causal_engine.check_conflicts(
                claims,
                constraints,
                example['book_name']
            )
            
            # Score
            score_result = scorer.score(
                claims=claims,
                temporal_conflicts=temporal_conflicts,
                causal_conflicts=causal_conflicts,
                evidence_map=evidence_map,
                constraints=constraints
            )
            
            # Extract features
            features = ml_classifier.extract_features(
                score_result['inconsistency_score'],
                temporal_conflicts,
                causal_conflicts,
                evidence_map,
                claims,
                score_result['component_scores']
            )
            
            X_train.append(features)
            
            # Label: 1 = consistent, 0 = inconsistent/contradict
            label = 1 if example['label'] == 'consistent' else 0
            y_train.append(label)
            
        except Exception as e:
            print(f"\nError processing example {idx}: {e}")
            continue
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\n[6] Training set shape: {X_train.shape}")
    print(f"    Consistent: {np.sum(y_train == 1)}")
    print(f"    Inconsistent: {np.sum(y_train == 0)}")
    
    # Train models
    print("\n[7] Training ML models...")
    cv_scores = ml_classifier.train(X_train, y_train)
    
    print("\n[8] Cross-validation results:")
    for model_name, score in cv_scores.items():
        print(f"    {model_name}: {score:.3f}")
    
    # Save trained classifier
    print("\n[9] Saving trained models...")
    with open('results/ml_classifier.pkl', 'wb') as f:
        pickle.dump(ml_classifier, f)
    print("    Saved to results/ml_classifier.pkl")
    
    # Feature importance
    print("\n[10] Feature importance (top 10):")
    feature_names = [
        'inconsistency_score', 'num_temporal', 'num_causal', 'total_conflicts',
        'max_temporal_sev', 'avg_temporal_sev', 'std_temporal_sev',
        'max_causal_sev', 'avg_causal_sev', 'std_causal_sev',
        'num_claims', 'evidence_coverage',
        'temporal_component', 'causal_component', 'entity_component',
        'semantic_component', 'evidence_component', 'reasoning_component',
        'score_x_conflicts', 'score_x_claims'
    ]
    
    importances = ml_classifier.get_feature_importance(feature_names)
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:10]:
        print(f"    {feat}: {imp:.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
