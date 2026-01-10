"""
Main Entry Point for KDSH System

Run this script to process test data and generate results.csv

Usage:
    python main.py --test data/test.csv --output results/results.csv
    python main.py --train data/train.csv --calibrate
"""

import argparse
import sys
import os
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    PathwayDocumentStore, HierarchicalNarrativeMemory,
    ClaimExtractor, ConstraintBuilder, MultiHopRetriever,
    CausalReasoningEngine, TemporalReasoningEngine,
    InconsistencyScorer, ConsistencyClassifier,
    load_csv_data, save_results, calculate_metrics, print_section
)
from tqdm import tqdm


class NarrativeConsistencyPipeline:
    """Complete pipeline for narrative consistency checking"""
    
    def __init__(self, config=None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dict with pipeline parameters
        """
        self.config = config or {
            'chunk_size': 1000,
            'max_hops': 3,
            'top_k_evidence': 5,
            'threshold': 0.5
        }
        
        # Initialize components
        self.document_store = PathwayDocumentStore(
            embedding_model=None,
            chunk_size=self.config['chunk_size']
        )
        self.claim_extractor = ClaimExtractor()
        self.constraint_builder = ConstraintBuilder()
        self.scorer = InconsistencyScorer()
        self.classifier = ConsistencyClassifier(
            threshold=self.config['threshold']
        )
        
        self.novel_cache = {}
        
    def process_example(self, story_id, novel_file, backstory):
        """
        Process a single example through the pipeline.
        
        Args:
            story_id: Unique identifier
            novel_file: Path to novel file
            backstory: Hypothetical backstory
        
        Returns:
            Dict with prediction, confidence, rationale
        """
        try:
            # Load/ingest novel (with caching)
            if novel_file not in self.novel_cache:
                novel_path = Path('data/novels') / novel_file
                
                if not novel_path.exists():
                    raise FileNotFoundError(f"Novel not found: {novel_path}")
                
                with open(novel_path, 'r', encoding='utf-8') as f:
                    novel_text = f.read()
                
                # Ingest
                self.document_store.ingest_novel(
                    novel_text=novel_text,
                    novel_id=novel_file,
                    metadata={'filename': novel_file}
                )
                self.novel_cache[novel_file] = True
            
            # Build memory
            chunks = []
            for chunk_id, doc in self.document_store.documents.items():
                if self.document_store.chunk_to_doc.get(chunk_id) == novel_file:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': doc.text,
                        'metadata': doc.metadata
                    })
            
            memory = HierarchicalNarrativeMemory()
            memory.extract_narrative_from_chunks(chunks, novel_file)
            
            # Extract claims
            claims = self.claim_extractor.extract_claims(backstory)
            
            # Build constraints
            constraint_graph = self.constraint_builder.build_graph(claims)
            
            # Retrieve evidence
            retriever = MultiHopRetriever(
                self.document_store,
                max_hops=self.config['max_hops']
            )
            evidence_map = retriever.retrieve_for_claims(
                claims=claims,
                novel_id=novel_file,
                top_k_per_claim=self.config['top_k_evidence']
            )
            
            # Reasoning
            causal_engine = CausalReasoningEngine(memory, constraint_graph)
            temporal_engine = TemporalReasoningEngine(memory, constraint_graph)
            
            temporal_engine.build_timeline(claims, evidence_map)
            temporal_conflicts = temporal_engine.check_temporal_consistency(
                claims, evidence_map
            )
            causal_conflicts = causal_engine.check_causal_consistency(
                claims, evidence_map
            )
            
            # Scoring
            score_result = self.scorer.score_backstory(
                claims=claims,
                evidence_map=evidence_map,
                temporal_conflicts=temporal_conflicts,
                causal_conflicts=causal_conflicts,
                memory=memory
            )
            
            # Classification
            classification = self.classifier.classify(
                inconsistency_score=score_result['overall_inconsistency'],
                temporal_conflicts=temporal_conflicts,
                causal_conflicts=causal_conflicts,
                evidence_map=evidence_map,
                claims=claims
            )
            
            return {
                'story_id': story_id,
                'prediction': classification['prediction'],
                'confidence': classification['confidence'],
                'rationale': classification['rationale']
            }
            
        except Exception as e:
            print(f"Error processing {story_id}: {str(e)}")
            return {
                'story_id': story_id,
                'prediction': 0,
                'confidence': 0.5,
                'rationale': f"Processing error: {str(e)[:200]}"
            }
    
    def run_on_dataset(self, data_path, output_path=None):
        """
        Run pipeline on entire dataset.
        
        Args:
            data_path: Path to CSV file
            output_path: Path to save results (optional)
        
        Returns:
            List of results
        """
        # Load data
        data = load_csv_data(data_path)
        print(f"\n✓ Loaded {len(data)} examples from {data_path}")
        
        # Process each example
        results = []
        
        print("\nProcessing examples...")
        for example in tqdm(data):
            result = self.process_example(
                story_id=example['story_id'],
                novel_file=example['novel_file'],
                backstory=example['backstory']
            )
            results.append(result)
        
        # Save results if output path provided
        if output_path:
            save_results(results, output_path)
            print(f"\n✓ Saved results to {output_path}")
        
        return results
    
    def calibrate(self, train_data_path):
        """
        Calibrate classifier on training data.
        
        Args:
            train_data_path: Path to training CSV
        """
        print("\nCalibrating on training data...")
        
        # Process training examples
        train_data = load_csv_data(train_data_path)
        
        scores = []
        labels = []
        
        for example in tqdm(train_data):
            result = self.process_example(
                story_id=example['story_id'],
                novel_file=example['novel_file'],
                backstory=example['backstory']
            )
            
            # Extract inconsistency score (would need to store this)
            # For now, use prediction as proxy
            scores.append(0.3 if result['prediction'] == 1 else 0.7)
            labels.append(int(example['label']))
        
        # Calibrate threshold
        self.classifier.calibrate_threshold(scores, labels)
        
        # Calculate metrics
        predictions = [r['prediction'] for r in [
            self.process_example(e['story_id'], e['novel_file'], e['backstory'])
            for e in train_data
        ]]
        
        metrics = calculate_metrics(predictions, labels)
        
        print("\nCalibration Results:")
        print("=" * 60)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")
        
        return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='KDSH Narrative Consistency System'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Path to test CSV file'
    )
    
    parser.add_argument(
        '--train',
        type=str,
        help='Path to train CSV file (for calibration)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/results.csv',
        help='Output path for results CSV'
    )
    
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Run calibration on training data'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for document splitting'
    )
    
    parser.add_argument(
        '--max-hops',
        type=int,
        default=3,
        help='Maximum hops for evidence retrieval'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_section("KDSH NARRATIVE CONSISTENCY SYSTEM")
    
    # Initialize pipeline
    config = {
        'chunk_size': args.chunk_size,
        'max_hops': args.max_hops,
        'top_k_evidence': 5,
        'threshold': args.threshold
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    pipeline = NarrativeConsistencyPipeline(config)
    
    # Run calibration if requested
    if args.calibrate and args.train:
        pipeline.calibrate(args.train)
    
    # Run on test set
    if args.test:
        results = pipeline.run_on_dataset(args.test, args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total processed: {len(results)}")
        print(f"Consistent (1): {sum(1 for r in results if r['prediction'] == 1)}")
        print(f"Inconsistent (0): {sum(1 for r in results if r['prediction'] == 0)}")
        
        if results:
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            print(f"Average confidence: {avg_conf:.2%}")
    
    # If no args, print usage
    if not args.test and not args.train:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --test data/test.csv --output results/results.csv")
        print("  python main.py --train data/train.csv --calibrate")


if __name__ == '__main__':
    main()
