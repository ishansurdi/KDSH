"""
Evidence-Grounded Inconsistency Scorer

Aggregates multiple signals into an inconsistency score:
- Temporal conflicts
- Causal conflicts
- Entity mismatches
- Semantic contradictions
- Evidence strength

Research foundation:
- Lattimer et al. 2023 (Long-doc inconsistency)
- Rashkin et al. 2021 (Evidence attribution)
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class InconsistencyScorer:
    """
    Multi-dimensional inconsistency scoring.
    
    Combines:
    1. Temporal consistency score
    2. Causal consistency score
    3. Entity consistency score
    4. Semantic contradiction score
    5. Evidence coverage score
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        """
        Initialize scorer with component weights.
        
        Args:
            weights: Dict of component weights (default: AGGRESSIVE weighting)
        """
        # AGGRESSIVE: Higher weights for conflicts, lower for evidence
        self.weights = weights or {
            'temporal': 0.40,   # CRITICAL: temporal conflicts are PRIMARY signal
            'causal': 0.35,     # CRITICAL: causal conflicts are key
            'entity': 0.10,     # DECREASED - less important than conflicts
            'semantic': 0.10,   # DECREASED - less important than conflicts
            'evidence': 0.05    # CRITICAL: Low evidence matters less than conflicts
        }
    
    def score_claim(
        self,
        claim: Any,
        evidence: List,
        temporal_conflicts: List,
        causal_conflicts: List,
        memory
    ) -> Dict[str, Any]:
        """
        Score a single claim for inconsistency.
        
        Args:
            claim: Claim object
            evidence: Evidence list for claim
            temporal_conflicts: List of temporal conflicts
            causal_conflicts: List of causal conflicts
            memory: HierarchicalNarrativeMemory
        
        Returns:
            Dict with scores and explanation
        """
        # Calculate component scores
        temporal_score = self._score_temporal(claim, temporal_conflicts)
        causal_score = self._score_causal(claim, causal_conflicts)
        entity_score = self._score_entity(claim, evidence, memory)
        semantic_score = self._score_semantic(claim, evidence)
        evidence_score = self._score_evidence(claim, evidence)
        
        # Weighted combination
        inconsistency_score = (
            self.weights['temporal'] * temporal_score +
            self.weights['causal'] * causal_score +
            self.weights['entity'] * entity_score +
            self.weights['semantic'] * semantic_score +
            self.weights['evidence'] * evidence_score
        )
        
        return {
            'claim_id': claim.claim_id,
            'overall_inconsistency': inconsistency_score,
            'components': {
                'temporal': temporal_score,
                'causal': causal_score,
                'entity': entity_score,
                'semantic': semantic_score,
                'evidence': evidence_score
            },
            'is_consistent': inconsistency_score < 0.5
        }
    
    def score_backstory(
        self,
        claims: List,
        evidence_map: Dict[str, List],
        temporal_conflicts: List,
        causal_conflicts: List,
        memory
    ) -> Dict[str, Any]:
        """
        Score entire backstory for consistency.
        
        Args:
            claims: List of all claims
            evidence_map: Evidence for each claim
            temporal_conflicts: All temporal conflicts
            causal_conflicts: All causal conflicts
            memory: HierarchicalNarrativeMemory
        
        Returns:
            Dict with overall score and per-claim breakdown
        """
        claim_scores = []
        
        for claim in claims:
            evidence = evidence_map.get(claim.claim_id, [])
            score = self.score_claim(
                claim,
                evidence,
                temporal_conflicts,
                causal_conflicts,
                memory
            )
            claim_scores.append(score)
        
        # AGGRESSIVE: More aggressive aggregation - worst conflict matters MOST
        overall_scores = [s['overall_inconsistency'] for s in claim_scores]
        
        # Initialize defaults
        total_conflicts = len(temporal_conflicts) + len(causal_conflicts)
        
        if overall_scores:
            # Use BOTH max and average for better discrimination
            max_inconsistency = max(overall_scores)
            avg_inconsistency = np.mean(overall_scores)
            median_inconsistency = np.median(overall_scores)
            
            # CRITICAL: AGGRESSIVE aggregation - worst case dominates
            overall_inconsistency = (
                max_inconsistency * 0.75 +  # CRITICAL: Maximum emphasis on worst conflict
                avg_inconsistency * 0.25     # Minimal averaging
            )
            
            # IMPROVED: Boost inconsistency if multiple claims are problematic
            inconsistent_claim_count = sum(1 for s in claim_scores if s['overall_inconsistency'] > 0.5)
            if inconsistent_claim_count >= 3:
                overall_inconsistency = min(overall_inconsistency + 0.25, 1.0)
            elif inconsistent_claim_count >= 2:
                overall_inconsistency = min(overall_inconsistency + 0.15, 1.0)
            
            # CRITICAL: Direct conflict penalty - any conflict is a strong signal
            if total_conflicts >= 2:
                overall_inconsistency = min(overall_inconsistency + 0.20, 1.0)
            elif total_conflicts >= 1:
                overall_inconsistency = min(overall_inconsistency + 0.10, 1.0)
            
        else:
            overall_inconsistency = 0.0
            avg_inconsistency = 0.0
            max_inconsistency = 0.0
        
        return {
            'overall_inconsistency': overall_inconsistency,
            'average_inconsistency': avg_inconsistency,
            'max_inconsistency': max_inconsistency,
            'claim_scores': claim_scores,
            'is_consistent': overall_inconsistency < 0.5,
            'num_inconsistent_claims': sum(1 for s in claim_scores if not s['is_consistent']),
            'total_conflicts': total_conflicts
        }
    
    def _score_temporal(
        self,
        claim: Any,
        temporal_conflicts: List
    ) -> float:
        """Score temporal consistency (0 = consistent, 1 = very inconsistent)"""
        # Count conflicts involving this claim
        relevant_conflicts = []
        
        for conflict in temporal_conflicts:
            # Check if conflict involves this claim
            conflict_dict = conflict.to_dict()
            event_ids = [conflict_dict.get('event1_id'), conflict_dict.get('event2_id')]
            
            # IMPROVED: Better matching of conflicts to claims
            if (claim.claim_id in str(conflict_dict) or 
                any(claim.claim_id in str(eid) for eid in event_ids if eid)):
                relevant_conflicts.append(conflict)
        
        if not relevant_conflicts:
            return 0.0  # No conflicts = consistent
        
        # AGGRESSIVE: Weight by severity MUCH more aggressively
        max_severity = max(c.severity for c in relevant_conflicts)
        avg_severity = np.mean([c.severity for c in relevant_conflicts])
        
        # AGGRESSIVE: Even ONE high-severity temporal conflict should matter
        num_conflicts_factor = min(len(relevant_conflicts) / 2, 1.0)  # Changed from /3 to /2
        
        # AGGRESSIVE: Combine with more weight on max severity
        # Changed from 0.7/0.3 to 0.8/0.2 - one bad conflict is enough!
        inconsistency = max_severity * 0.8 + num_conflicts_factor * 0.2
        
        # AGGRESSIVE: Boost score if multiple high-severity conflicts
        if len(relevant_conflicts) >= 2 and max_severity > 0.7:
            inconsistency = min(inconsistency + 0.15, 1.0)  # Add bonus penalty
        
        return min(inconsistency, 1.0)
    
    def _score_causal(
        self,
        claim: Any,
        causal_conflicts: List
    ) -> float:
        """Score causal consistency AGGRESSIVELY"""
        # Check all claims for causal issues, not just causal-type claims
        # Any claim can have causal implications
        
        # Count conflicts
        relevant_conflicts = [
            c for c in causal_conflicts
            if c.claim_id == claim.claim_id
        ]
        
        if not relevant_conflicts:
            # AGGRESSIVE: Causal claims without conflicts still suspicious
            if claim.claim_type == 'causal':
                return 0.3  # INCREASED from 0.2 - causal claims need strong evidence
            return 0.0
        
        # AGGRESSIVE: Weight by both severity and count with higher penalties
        max_severity = max(c.severity for c in relevant_conflicts)
        avg_severity = np.mean([c.severity for c in relevant_conflicts])
        num_conflicts = len(relevant_conflicts)
        
        # AGGRESSIVE: Multiple causal conflicts are VERY problematic
        if num_conflicts >= 2:
            return min(max_severity + 0.3, 1.0)  # INCREASED from 0.2
        
        # AGGRESSIVE: Even single causal conflict matters more
        return min(max_severity * 1.1, 1.0)  # Amplify severity by 10%
    
    def _score_entity(
        self,
        claim: Any,
        evidence: List,
        memory
    ) -> float:
        """Score entity consistency"""
        if not claim.entities:
            return 0.0  # No entities to check
        
        inconsistencies = 0
        total_checks = 0
        
        for entity in claim.entities:
            # Check entity consistency in memory
            consistency_check = memory.check_character_consistency(
                entity,
                claim.text
            )
            
            total_checks += 1
            
            if not consistency_check['is_consistent']:
                inconsistencies += len(consistency_check['conflicts'])
        
        if total_checks == 0:
            return 0.0
        
        # Normalize
        return min(inconsistencies / total_checks, 1.0)
    
    def _score_semantic(
        self,
        claim: Any,
        evidence: List
    ) -> float:
        """Score semantic contradiction AGGRESSIVELY"""
        if not evidence:
            return 0.6  # INCREASED from 0.5 - No evidence = higher concern
        
        # AGGRESSIVE: Expanded contradiction keywords
        contradiction_signals = [
            'not', 'never', 'no', "didn't", "wasn't", "weren't", "isn't",
            'contradiction', 'impossible', 'incorrect', 'false', 'untrue',
            'contrary', 'opposite', 'different from', 'rather than',
            'instead', 'actually', 'however', 'but'
        ]
        
        contradiction_count = 0
        strong_contradiction_count = 0
        
        for ev in evidence:
            text_lower = ev.text.lower()
            
            # Count contradiction signals
            for signal in contradiction_signals:
                if signal in text_lower:
                    contradiction_count += 1
        
        if contradiction_count == 0:
            return 0.0
        
        # Normalize by evidence count
        return min(contradiction_count / len(evidence), 1.0)
    
    def _score_evidence(
        self,
        claim: Any,
        evidence: List
    ) -> float:
        """Score evidence quality and coverage"""
        if not evidence:
            return 1.0  # No evidence = very inconsistent
        
        # Check evidence strength
        avg_score = np.mean([ev.score for ev in evidence])
        
        # Check evidence count
        count_penalty = 0.0
        if len(evidence) < 2:
            count_penalty = 0.3
        elif len(evidence) < 3:
            count_penalty = 0.1
        
        # Lower score = better evidence
        # Higher inconsistency = worse evidence
        evidence_inconsistency = (1.0 - avg_score) + count_penalty
        
        return min(evidence_inconsistency, 1.0)
    
    def calibrate_weights(
        self,
        train_data: List[Dict[str, Any]],
        train_labels: List[int]
    ):
        """
        Calibrate component weights using training data.
        
        Args:
            train_data: List of training examples with scores
            train_labels: True labels (1 = consistent, 0 = inconsistent)
        """
        # Simple weight optimization (in production, use proper ML)
        # This is a placeholder for demonstration
        
        # Try different weight combinations
        best_accuracy = 0.0
        best_weights = self.weights.copy()
        
        weight_options = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        
        for temp_w in weight_options:
            for caus_w in weight_options:
                remaining = 1.0 - temp_w - caus_w
                if remaining < 0:
                    continue
                
                # Distribute remaining weight
                ent_w = remaining * 0.4
                sem_w = remaining * 0.4
                ev_w = remaining * 0.2
                
                test_weights = {
                    'temporal': temp_w,
                    'causal': caus_w,
                    'entity': ent_w,
                    'semantic': sem_w,
                    'evidence': ev_w
                }
                
                # Evaluate on training data
                correct = 0
                for data, label in zip(train_data, train_labels):
                    # Calculate score with test weights
                    score = (
                        test_weights['temporal'] * data.get('temporal', 0) +
                        test_weights['causal'] * data.get('causal', 0) +
                        test_weights['entity'] * data.get('entity', 0) +
                        test_weights['semantic'] * data.get('semantic', 0) +
                        test_weights['evidence'] * data.get('evidence', 0)
                    )
                    
                    pred = 1 if score < 0.5 else 0
                    if pred == label:
                        correct += 1
                
                accuracy = correct / len(train_labels)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = test_weights
        
        self.weights = best_weights
        print(f"✓ Calibrated weights: accuracy = {best_accuracy:.3f}")
        print(f"  Weights: {self.weights}")
