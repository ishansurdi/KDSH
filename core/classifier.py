"""
Final Classifier with Rationale Generation

Rule-guided classifier that:
- Makes binary decision (consistent/inconsistent)
- Outputs confidence score
- Generates human-readable rationale

Research foundation:
- Zhu et al. 2025 (Process-supervised reasoning)
- Rashkin et al. 2021 (Evidence attribution)
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class ConsistencyClassifier:
    """
    Final classifier for backstory consistency.
    
    Makes binary decision based on:
    1. Inconsistency score
    2. Conflict severity
    3. Evidence quality
    4. Reasoning confidence
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        confidence_model: str = 'rule_based'
    ):
        """
        Initialize classifier.
        
        Args:
            threshold: Inconsistency threshold for classification
            confidence_model: Method for confidence estimation
        """
        self.threshold = threshold
        self.confidence_model = confidence_model
        self.calibration_params = {
            'threshold': threshold,
            'severity_weight': 0.3,
            'evidence_weight': 0.2
        }
    
    def classify(
        self,
        inconsistency_score: float,
        temporal_conflicts: List,
        causal_conflicts: List,
        evidence_map: Dict[str, List],
        claims: List
    ) -> Dict[str, Any]:
        """
        Classify backstory consistency AGGRESSIVELY.
        
        Args:
            inconsistency_score: Overall inconsistency score
            temporal_conflicts: List of temporal conflicts
            causal_conflicts: List of causal conflicts
            evidence_map: Evidence for claims
            claims: List of claims
        
        Returns:
            Dict with prediction, confidence, and rationale
        """
        # AGGRESSIVE: Use MULTIPLE signals for classification, not just threshold
        base_prediction = 1 if inconsistency_score < self.threshold else 0
        
        # AGGRESSIVE: Override to inconsistent if HIGH-SEVERITY conflicts exist
        high_severity_temporal = [c for c in temporal_conflicts if c.severity > 0.85]
        high_severity_causal = [c for c in causal_conflicts if c.severity > 0.85]
        
        # Even if score is below threshold, classify as inconsistent if multiple high-severity conflicts
        if len(high_severity_temporal) >= 2 or len(high_severity_causal) >= 2:
            prediction = 0  # Inconsistent
        elif len(high_severity_temporal) >= 1 and len(high_severity_causal) >= 1:
            prediction = 0  # Inconsistent (one of each type)
        else:
            prediction = base_prediction
        
        # AGGRESSIVE: Also override if score is close to threshold and conflicts exist
        if base_prediction == 1 and inconsistency_score > (self.threshold * 0.85):
            if len(temporal_conflicts) + len(causal_conflicts) >= 3:
                prediction = 0  # Too many conflicts, even if score barely passes
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            inconsistency_score,
            temporal_conflicts,
            causal_conflicts,
            evidence_map
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            prediction,
            inconsistency_score,
            temporal_conflicts,
            causal_conflicts,
            evidence_map,
            claims
        )
        
        # CRITICAL: FINAL OVERRIDE RULES - Red flags that force inconsistent classification
        high_severity_temporal = [c for c in temporal_conflicts if c.severity > 0.85]
        high_severity_causal = [c for c in causal_conflicts if c.severity > 0.85]
        
        # RULE 1: ANY single high-severity temporal conflict = inconsistent
        if len(high_severity_temporal) >= 1:
            prediction = 0
            confidence = min(confidence + 0.15, 0.95)
        
        # RULE 2: Multiple moderate conflicts = inconsistent
        total_conflicts = len(temporal_conflicts) + len(causal_conflicts)
        if total_conflicts >= 3:
            if inconsistency_score > 0.25:  # Even if below threshold
                prediction = 0
                confidence = min(confidence + 0.10, 0.95)
        
        # RULE 3: Mixed conflict types = very suspicious
        if len(temporal_conflicts) >= 1 and len(causal_conflicts) >= 1:
            if inconsistency_score > 0.28:
                prediction = 0
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'rationale': rationale,
            'inconsistency_score': inconsistency_score,
            'num_conflicts': len(temporal_conflicts) + len(causal_conflicts)
        }
    
    def classify_batch(
        self,
        backstory_scores: List[Dict[str, Any]],
        all_conflicts: List[Dict[str, Any]],
        all_evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple backstories.
        
        Args:
            backstory_scores: List of inconsistency scores
            all_conflicts: List of conflict data
            all_evidence: List of evidence data
        
        Returns:
            List of classification results
        """
        results = []
        
        for i, score_data in enumerate(backstory_scores):
            conflicts = all_conflicts[i] if i < len(all_conflicts) else {}
            evidence = all_evidence[i] if i < len(all_evidence) else {}
            
            result = self.classify(
                inconsistency_score=score_data.get('overall_inconsistency', 0.5),
                temporal_conflicts=conflicts.get('temporal', []),
                causal_conflicts=conflicts.get('causal', []),
                evidence_map=evidence.get('evidence_map', {}),
                claims=score_data.get('claims', [])
            )
            
            results.append(result)
        
        return results
    
    def calibrate_threshold(
        self,
        train_scores: List[float],
        train_labels: List[int]
    ):
        """
        Calibrate decision threshold using training data.
        
        Args:
            train_scores: List of inconsistency scores
            train_labels: List of true labels (1 = consistent, 0 = inconsistent)
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        # Try different thresholds
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.arange(0.3, 0.7, 0.05):
            predictions = [1 if score < threshold else 0 for score in train_scores]
            f1 = f1_score(train_labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.calibration_params['threshold'] = best_threshold
        
        # Calculate final metrics
        final_predictions = [1 if score < best_threshold else 0 for score in train_scores]
        accuracy = accuracy_score(train_labels, final_predictions)
        
        print(f"✓ Calibrated threshold: {best_threshold:.3f}")
        print(f"  Training accuracy: {accuracy:.3f}")
        print(f"  Training F1: {best_f1:.3f}")
    
    def _calculate_confidence(
        self,
        inconsistency_score: float,
        temporal_conflicts: List,
        causal_conflicts: List,
        evidence_map: Dict[str, List]
    ) -> float:
        """Calculate confidence in prediction - IMPROVED for better discrimination"""
        # Distance from threshold
        distance = abs(inconsistency_score - self.threshold)
        distance_confidence = min(distance * 2.5, 1.0)  # TUNED: More aggressive scaling
        
        # IMPROVED: Conflict clarity with better weighting
        num_conflicts = len(temporal_conflicts) + len(causal_conflicts)
        if num_conflicts == 0:
            conflict_confidence = 0.85  # Clear: no conflicts (high confidence consistent)
        elif num_conflicts >= 5:
            conflict_confidence = 0.9  # Clear: many conflicts (high confidence inconsistent)
        elif num_conflicts >= 3:
            conflict_confidence = 0.8  # Multiple conflicts
        elif num_conflicts >= 1:
            conflict_confidence = 0.65  # Some conflicts
        else:
            conflict_confidence = 0.5
        
        # IMPROVED: Evidence strength with better assessment
        if evidence_map:
            evidence_counts = [len(evidences) for evidences in evidence_map.values()]
            avg_evidence_count = np.mean(evidence_counts)
            min_evidence_count = min(evidence_counts)
            
            # Penalty for claims with very little evidence
            if min_evidence_count < 2:
                evidence_confidence = 0.5
            else:
                evidence_confidence = min(avg_evidence_count / 4, 1.0)
        else:
            evidence_confidence = 0.3
        
        # IMPROVED: Better combination with conflict signal priority
        if num_conflicts >= 3:
            # When we have clear conflicts, prioritize that signal
            overall_confidence = (
                0.4 * distance_confidence +
                0.5 * conflict_confidence +
                0.1 * evidence_confidence
            )
        else:
            # When conflicts are unclear, balance all signals
            overall_confidence = (
                0.5 * distance_confidence +
                0.3 * conflict_confidence +
                0.2 * evidence_confidence
            )
        
        return min(max(overall_confidence, 0.35), 0.95)  # Bounded confidence
    
    def _generate_rationale(
        self,
        prediction: int,
        inconsistency_score: float,
        temporal_conflicts: List,
        causal_conflicts: List,
        evidence_map: Dict[str, List],
        claims: List
    ) -> str:
        """
        Generate human-readable rationale for decision.
        
        Includes:
        - Decision summary
        - Key conflicts (if any)
        - Evidence summary
        - Confidence explanation
        """
        from .utils import Evidence, format_rationale
        
        parts = []
        
        # Decision summary
        if prediction == 1:
            parts.append("CONSISTENT: Backstory is globally consistent with the novel.")
        else:
            parts.append("INCONSISTENT: Backstory contradicts the novel.")
        
        # Score
        parts.append(f"Inconsistency score: {inconsistency_score:.3f}")
        
        # Conflicts
        if temporal_conflicts:
            parts.append(f"Temporal conflicts: {len(temporal_conflicts)}")
            # Add top conflict
            if temporal_conflicts:
                top_conflict = temporal_conflicts[0]
                parts.append(f"  - {top_conflict.description[:100]}")
        
        if causal_conflicts:
            parts.append(f"Causal conflicts: {len(causal_conflicts)}")
            if causal_conflicts:
                top_conflict = causal_conflicts[0]
                parts.append(f"  - {top_conflict.description[:100]}")
        
        # Evidence summary
        if evidence_map:
            total_evidence = sum(len(ev) for ev in evidence_map.values())
            parts.append(f"Evidence pieces examined: {total_evidence}")
            
            # Add top evidence
            for claim_id, evidences in list(evidence_map.items())[:2]:
                if evidences:
                    top_ev = evidences[0]
                    excerpt = top_ev.text[:80] + "..." if len(top_ev.text) > 80 else top_ev.text
                    parts.append(f"  - \"{excerpt}\"")
        
        # Combine into rationale
        rationale = " | ".join(parts)
        
        # Truncate if too long
        if len(rationale) > 500:
            rationale = rationale[:497] + "..."
        
        return rationale
    
    def explain_decision(
        self,
        classification: Dict[str, Any]
    ) -> str:
        """
        Generate detailed explanation of classification decision.
        
        Args:
            classification: Output from classify()
        
        Returns:
            Detailed explanation string
        """
        explanation = []
        
        explanation.append("=" * 60)
        explanation.append("CLASSIFICATION EXPLANATION")
        explanation.append("=" * 60)
        
        # Prediction
        pred_label = "CONSISTENT" if classification['prediction'] == 1 else "INCONSISTENT"
        explanation.append(f"\nPrediction: {pred_label}")
        explanation.append(f"Confidence: {classification['confidence']:.2%}")
        
        # Score
        explanation.append(f"\nInconsistency Score: {classification['inconsistency_score']:.3f}")
        explanation.append(f"Threshold: {self.threshold:.3f}")
        
        # Conflicts
        explanation.append(f"\nTotal Conflicts: {classification['num_conflicts']}")
        
        # Rationale
        explanation.append(f"\nRationale:")
        explanation.append(classification['rationale'])
        
        explanation.append("\n" + "=" * 60)
        
        return "\n".join(explanation)
    
    def get_decision_factors(
        self,
        inconsistency_score: float,
        temporal_conflicts: List,
        causal_conflicts: List
    ) -> Dict[str, Any]:
        """
        Get factors contributing to decision.
        
        Returns:
            Dict with factor contributions
        """
        # Calculate contribution of each factor
        score_contribution = abs(inconsistency_score - self.threshold)
        
        temporal_contribution = (
            len(temporal_conflicts) * self.calibration_params['severity_weight']
        )
        
        causal_contribution = (
            len(causal_conflicts) * self.calibration_params['severity_weight']
        )
        
        return {
            'score_contribution': score_contribution,
            'temporal_contribution': temporal_contribution,
            'causal_contribution': causal_contribution,
            'dominant_factor': max(
                [
                    ('score', score_contribution),
                    ('temporal', temporal_contribution),
                    ('causal', causal_contribution)
                ],
                key=lambda x: x[1]
            )[0]
        }
