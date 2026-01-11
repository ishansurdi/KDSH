"""
Machine Learning Classifier for Narrative Consistency
Fast ensemble approach combining multiple ML algorithms
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class MLConsistencyClassifier:
    """
    Machine Learning classifier using ensemble of algorithms:
    - Random Forest
    - Gradient Boosting
    - Neural Network (MLP)
    - Logistic Regression
    
    Trained on features extracted from scoring components.
    """
    
    def __init__(self):
        """Initialize ensemble of ML models"""
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            'lr': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def extract_features(
        self,
        inconsistency_score: float,
        temporal_conflicts: List,
        causal_conflicts: List,
        evidence_map: Dict[str, List],
        claims: List,
        component_scores: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Extract feature vector for ML models.
        
        Features:
        1. Inconsistency score
        2. Number of temporal conflicts
        3. Number of causal conflicts
        4. Max temporal severity
        5. Max causal severity
        6. Avg temporal severity
        7. Avg causal severity
        8. Number of claims
        9. Evidence coverage ratio
        10-15. Component scores (temporal, causal, entity, semantic, evidence, reasoning)
        """
        features = []
        
        # Basic scores
        features.append(inconsistency_score)
        
        # Conflict counts
        num_temporal = len(temporal_conflicts)
        num_causal = len(causal_conflicts)
        features.append(num_temporal)
        features.append(num_causal)
        features.append(num_temporal + num_causal)  # Total conflicts
        
        # Temporal severity statistics
        if temporal_conflicts:
            temporal_severities = [c.severity for c in temporal_conflicts]
            features.append(max(temporal_severities))
            features.append(np.mean(temporal_severities))
            features.append(np.std(temporal_severities) if len(temporal_severities) > 1 else 0)
        else:
            features.extend([0, 0, 0])
        
        # Causal severity statistics
        if causal_conflicts:
            causal_severities = [c.severity for c in causal_conflicts]
            features.append(max(causal_severities))
            features.append(np.mean(causal_severities))
            features.append(np.std(causal_severities) if len(causal_severities) > 1 else 0)
        else:
            features.extend([0, 0, 0])
        
        # Claims and evidence
        num_claims = len(claims)
        features.append(num_claims)
        
        # Evidence coverage
        if evidence_map and num_claims > 0:
            claims_with_evidence = sum(1 for claim_id in evidence_map if evidence_map[claim_id])
            evidence_coverage = claims_with_evidence / num_claims
        else:
            evidence_coverage = 0
        features.append(evidence_coverage)
        
        # Component scores (if available)
        if component_scores:
            features.append(component_scores.get('temporal', 0))
            features.append(component_scores.get('causal', 0))
            features.append(component_scores.get('entity', 0))
            features.append(component_scores.get('semantic', 0))
            features.append(component_scores.get('evidence', 0))
            features.append(component_scores.get('reasoning', 0))
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Interaction features
        features.append(inconsistency_score * (num_temporal + num_causal))  # Score × conflicts
        features.append(inconsistency_score * num_claims)  # Score × claims
        
        return np.array(features)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train all models in ensemble.
        
        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Labels (0=inconsistent, 1=consistent)
        
        Returns:
            Dict with cross-validation scores for each model
        """
        print(f"\n[ML Classifier] Training on {len(X_train)} examples with {X_train.shape[1]} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train each model and evaluate
        cv_scores = {}
        for name, model in self.models.items():
            print(f"  Training {name}...", end=" ")
            model.fit(X_scaled, y_train)
            
            # Cross-validation score
            scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring='accuracy')
            cv_scores[name] = scores.mean()
            print(f"CV Accuracy: {scores.mean():.3f} (±{scores.std():.3f})")
        
        self.is_trained = True
        return cv_scores
    
    def predict(
        self,
        inconsistency_score: float,
        temporal_conflicts: List,
        causal_conflicts: List,
        evidence_map: Dict[str, List],
        claims: List,
        component_scores: Dict[str, float] = None,
        use_voting: bool = True
    ) -> Dict[str, Any]:
        """
        Predict consistency using trained models.
        
        Args:
            inconsistency_score: Overall inconsistency score
            temporal_conflicts: List of temporal conflicts
            causal_conflicts: List of causal conflicts
            evidence_map: Evidence for claims
            claims: List of claims
            component_scores: Optional dict of component scores
            use_voting: If True, use majority voting; else use average probabilities
        
        Returns:
            Dict with prediction, confidence, and model details
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        # Extract features
        features = self.extract_features(
            inconsistency_score,
            temporal_conflicts,
            causal_conflicts,
            evidence_map,
            claims,
            component_scores
        )
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[name] = pred
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                probabilities[name] = proba
        
        # Ensemble prediction
        if use_voting:
            # Majority voting
            votes = list(predictions.values())
            final_prediction = int(np.round(np.mean(votes)))
            
            # Confidence based on agreement
            agreement = votes.count(final_prediction) / len(votes)
            confidence = agreement
        else:
            # Average probabilities
            if probabilities:
                avg_proba = np.mean([p for p in probabilities.values()], axis=0)
                final_prediction = int(np.argmax(avg_proba))
                confidence = float(avg_proba[final_prediction])
            else:
                # Fallback to voting
                votes = list(predictions.values())
                final_prediction = int(np.round(np.mean(votes)))
                confidence = votes.count(final_prediction) / len(votes)
        
        # Override for extreme conflict counts
        total_conflicts = len(temporal_conflicts) + len(causal_conflicts)
        if total_conflicts >= 10:
            final_prediction = 0  # Inconsistent
            confidence = 0.95
            rationale = f"OVERRIDE: {total_conflicts} conflicts detected"
        else:
            rationale = f"ML Ensemble: {list(predictions.values()).count(final_prediction)}/{len(predictions)} models agree"
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'rationale': rationale,
            'model_predictions': predictions,
            'inconsistency_score': inconsistency_score,
            'num_conflicts': total_conflicts
        }
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            return {}
        
        rf_model = self.models['rf']
        importances = rf_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
