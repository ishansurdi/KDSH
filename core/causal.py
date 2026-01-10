"""
Causal Reasoning Engine

Checks causal consistency between claims and narrative evidence.
Detects:
- Missing causal links
- Impossible causal chains
- Contradictory cause-effect relationships

Research foundation:
- Feder et al. 2022 (Causal inference in NLP)
- Basu et al. 2022 (Neuro-symbolic reasoning)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CausalLink:
    """Represents a cause-effect relationship"""
    cause_id: str
    effect_id: str
    cause_text: str
    effect_text: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cause_id': self.cause_id,
            'effect_id': self.effect_id,
            'cause_text': self.cause_text,
            'effect_text': self.effect_text,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


@dataclass
class CausalConflict:
    """Represents a detected causal inconsistency"""
    conflict_type: str  # missing_link, impossible_chain, contradiction
    claim_id: str
    description: str
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    severity: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'conflict_type': self.conflict_type,
            'claim_id': self.claim_id,
            'description': self.description,
            'evidence_for': self.evidence_for,
            'evidence_against': self.evidence_against,
            'severity': self.severity
        }


class CausalReasoningEngine:
    """
    Engine for causal consistency checking.
    
    Performs:
    1. Causal link extraction from evidence
    2. Causal chain validation
    3. Contradiction detection
    4. Counterfactual reasoning
    """
    
    def __init__(self, memory, constraint_graph):
        """
        Initialize engine.
        
        Args:
            memory: HierarchicalNarrativeMemory instance
            constraint_graph: ConstraintGraph instance
        """
        self.memory = memory
        self.constraint_graph = constraint_graph
        self.causal_links: List[CausalLink] = []
    
    def extract_causal_links(
        self,
        evidence_map: Dict[str, List]
    ) -> List[CausalLink]:
        """
        Extract causal links from evidence.
        
        Args:
            evidence_map: Dict mapping claim_id to evidence results
        
        Returns:
            List of CausalLink objects
        """
        from .utils import extract_causal_markers
        
        links = []
        
        for claim_id, evidence_list in evidence_map.items():
            for evidence in evidence_list:
                # Extract causal markers from evidence text
                markers = extract_causal_markers(evidence.text)
                
                if markers:
                    # Try to identify cause and effect
                    link = self._parse_causal_structure(
                        evidence.text,
                        markers,
                        claim_id
                    )
                    if link:
                        links.append(link)
        
        self.causal_links = links
        print(f"✓ Extracted {len(links)} causal links from evidence")
        
        return links
    
    def check_causal_consistency(
        self,
        claims: List,
        evidence_map: Dict[str, List]
    ) -> List[CausalConflict]:
        """
        Check causal consistency of claims against evidence.
        
        Args:
            claims: List of Claim objects
            evidence_map: Evidence for each claim
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Extract causal links if not done
        if not self.causal_links:
            self.extract_causal_links(evidence_map)
        
        # Check each causal claim
        causal_claims = [c for c in claims if c.claim_type == 'causal']
        
        for claim in causal_claims:
            # Check for missing causal links
            missing_link = self._check_missing_links(claim, evidence_map)
            if missing_link:
                conflicts.append(missing_link)
            
            # Check for impossible chains
            impossible = self._check_impossible_chain(claim, evidence_map)
            if impossible:
                conflicts.append(impossible)
            
            # Check for contradictions
            contradiction = self._check_causal_contradiction(claim, evidence_map)
            if contradiction:
                conflicts.append(contradiction)
        
        print(f"⚠ Found {len(conflicts)} causal conflicts")
        
        return conflicts
    
    def validate_causal_chain(
        self,
        chain: List[str],
        evidence_map: Dict[str, List]
    ) -> Tuple[bool, str]:
        """
        Validate a causal chain of claims.
        
        Args:
            chain: List of claim IDs forming a chain
            evidence_map: Evidence for each claim
        
        Returns:
            (is_valid, explanation)
        """
        if len(chain) < 2:
            return True, "Chain too short to validate"
        
        # Check each link in chain
        for i in range(len(chain) - 1):
            cause_claim_id = chain[i]
            effect_claim_id = chain[i + 1]
            
            # Check if evidence supports this link
            cause_evidence = evidence_map.get(cause_claim_id, [])
            effect_evidence = evidence_map.get(effect_claim_id, [])
            
            # Look for causal markers connecting them
            has_link = False
            for c_ev in cause_evidence:
                for e_ev in effect_evidence:
                    if self._check_causal_connection(c_ev.text, e_ev.text):
                        has_link = True
                        break
                if has_link:
                    break
            
            if not has_link:
                return False, f"Missing causal link between {cause_claim_id} and {effect_claim_id}"
        
        return True, "Causal chain is valid"
    
    def counterfactual_check(
        self,
        claim: Any,
        evidence_map: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Perform counterfactual reasoning.
        Check: "If claim is true, what must also be true?"
        
        Args:
            claim: Claim to check
            evidence_map: Evidence map
        
        Returns:
            Dict with counterfactual analysis
        """
        implications = []
        conflicts = []
        
        # Get causal constraints involving this claim
        constraints = self.constraint_graph.get_constraints_from(claim.claim_id)
        
        for constraint in constraints:
            if constraint.constraint_type == 'causal':
                # If claim is cause, effect must be true
                target_claim_id = constraint.target
                
                # Check if evidence supports the effect
                target_evidence = evidence_map.get(target_claim_id, [])
                
                if not target_evidence or all(ev.score < 0.3 for ev in target_evidence):
                    conflicts.append({
                        'implied': target_claim_id,
                        'reason': 'Causal implication not supported by evidence',
                        'constraint': constraint.to_dict()
                    })
                else:
                    implications.append({
                        'implied': target_claim_id,
                        'support': 'Evidence supports causal implication',
                        'constraint': constraint.to_dict()
                    })
        
        return {
            'claim_id': claim.claim_id,
            'implications': implications,
            'conflicts': conflicts,
            'is_consistent': len(conflicts) == 0
        }
    
    def _parse_causal_structure(
        self,
        text: str,
        markers: List[Dict],
        claim_id: str
    ) -> Optional[CausalLink]:
        """Parse causal structure from text with markers"""
        # Simple heuristic: split on causal marker
        for marker in markers:
            marker_text = marker['text'].lower()
            
            if marker_text in ['because', 'since', 'due to']:
                # Cause comes after marker
                parts = text.lower().split(marker_text)
                if len(parts) >= 2:
                    effect_text = parts[0].strip()
                    cause_text = parts[1].strip()
                    
                    return CausalLink(
                        cause_id=f"{claim_id}_cause",
                        effect_id=f"{claim_id}_effect",
                        cause_text=cause_text[:200],
                        effect_text=effect_text[:200],
                        confidence=0.6,
                        evidence=[text]
                    )
            
            elif marker_text in ['therefore', 'thus', 'consequently']:
                # Effect comes after marker
                parts = text.lower().split(marker_text)
                if len(parts) >= 2:
                    cause_text = parts[0].strip()
                    effect_text = parts[1].strip()
                    
                    return CausalLink(
                        cause_id=f"{claim_id}_cause",
                        effect_id=f"{claim_id}_effect",
                        cause_text=cause_text[:200],
                        effect_text=effect_text[:200],
                        confidence=0.6,
                        evidence=[text]
                    )
        
        return None
    
    def _check_missing_links(
        self,
        claim: Any,
        evidence_map: Dict[str, List]
    ) -> Optional[CausalConflict]:
        """Check if causal claim lacks supporting evidence"""
        evidence = evidence_map.get(claim.claim_id, [])
        
        # Check if any evidence contains causal markers
        has_causal_evidence = False
        for ev in evidence:
            from .utils import extract_causal_markers
            markers = extract_causal_markers(ev.text)
            if markers:
                has_causal_evidence = True
                break
        
        if not has_causal_evidence and len(evidence) < 3:
            return CausalConflict(
                conflict_type='missing_link',
                claim_id=claim.claim_id,
                description=f"Causal claim lacks causal evidence: {claim.text[:100]}",
                evidence_against=[],
                severity=0.6
            )
        
        return None
    
    def _check_impossible_chain(
        self,
        claim: Any,
        evidence_map: Dict[str, List]
    ) -> Optional[CausalConflict]:
        """Check if causal chain is logically impossible"""
        # Check for circular causation
        causal_constraints = [
            c for c in self.constraint_graph.get_constraints_from(claim.claim_id)
            if c.constraint_type == 'causal'
        ]
        
        for constraint in causal_constraints:
            # Check if target eventually points back to source
            path = self.constraint_graph.find_path(
                constraint.target,
                constraint.source,
                max_depth=5
            )
            
            if path:
                return CausalConflict(
                    conflict_type='impossible_chain',
                    claim_id=claim.claim_id,
                    description=f"Circular causation detected: {claim.text[:100]}",
                    severity=0.8
                )
        
        return None
    
    def _check_causal_contradiction(
        self,
        claim: Any,
        evidence_map: Dict[str, List]
    ) -> Optional[CausalConflict]:
        """Check if causal claim contradicts evidence"""
        evidence = evidence_map.get(claim.claim_id, [])
        
        # Look for contradicting causal links in evidence
        claim_text_lower = claim.text.lower()
        
        for ev in evidence:
            ev_text_lower = ev.text.lower()
            
            # Simple contradiction detection
            if 'not' in ev_text_lower and any(
                word in ev_text_lower 
                for word in ['cause', 'led to', 'result']
            ):
                # Evidence mentions causal negation
                return CausalConflict(
                    conflict_type='contradiction',
                    claim_id=claim.claim_id,
                    description=f"Evidence contradicts causal claim",
                    evidence_for=[],
                    evidence_against=[ev.text[:200]],
                    severity=0.7
                )
        
        return None
    
    def _check_causal_connection(self, text1: str, text2: str) -> bool:
        """Check if two texts suggest causal connection"""
        from .utils import extract_causal_markers
        
        combined = text1 + " " + text2
        markers = extract_causal_markers(combined)
        
        return len(markers) > 0
