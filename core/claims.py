"""
Claim Extraction from Backstories

Extracts structured claims from hypothetical backstories.
Each claim represents a testable assertion about the narrative.

Research foundation:
- Zhang & Long 2024 (Narrative coherence)
- Basu et al. 2022 (Neuro-symbolic reasoning)
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import re


@dataclass
class Claim:
    """A single claim extracted from a backstory"""
    claim_id: str
    text: str
    claim_type: str  # entity, event, temporal, causal, relationship
    entities: List[str] = field(default_factory=list)
    temporal_info: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Other claim IDs this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'claim_id': self.claim_id,
            'text': self.text,
            'claim_type': self.claim_type,
            'entities': self.entities,
            'temporal_info': self.temporal_info,
            'dependencies': self.dependencies
        }


class ClaimExtractor:
    """
    Extract structured claims from backstory text.
    
    Uses rule-based patterns and NLP to identify:
    - Entity claims (character existence, attributes)
    - Event claims (actions, occurrences)
    - Temporal claims (time ordering, duration)
    - Causal claims (cause-effect relationships)
    - Relationship claims (character interactions)
    """
    
    def __init__(self):
        self.claim_counter = 0
    
    def extract_claims(self, backstory: str) -> List[Claim]:
        """
        Extract all claims from a backstory.
        
        Args:
            backstory: The hypothetical backstory text
        
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', backstory)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            # Extract different types of claims
            claims.extend(self._extract_entity_claims(sentence))
            claims.extend(self._extract_event_claims(sentence))
            claims.extend(self._extract_temporal_claims(sentence))
            claims.extend(self._extract_causal_claims(sentence))
            claims.extend(self._extract_relationship_claims(sentence))
        
        return claims
    
    def _extract_entity_claims(self, text: str) -> List[Claim]:
        """Extract claims about entity existence and attributes"""
        from .utils import extract_entities
        
        claims = []
        entities = extract_entities(text)
        
        # Existence claims
        for entity in entities:
            claim = Claim(
                claim_id=self._next_id(),
                text=f"Entity '{entity['text']}' exists",
                claim_type='entity',
                entities=[entity['text']]
            )
            claims.append(claim)
        
        # Attribute claims (simple patterns)
        patterns = [
            (r'(\w+) (?:was|is) (?:a|an) (\w+)', 'attribute'),
            (r'(\w+) (?:had|has) (?:a|an|the) (\w+)', 'possession'),
        ]
        
        for pattern, attr_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity, attribute = match.group(1), match.group(2)
                claim = Claim(
                    claim_id=self._next_id(),
                    text=f"{entity} has attribute: {attribute}",
                    claim_type='entity',
                    entities=[entity]
                )
                claims.append(claim)
        
        return claims
    
    def _extract_event_claims(self, text: str) -> List[Claim]:
        """Extract claims about events and actions"""
        claims = []
        
        # Action patterns
        action_patterns = [
            r'(\w+) (?:went|traveled|moved) to (\w+)',
            r'(\w+) (?:met|saw|encountered) (\w+)',
            r'(\w+) (?:killed|defeated|destroyed) (\w+)',
            r'(\w+) (?:married|loved|befriended) (\w+)',
            r'(\w+) (?:discovered|found|learned) (?:about |that )?(.+)',
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim = Claim(
                    claim_id=self._next_id(),
                    text=match.group(0),
                    claim_type='event',
                    entities=[match.group(1)]
                )
                if len(match.groups()) > 1:
                    claim.entities.append(match.group(2))
                claims.append(claim)
        
        return claims
    
    def _extract_temporal_claims(self, text: str) -> List[Claim]:
        """Extract claims about temporal ordering and duration"""
        from .utils import extract_temporal_markers
        
        claims = []
        markers = extract_temporal_markers(text)
        
        for marker in markers:
            claim = Claim(
                claim_id=self._next_id(),
                text=f"Temporal: {marker['text']}",
                claim_type='temporal',
                temporal_info=marker['text']
            )
            claims.append(claim)
        
        # Before/after patterns
        before_after_pattern = r'(.+) (?:before|after|during) (.+)'
        matches = re.finditer(before_after_pattern, text, re.IGNORECASE)
        for match in matches:
            claim = Claim(
                claim_id=self._next_id(),
                text=match.group(0),
                claim_type='temporal',
                temporal_info=match.group(0)
            )
            claims.append(claim)
        
        return claims
    
    def _extract_causal_claims(self, text: str) -> List[Claim]:
        """Extract claims about causal relationships"""
        from .utils import extract_causal_markers
        
        claims = []
        markers = extract_causal_markers(text)
        
        for marker in markers:
            # Find surrounding context
            start = max(0, marker['start'] - 50)
            end = min(len(text), marker['end'] + 50)
            context = text[start:end]
            
            claim = Claim(
                claim_id=self._next_id(),
                text=f"Causal: {context}",
                claim_type='causal'
            )
            claims.append(claim)
        
        return claims
    
    def _extract_relationship_claims(self, text: str) -> List[Claim]:
        """Extract claims about character relationships"""
        claims = []
        
        relationship_patterns = [
            (r'(\w+) (?:is|was) (?:the )?(son|daughter|father|mother|brother|sister) of (\w+)', 'family'),
            (r'(\w+) (?:is|was) (?:the )?(friend|enemy|rival|ally) of (\w+)', 'social'),
            (r'(\w+) (?:worked for|served|betrayed) (\w+)', 'professional'),
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity1 = match.group(1)
                entity2 = match.group(3) if len(match.groups()) > 2 else match.group(2)
                
                claim = Claim(
                    claim_id=self._next_id(),
                    text=match.group(0),
                    claim_type='relationship',
                    entities=[entity1, entity2]
                )
                claims.append(claim)
        
        return claims
    
    def extract_claims_aggressive(self, backstory: str) -> List[Claim]:
        """Enhanced claim extraction that catches MORE patterns."""
        import re
        
        claims = []
        
        # Split on more punctuation
        sentences = re.split(r'[.!?;]+', backstory)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            # ALWAYS create a basic claim for each sentence
            claim_id = self._next_id()
            
            # Determine type more aggressively
            text_lower = sentence.lower()
            
            if any(marker in text_lower for marker in ['before', 'after', 'during', 'when', 'since', 'until', 'while']):
                claim_type = 'temporal'
            elif any(marker in text_lower for marker in ['because', 'caused', 'led to', 'resulted', 'due to']):
                claim_type = 'causal'
            elif any(marker in text_lower for marker in ['was', 'is', 'had', 'has', 'became']):
                claim_type = 'entity'
            else:
                claim_type = 'event'
            
            # Extract entities
            from .utils import extract_entities
            entities = extract_entities(sentence)
            entity_names = [e['text'] for e in entities]
            
            # Create claim
            claim = Claim(
                claim_id=claim_id,
                text=sentence,
                claim_type=claim_type,
                entities=entity_names,
                temporal_info=sentence if claim_type == 'temporal' else None
            )
            claims.append(claim)
            
            # EXTRACT AGE/DATE CLAIMS EXPLICITLY
            age_pattern = r'(\d+)\s*(?:years?\s*old|aged|age)'
            age_match = re.search(age_pattern, text_lower)
            if age_match:
                age_claim = Claim(
                    claim_id=self._next_id(),
                    text=f"Age {age_match.group(1)} mentioned",
                    claim_type='temporal',
                    entities=entity_names,
                    temporal_info=age_match.group(0)
                )
                claims.append(age_claim)
            
            # Extract year claims
            year_pattern = r'\b(1[7-9]\d{2}|20[0-2]\d)\b'
            year_matches = re.findall(year_pattern, sentence)
            for year in year_matches:
                year_claim = Claim(
                    claim_id=self._next_id(),
                    text=f"Year {year} mentioned in context",
                    claim_type='temporal',
                    entities=entity_names,
                    temporal_info=f"year {year}"
                )
                claims.append(year_claim)
        
        return claims
    
    def _next_id(self) -> str:
        """Generate next claim ID"""
        self.claim_counter += 1
        return f"claim_{self.claim_counter}"


def analyze_claim_dependencies(claims: List[Claim]) -> Dict[str, List[str]]:
    """
    Analyze dependencies between claims.
    
    Returns:
        Dict mapping claim_id to list of dependent claim_ids
    """
    dependencies = {}
    
    for i, claim in enumerate(claims):
        deps = []
        
        # Check if other claims mention same entities
        for j, other_claim in enumerate(claims):
            if i == j:
                continue
            
            # If claims share entities, potential dependency
            shared_entities = set(claim.entities) & set(other_claim.entities)
            if shared_entities:
                deps.append(other_claim.claim_id)
        
        dependencies[claim.claim_id] = deps
    
    return dependencies
