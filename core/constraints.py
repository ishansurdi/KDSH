"""
Constraint Graph Construction

Builds a graph of constraints from claims:
- Temporal constraints (ordering, duration)
- Causal constraints (cause-effect chains)
- Entity constraints (existence, state)

Research foundation:
- Sun et al. 2013 (Temporal constraint tracking)
- Feder et al. 2022 (Causal inference)
- Basu et al. 2022 (Neuro-symbolic reasoning)
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class Constraint:
    """A constraint between claims or entities"""
    constraint_id: str
    constraint_type: str  # temporal, causal, entity, logical
    source: str  # source claim/entity ID
    target: str  # target claim/entity ID
    relation: str  # before, after, causes, requires, etc.
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'constraint_id': self.constraint_id,
            'constraint_type': self.constraint_type,
            'source': self.source,
            'target': self.target,
            'relation': self.relation,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class ConstraintGraph:
    """
    Graph structure for tracking constraints between claims.
    
    Supports:
    - Adding constraints of various types
    - Querying constraint chains
    - Detecting conflicts
    - Path finding between claims
    """
    
    def __init__(self):
        self.constraints: Dict[str, Constraint] = {}
        self.forward_edges: Dict[str, List[str]] = defaultdict(list)  # source -> [targets]
        self.backward_edges: Dict[str, List[str]] = defaultdict(list)  # target -> [sources]
        self.constraint_counter = 0
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the graph"""
        self.constraints[constraint.constraint_id] = constraint
        self.forward_edges[constraint.source].append(constraint.target)
        self.backward_edges[constraint.target].append(constraint.source)
    
    def get_constraints_from(self, node_id: str) -> List[Constraint]:
        """Get all constraints originating from a node"""
        constraints = []
        for target in self.forward_edges.get(node_id, []):
            for c_id, constraint in self.constraints.items():
                if constraint.source == node_id and constraint.target == target:
                    constraints.append(constraint)
        return constraints
    
    def get_constraints_to(self, node_id: str) -> List[Constraint]:
        """Get all constraints targeting a node"""
        constraints = []
        for source in self.backward_edges.get(node_id, []):
            for c_id, constraint in self.constraints.items():
                if constraint.source == source and constraint.target == node_id:
                    constraints.append(constraint)
        return constraints
    
    def find_path(
        self, 
        start: str, 
        end: str, 
        max_depth: int = 5
    ) -> Optional[List[Constraint]]:
        """
        Find a path of constraints from start to end node.
        Uses BFS to find shortest path.
        
        Args:
            start: Starting node ID
            end: Target node ID
            max_depth: Maximum path length
        
        Returns:
            List of constraints forming the path, or None
        """
        if start == end:
            return []
        
        # BFS
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            # Check neighbors
            for neighbor in self.forward_edges.get(current, []):
                if neighbor == end:
                    # Found path
                    final_constraints = []
                    for constraint in self.constraints.values():
                        if constraint.source == current and constraint.target == neighbor:
                            final_constraints.append(constraint)
                            break
                    return path + final_constraints
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Find constraint
                    for constraint in self.constraints.values():
                        if constraint.source == current and constraint.target == neighbor:
                            queue.append((neighbor, path + [constraint]))
                            break
        
        return None
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the constraint graph"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.forward_edges.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
            
            rec_stack.remove(node)
        
        for node in self.forward_edges.keys():
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def get_temporal_ordering(self) -> List[str]:
        """
        Get topological ordering of temporal constraints.
        Returns nodes in temporal order if possible.
        """
        # Filter temporal constraints
        temporal_graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for constraint in self.constraints.values():
            if constraint.constraint_type == 'temporal':
                temporal_graph[constraint.source].append(constraint.target)
                in_degree[constraint.target] += 1
                all_nodes.add(constraint.source)
                all_nodes.add(constraint.target)
        
        # Kahn's algorithm for topological sort
        queue = [node for node in all_nodes if in_degree[node] == 0]
        ordering = []
        
        while queue:
            node = queue.pop(0)
            ordering.append(node)
            
            for neighbor in temporal_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return ordering
    
    def get_causal_chains(self) -> List[List[Constraint]]:
        """Get all causal chains in the graph"""
        chains = []
        
        # Find all causal constraints
        causal_constraints = [
            c for c in self.constraints.values()
            if c.constraint_type == 'causal'
        ]
        
        # Build chains
        for constraint in causal_constraints:
            chain = [constraint]
            
            # Follow forward
            current = constraint.target
            while current in self.forward_edges:
                found = False
                for c in self.constraints.values():
                    if c.source == current and c.constraint_type == 'causal':
                        chain.append(c)
                        current = c.target
                        found = True
                        break
                if not found:
                    break
            
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def export_graph(self, filepath: str):
        """Export graph to JSON"""
        graph_data = {
            'constraints': [c.to_dict() for c in self.constraints.values()],
            'statistics': {
                'total_constraints': len(self.constraints),
                'temporal_constraints': len([c for c in self.constraints.values() if c.constraint_type == 'temporal']),
                'causal_constraints': len([c for c in self.constraints.values() if c.constraint_type == 'causal']),
                'entity_constraints': len([c for c in self.constraints.values() if c.constraint_type == 'entity']),
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"✓ Exported constraint graph to {filepath}")


class ConstraintBuilder:
    """
    Build constraint graph from claims.
    
    Analyzes claims to extract:
    - Temporal constraints from time markers
    - Causal constraints from causal markers
    - Entity constraints from entity relationships
    """
    
    def __init__(self):
        self.constraint_counter = 0
    
    def build_graph(self, claims: List) -> ConstraintGraph:
        """
        Build constraint graph from list of claims.
        
        Args:
            claims: List of Claim objects
        
        Returns:
            ConstraintGraph with all extracted constraints
        """
        graph = ConstraintGraph()
        
        # Build temporal constraints
        temporal_constraints = self._extract_temporal_constraints(claims)
        for constraint in temporal_constraints:
            graph.add_constraint(constraint)
        
        # Build causal constraints
        causal_constraints = self._extract_causal_constraints(claims)
        for constraint in causal_constraints:
            graph.add_constraint(constraint)
        
        # Build entity constraints
        entity_constraints = self._extract_entity_constraints(claims)
        for constraint in entity_constraints:
            graph.add_constraint(constraint)
        
        print(f"✓ Built constraint graph: {len(graph.constraints)} constraints")
        
        return graph
    
    def _extract_temporal_constraints(self, claims: List) -> List[Constraint]:
        """Extract temporal ordering constraints AGGRESSIVELY"""
        constraints = []
        
        # AGGRESSIVE: Extract ALL temporal patterns
        temporal_patterns = {
            'before': ['before', 'prior to', 'earlier than', 'preceded', 'until', 'up to'],
            'after': ['after', 'following', 'later than', 'subsequent to', 'since', 'from'],
            'during': ['during', 'while', 'as', 'when', 'at the time'],
            'age': ['age', 'years old', 'year old', 'aged'],
            'date': ['in 18', 'in 19', 'in 20'],  # Date patterns
            'life_stage': ['born', 'childhood', 'youth', 'married', 'died', 'graduated', 'retired']
        }
        
        # AGGRESSIVE: Create constraints for EVERY claim pair with shared entities
        for i, claim1 in enumerate(claims):
            text1 = claim1.text.lower()
            
            # Extract temporal info from this claim
            has_temporal1 = any(kw in text1 for patterns in temporal_patterns.values() for kw in patterns)
            
            for j, claim2 in enumerate(claims):
                if i >= j:  # Skip self and already processed pairs
                    continue
                    
                text2 = claim2.text.lower()
                has_temporal2 = any(kw in text2 for patterns in temporal_patterns.values() for kw in patterns)
                
                # If shared entities OR both have temporal markers
                if set(claim1.entities) & set(claim2.entities) or (has_temporal1 and has_temporal2):
                    
                    # Check for explicit ordering keywords
                    for relation_type, keywords in temporal_patterns.items():
                        if relation_type in ['before', 'after', 'during']:
                            if any(kw in text1 for kw in keywords):
                                constraints.append(Constraint(
                                    constraint_id=self._next_id(),
                                    constraint_type='temporal',
                                    source=claim1.claim_id,
                                    target=claim2.claim_id,
                                    relation=relation_type,
                                    confidence=0.85
                                ))
                            if any(kw in text2 for kw in keywords):
                                constraints.append(Constraint(
                                    constraint_id=self._next_id(),
                                    constraint_type='temporal',
                                    source=claim2.claim_id,
                                    target=claim1.claim_id,
                                    relation=relation_type,
                                    confidence=0.85
                                ))
                    
                    # Extract age constraints
                    import re
                    age_pattern = r'(\d+)\s*(?:years?\s*old|aged)'
                    age1 = re.search(age_pattern, text1)
                    age2 = re.search(age_pattern, text2)
                    if age1 and age2:
                        # Create age comparison constraint
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='temporal',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation=f'age_constraint_{age1.group(1)}_{age2.group(1)}',
                            confidence=0.95
                        ))
                    
                    # Life stage ordering constraints (birth before marriage before death)
                    stages = ['born', 'childhood', 'youth', 'student', 'graduated', 'married', 'career', 'retired', 'died']
                    stage1_idx = -1
                    stage2_idx = -1
                    for idx, stage in enumerate(stages):
                        if stage in text1:
                            stage1_idx = idx
                        if stage in text2:
                            stage2_idx = idx
                    
                    if stage1_idx >= 0 and stage2_idx >= 0 and stage1_idx != stage2_idx:
                        if stage1_idx < stage2_idx:
                            constraints.append(Constraint(
                                constraint_id=self._next_id(),
                                constraint_type='temporal',
                                source=claim1.claim_id,
                                target=claim2.claim_id,
                                relation='life_stage_before',
                                confidence=0.95
                            ))
                        else:
                            constraints.append(Constraint(
                                constraint_id=self._next_id(),
                                constraint_type='temporal',
                                source=claim2.claim_id,
                                target=claim1.claim_id,
                                relation='life_stage_before',
                                confidence=0.95
                            ))
                    
                    # Default weak temporal ordering for sequential claims with shared entities
                    if set(claim1.entities) & set(claim2.entities):
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='temporal',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='sequential',
                            confidence=0.5
                        ))
        
        return constraints
    
    def _extract_causal_constraints(self, claims: List) -> List[Constraint]:
        """Extract causal constraints AGGRESSIVELY"""
        constraints = []
        
        # AGGRESSIVE: Expanded causal markers
        causal_markers = {
            'causes': ['caused', 'led to', 'resulted in', 'brought about', 'triggered', 'produced', 'created'],
            'because': ['because', 'due to', 'owing to', 'since', 'as a result of', 'thanks to', 'stems from'],
            'enables': ['enabled', 'allowed', 'made possible', 'facilitated', 'permitted', 'let'],
            'prevents': ['prevented', 'stopped', 'hindered', 'blocked', 'forbidden', 'unable to'],
            'requires': ['requires', 'needs', 'demands', 'necessitates', 'must', 'depends on'],
            'consequence': ['therefore', 'thus', 'consequently', 'as a consequence', 'so']
        }
        
        # AGGRESSIVE: Check ALL claims, not just causal type
        for i, claim1 in enumerate(claims):
            text1 = claim1.text.lower()
            
            # Check for causal markers in this claim
            for relation, markers in causal_markers.items():
                if any(marker in text1 for marker in markers):
                    # Link to ALL other claims with shared entities
                    for j, claim2 in enumerate(claims):
                        if i != j and (set(claim1.entities) & set(claim2.entities)):
                            confidence = 0.9 if relation in ['requires', 'prevents'] else 0.75
                            constraints.append(Constraint(
                                constraint_id=self._next_id(),
                                constraint_type='causal',
                                source=claim1.claim_id,
                                target=claim2.claim_id,
                                relation=relation,
                                confidence=confidence
                            ))
        
        # AGGRESSIVE: Prerequisite detection (actions need preconditions)
        prerequisite_actions = {
            'graduated': ['attended', 'studied', 'enrolled'],
            'married': ['met', 'courted', 'engaged'],
            'died': ['lived', 'born', 'existed'],
            'won': ['competed', 'participated', 'entered'],
            'published': ['wrote', 'composed', 'created'],
            'retired': ['worked', 'employed', 'career']
        }
        
        for i, claim1 in enumerate(claims):
            text1 = claim1.text.lower()
            for action, prerequisites in prerequisite_actions.items():
                if action in text1:
                    # This action requires prerequisites
                    for j, claim2 in enumerate(claims):
                        if i != j:
                            text2 = claim2.text.lower()
                            if any(prereq in text2 for prereq in prerequisites):
                                if set(claim1.entities) & set(claim2.entities):
                                    constraints.append(Constraint(
                                        constraint_id=self._next_id(),
                                        constraint_type='causal',
                                        source=claim2.claim_id,
                                        target=claim1.claim_id,
                                        relation='prerequisite',
                                        confidence=0.95
                                    ))
        
        # AGGRESSIVE: Event chains (consecutive events with shared entities likely causal)
        for i, claim1 in enumerate(claims):
            for j in range(i+1, min(i+5, len(claims))):  # Check next 4 claims
                claim2 = claims[j]
                if set(claim1.entities) & set(claim2.entities):
                    constraints.append(Constraint(
                        constraint_id=self._next_id(),
                        constraint_type='causal',
                        source=claim1.claim_id,
                        target=claim2.claim_id,
                        relation='event_chain',
                        confidence=0.6
                    ))
        
        return constraints
    
    def _extract_entity_constraints(self, claims: List) -> List[Constraint]:
        """Extract entity-based constraints AGGRESSIVELY"""
        constraints = []
        
        # Group claims by entity
        entity_claims = defaultdict(list)
        for claim in claims:
            for entity in claim.entities:
                entity_claims[entity].append(claim)
        
        # AGGRESSIVE: Create constraints for EVERY entity claim pair
        for entity, entity_claim_list in entity_claims.items():
            for i, claim1 in enumerate(entity_claim_list):
                for j, claim2 in enumerate(entity_claim_list):
                    if i >= j:
                        continue
                    
                    text1 = claim1.text.lower()
                    text2 = claim2.text.lower()
                    
                    # State/attribute constraints (high confidence)
                    state_words = ['was', 'is', 'had', 'has', 'became', 'turned', 'remained']
                    if any(w in text1 for w in state_words) and any(w in text2 for w in state_words):
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='state_consistency',
                            confidence=0.9
                        ))
                    
                    # Ability/capacity constraints
                    ability_words = ['can', 'could', 'able to', 'unable', 'capable', 'incapable']
                    if any(w in text1 for w in ability_words) or any(w in text2 for w in ability_words):
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='ability_constraint',
                            confidence=0.85
                        ))
                    
                    # Physical property constraints (blind can't see, etc)
                    physical_props = ['blind', 'deaf', 'mute', 'paralyzed', 'dead', 'alive', 'young', 'old']
                    if any(prop in text1 for prop in physical_props) or any(prop in text2 for prop in physical_props):
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='physical_property',
                            confidence=0.95
                        ))
                    
                    # Role/occupation constraints
                    role_words = ['worked as', 'was a', 'served as', 'became a', 'profession', 'job', 'career']
                    if any(r in text1 for r in role_words) or any(r in text2 for r in role_words):
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='role_constraint',
                            confidence=0.85
                        ))
                    
                    # Location constraints (can't be in two places)
                    location_words = ['in', 'at', 'lived', 'moved to', 'traveled', 'visited']
                    if any(loc in text1 for loc in location_words) and any(loc in text2 for loc in location_words):
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='location_constraint',
                            confidence=0.8
                        ))
                    
                    # Action sequence constraints
                    if claim1.claim_type == 'event' and claim2.claim_type == 'event':
                        constraints.append(Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='action_sequence',
                            confidence=0.7
                        ))
                    
                    # Default entity consistency constraint
                    constraints.append(Constraint(
                        constraint_id=self._next_id(),
                        constraint_type='entity',
                        source=claim1.claim_id,
                        target=claim2.claim_id,
                        relation='same_entity',
                        confidence=0.8
                    ))
        
        # IMPROVED: Add mutual exclusion constraints for contradictory states
        for entity, entity_claim_list in entity_claims.items():
            for i, claim1 in enumerate(entity_claim_list):
                for claim2 in entity_claim_list[i+1:]:
                    # Check for contradictory states (e.g., alive vs dead, married vs single)
                    if self._are_contradictory_states(claim1.text, claim2.text):
                        constraint = Constraint(
                            constraint_id=self._next_id(),
                            constraint_type='entity',
                            source=claim1.claim_id,
                            target=claim2.claim_id,
                            relation='mutually_exclusive',
                            confidence=0.9,
                            metadata={'entity': entity, 'conflict_type': 'state_contradiction'}
                        )
                        constraints.append(constraint)
        
        return constraints
    
    def _are_contradictory_states(self, text1: str, text2: str) -> bool:
        """Check if two texts describe contradictory states"""
        contradictions = [
            (['alive', 'living'], ['dead', 'died', 'deceased']),
            (['married', 'wed'], ['single', 'unmarried', 'bachelor']),
            (['rich', 'wealthy'], ['poor', 'impoverished']),
            (['friend', 'ally'], ['enemy', 'foe', 'rival']),
            (['innocent'], ['guilty', 'criminal']),
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for positive_terms, negative_terms in contradictions:
            has_positive_in_1 = any(term in text1_lower for term in positive_terms)
            has_negative_in_2 = any(term in text2_lower for term in negative_terms)
            has_positive_in_2 = any(term in text2_lower for term in positive_terms)
            has_negative_in_1 = any(term in text1_lower for term in negative_terms)
            
            if (has_positive_in_1 and has_negative_in_2) or (has_positive_in_2 and has_negative_in_1):
                return True
        
        return False
    
    def _next_id(self) -> str:
        """Generate next constraint ID"""
        self.constraint_counter += 1
        return f"constraint_{self.constraint_counter}"
