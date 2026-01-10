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
        """Extract temporal ordering constraints"""
        constraints = []
        
        # Find temporal claims
        temporal_claims = [c for c in claims if c.claim_type == 'temporal']
        
        # IMPROVED: Extract before/after relations with better patterns
        temporal_keywords = {
            'before': ['before', 'prior to', 'earlier than', 'preceded'],
            'after': ['after', 'following', 'later than', 'subsequent to'],
            'during': ['during', 'while', 'as', 'when']
        }
        
        for claim in temporal_claims:
            text = claim.text.lower()
            
            # Check for temporal keywords
            for relation_type, keywords in temporal_keywords.items():
                if any(kw in text for kw in keywords):
                    # Find related claims
                    for other_claim in claims:
                        if other_claim.claim_id != claim.claim_id:
                            # Check if entities overlap or sequential events
                            if set(claim.entities) & set(other_claim.entities):
                                if relation_type == 'before':
                                    constraint = Constraint(
                                        constraint_id=self._next_id(),
                                        constraint_type='temporal',
                                        source=claim.claim_id,
                                        target=other_claim.claim_id,
                                        relation='before',
                                        confidence=0.8
                                    )
                                    constraints.append(constraint)
                                elif relation_type == 'after':
                                    constraint = Constraint(
                                        constraint_id=self._next_id(),
                                        constraint_type='temporal',
                                        source=other_claim.claim_id,
                                        target=claim.claim_id,
                                        relation='before',
                                        confidence=0.8
                                    )
                                    constraints.append(constraint)
                                elif relation_type == 'during':
                                    constraint = Constraint(
                                        constraint_id=self._next_id(),
                                        constraint_type='temporal',
                                        source=claim.claim_id,
                                        target=other_claim.claim_id,
                                        relation='concurrent',
                                        confidence=0.7
                                    )
                                    constraints.append(constraint)
        
        # IMPROVED: Infer temporal order from claim sequence (early events likely come first)
        for i, claim1 in enumerate(claims):
            if claim1.claim_type in ['event', 'temporal']:
                for claim2 in claims[i+1:]:
                    if claim2.claim_type in ['event', 'temporal']:
                        if set(claim1.entities) & set(claim2.entities):
                            # Weak temporal ordering assumption
                            constraint = Constraint(
                                constraint_id=self._next_id(),
                                constraint_type='temporal',
                                source=claim1.claim_id,
                                target=claim2.claim_id,
                                relation='possibly_before',
                                confidence=0.3
                            )
                            constraints.append(constraint)
        
        return constraints
    
    def _extract_causal_constraints(self, claims: List) -> List[Constraint]:
        """Extract causal constraints"""
        constraints = []
        
        # IMPROVED: More comprehensive causal markers
        causal_markers = {
            'causes': ['caused', 'led to', 'resulted in', 'brought about', 'triggered'],
            'because': ['because', 'due to', 'owing to', 'since', 'as a result of'],
            'enables': ['enabled', 'allowed', 'made possible', 'facilitated'],
            'prevents': ['prevented', 'stopped', 'hindered', 'blocked']
        }
        
        # Find causal claims
        causal_claims = [c for c in claims if c.claim_type == 'causal']
        
        for claim in causal_claims:
            text = claim.text.lower()
            
            # Check for each type of causal marker
            for relation, markers in causal_markers.items():
                if any(marker in text for marker in markers):
                    # Link to related claims
                    for other_claim in claims:
                        if other_claim.claim_id != claim.claim_id:
                            if set(claim.entities) & set(other_claim.entities):
                                confidence = 0.8 if relation in ['causes', 'because'] else 0.7
                                constraint = Constraint(
                                    constraint_id=self._next_id(),
                                    constraint_type='causal',
                                    source=claim.claim_id,
                                    target=other_claim.claim_id,
                                    relation=relation,
                                    confidence=confidence
                                )
                                constraints.append(constraint)
        
        # IMPROVED: Infer implicit causal relationships
        # Events involving the same entity often have causal relationships
        event_claims = [c for c in claims if c.claim_type == 'event']
        for i, claim1 in enumerate(event_claims):
            for claim2 in event_claims[i+1:i+4]:  # Check next few events
                if set(claim1.entities) & set(claim2.entities):
                    # Weak causal link (earlier event may influence later)
                    constraint = Constraint(
                        constraint_id=self._next_id(),
                        constraint_type='causal',
                        source=claim1.claim_id,
                        target=claim2.claim_id,
                        relation='may_influence',
                        confidence=0.4
                    )
                    constraints.append(constraint)
        
        return constraints
    
    def _extract_entity_constraints(self, claims: List) -> List[Constraint]:
        """Extract entity-based constraints"""
        constraints = []
        
        # Group claims by entity
        entity_claims = defaultdict(list)
        for claim in claims:
            for entity in claim.entities:
                entity_claims[entity].append(claim)
        
        # IMPROVED: Create different types of entity constraints
        for entity, entity_claim_list in entity_claims.items():
            for i, claim1 in enumerate(entity_claim_list):
                for claim2 in entity_claim_list[i+1:]:
                    # Determine constraint relation type based on claim types
                    relation = 'same_entity'
                    confidence = 0.9
                    
                    # State consistency constraints
                    if 'was' in claim1.text.lower() and 'was' in claim2.text.lower():
                        relation = 'state_consistency'
                        confidence = 0.85
                    
                    # Existence constraints
                    elif 'exist' in claim1.text.lower() or 'exist' in claim2.text.lower():
                        relation = 'existence_requirement'
                        confidence = 0.95
                    
                    # Action constraints
                    elif claim1.claim_type == 'event' and claim2.claim_type == 'event':
                        relation = 'action_sequence'
                        confidence = 0.7
                    
                    constraint = Constraint(
                        constraint_id=self._next_id(),
                        constraint_type='entity',
                        source=claim1.claim_id,
                        target=claim2.claim_id,
                        relation=relation,
                        confidence=confidence,
                        metadata={'entity': entity}
                    )
                    constraints.append(constraint)
        
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
