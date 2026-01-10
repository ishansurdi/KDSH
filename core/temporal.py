"""
Temporal Reasoning Engine

Checks temporal consistency:
- Event ordering
- Timeline validity
- Temporal constraint satisfaction
- Anachronisms

Research foundation:
- Sun et al. 2013 (Temporal constraint tracking)
- Zhang & Long 2024 (Narrative coherence)
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class TemporalEvent:
    """Represents an event with temporal information"""
    event_id: str
    description: str
    timestamp: Optional[int] = None  # Relative time (0, 1, 2, ...)
    absolute_time: Optional[str] = None  # Absolute time string
    duration: Optional[int] = None
    uncertainty: float = 0.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'description': self.description,
            'timestamp': self.timestamp,
            'absolute_time': self.absolute_time,
            'duration': self.duration,
            'uncertainty': self.uncertainty,
            'evidence': self.evidence
        }


@dataclass
class TemporalConflict:
    """Represents a temporal inconsistency"""
    conflict_type: str  # ordering, impossibility, anachronism, gap
    event1_id: str
    event2_id: Optional[str] = None
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    severity: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'conflict_type': self.conflict_type,
            'event1_id': self.event1_id,
            'event2_id': self.event2_id,
            'description': self.description,
            'evidence': self.evidence,
            'severity': self.severity
        }


class TemporalReasoningEngine:
    """
    Engine for temporal consistency checking.
    
    Performs:
    1. Timeline construction from evidence
    2. Event ordering validation
    3. Temporal constraint checking
    4. Anachronism detection
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
        self.events: Dict[str, TemporalEvent] = {}
        self.timeline: List[Tuple[int, str]] = []  # (timestamp, event_id)
    
    def build_timeline(
        self,
        claims: List,
        evidence_map: Dict[str, List]
    ):
        """
        Build timeline from claims and evidence.
        
        Args:
            claims: List of Claim objects
            evidence_map: Evidence for each claim
        """
        from .utils import extract_temporal_markers
        
        event_counter = 0
        
        # Process each claim
        for claim in claims:
            # Extract temporal info from claim
            markers = extract_temporal_markers(claim.text)
            
            # Get evidence
            evidence = evidence_map.get(claim.claim_id, [])
            
            # Create temporal event
            event = TemporalEvent(
                event_id=f"event_{event_counter}",
                description=claim.text,
                timestamp=None,  # Will be inferred
                evidence=[ev.text for ev in evidence[:2]]
            )
            
            # Try to extract temporal info
            for marker in markers:
                if marker['type'] == 'ABSOLUTE_DATE':
                    event.absolute_time = marker['text']
                elif marker['type'] == 'YEAR':
                    event.absolute_time = marker['text']
                elif marker['type'] in ['RELATIVE', 'DEICTIC']:
                    # Will need to resolve relative to other events
                    pass
            
            self.events[claim.claim_id] = event
            event_counter += 1
        
        # Infer relative timestamps using constraints
        self._infer_timestamps()
        
        # Build timeline
        self._construct_timeline()
        
        print(f"✓ Built timeline with {len(self.events)} events")
    
    def check_temporal_consistency(
        self,
        claims: List,
        evidence_map: Dict[str, List]
    ) -> List[TemporalConflict]:
        """
        Check temporal consistency of claims.
        
        Args:
            claims: List of Claim objects
            evidence_map: Evidence for each claim
        
        Returns:
            List of detected temporal conflicts
        """
        conflicts = []
        
        # Build timeline if not done
        if not self.events:
            self.build_timeline(claims, evidence_map)
        
        # Check ordering conflicts
        ordering_conflicts = self._check_ordering_conflicts(claims)
        conflicts.extend(ordering_conflicts)
        
        # Check impossible durations
        duration_conflicts = self._check_duration_conflicts()
        conflicts.extend(duration_conflicts)
        
        # Check anachronisms
        anachronisms = self._check_anachronisms(evidence_map)
        conflicts.extend(anachronisms)
        
        # Check temporal gaps
        gaps = self._check_temporal_gaps()
        conflicts.extend(gaps)
        
        print(f"⚠ Found {len(conflicts)} temporal conflicts")
        
        return conflicts
    
    def validate_event_order(
        self,
        event1_id: str,
        event2_id: str,
        expected_order: str  # 'before', 'after', 'concurrent'
    ) -> Tuple[bool, str]:
        """
        Validate ordering of two events.
        
        Args:
            event1_id: First event ID
            event2_id: Second event ID
            expected_order: Expected relationship
        
        Returns:
            (is_valid, explanation)
        """
        event1 = self.events.get(event1_id)
        event2 = self.events.get(event2_id)
        
        if not event1 or not event2:
            return False, "Event not found"
        
        if event1.timestamp is None or event2.timestamp is None:
            return True, "Cannot validate - timestamps unknown"
        
        # Check actual order
        if expected_order == 'before':
            if event1.timestamp < event2.timestamp:
                return True, "Order is correct"
            else:
                return False, f"Event {event1_id} occurs after {event2_id}, not before"
        
        elif expected_order == 'after':
            if event1.timestamp > event2.timestamp:
                return True, "Order is correct"
            else:
                return False, f"Event {event1_id} occurs before {event2_id}, not after"
        
        elif expected_order == 'concurrent':
            if event1.timestamp == event2.timestamp:
                return True, "Events are concurrent"
            else:
                return False, f"Events occur at different times"
        
        return True, "Unknown order relation"
    
    def get_timeline_summary(self) -> List[Dict[str, Any]]:
        """Get ordered list of events in timeline"""
        summary = []
        
        for timestamp, event_id in sorted(self.timeline):
            event = None
            for eid, ev in self.events.items():
                if ev.event_id == event_id:
                    event = ev
                    break
            
            if event:
                summary.append({
                    'timestamp': timestamp,
                    'event_id': event_id,
                    'description': event.description[:100],
                    'absolute_time': event.absolute_time
                })
        
        return summary
    
    def _infer_timestamps(self):
        """Infer relative timestamps from constraints"""
        # Get temporal constraints
        temporal_constraints = [
            c for c in self.constraint_graph.constraints.values()
            if c.constraint_type == 'temporal'
        ]
        
        # Initialize timestamps
        assigned_timestamps = {}
        current_time = 0
        
        # Use topological ordering from constraint graph
        ordering = self.constraint_graph.get_temporal_ordering()
        
        for i, node_id in enumerate(ordering):
            if node_id in self.events:
                event = None
                for eid, ev in self.events.items():
                    if eid == node_id:
                        event = ev
                        break
                
                if event:
                    event.timestamp = i
                    assigned_timestamps[node_id] = i
                    self.timeline.append((i, event.event_id))
        
        # Assign timestamps to remaining events
        for claim_id, event in self.events.items():
            if event.timestamp is None:
                current_time += 1
                event.timestamp = current_time
                self.timeline.append((current_time, event.event_id))
    
    def _construct_timeline(self):
        """Construct ordered timeline"""
        # Already constructed in _infer_timestamps
        self.timeline.sort(key=lambda x: x[0])
    
    def _check_ordering_conflicts(self, claims: List) -> List[TemporalConflict]:
        """Check for event ordering conflicts"""
        conflicts = []
        
        # Get temporal constraints
        temporal_constraints = [
            c for c in self.constraint_graph.constraints.values()
            if c.constraint_type == 'temporal'
        ]
        
        for constraint in temporal_constraints:
            # Check if constraint is satisfied
            source_event = None
            target_event = None
            
            for claim_id, event in self.events.items():
                if claim_id == constraint.source:
                    source_event = event
                if claim_id == constraint.target:
                    target_event = event
            
            if source_event and target_event:
                if source_event.timestamp is not None and target_event.timestamp is not None:
                    if constraint.relation == 'before':
                        if source_event.timestamp >= target_event.timestamp:
                            conflicts.append(TemporalConflict(
                                conflict_type='ordering',
                                event1_id=source_event.event_id,
                                event2_id=target_event.event_id,
                                description=f"Event order violation: {source_event.event_id} should be before {target_event.event_id}",
                                severity=0.8
                            ))
        
        return conflicts
    
    def _check_duration_conflicts(self) -> List[TemporalConflict]:
        """Check for impossible event durations"""
        conflicts = []
        
        for event_id, event in self.events.items():
            if event.duration and event.duration < 0:
                conflicts.append(TemporalConflict(
                    conflict_type='impossibility',
                    event1_id=event.event_id,
                    description=f"Impossible duration: {event.duration}",
                    severity=0.9
                ))
        
        return conflicts
    
    def _check_anachronisms(
        self,
        evidence_map: Dict[str, List]
    ) -> List[TemporalConflict]:
        """Check for anachronisms (time period mismatches)"""
        conflicts = []
        
        # Simple anachronism detection: check for modern terms in historical contexts
        modern_terms = [
            'computer', 'internet', 'smartphone', 'email',
            'television', 'airplane', 'car', 'phone'
        ]
        
        historical_markers = [
            'medieval', 'ancient', 'renaissance', '18th century',
            '19th century', 'victorian'
        ]
        
        for claim_id, event in self.events.items():
            desc_lower = event.description.lower()
            
            # Check if context is historical
            is_historical = any(marker in desc_lower for marker in historical_markers)
            
            if is_historical:
                # Check for modern terms
                for term in modern_terms:
                    if term in desc_lower:
                        conflicts.append(TemporalConflict(
                            conflict_type='anachronism',
                            event1_id=event.event_id,
                            description=f"Anachronism detected: '{term}' in historical context",
                            evidence=[event.description],
                            severity=0.7
                        ))
        
        return conflicts
    
    def _check_temporal_gaps(self) -> List[TemporalConflict]:
        """Check for unexplained temporal gaps"""
        conflicts = []
        
        # Check for large gaps in timeline
        sorted_timeline = sorted(self.timeline, key=lambda x: x[0])
        
        for i in range(len(sorted_timeline) - 1):
            time1, event1_id = sorted_timeline[i]
            time2, event2_id = sorted_timeline[i + 1]
            
            gap = time2 - time1
            
            if gap > 10:  # Arbitrary threshold
                conflicts.append(TemporalConflict(
                    conflict_type='gap',
                    event1_id=event1_id,
                    event2_id=event2_id,
                    description=f"Large temporal gap between events: {gap} time units",
                    severity=0.3
                ))
        
        return conflicts
    
    def calculate_temporal_distance(
        self,
        event1_id: str,
        event2_id: str
    ) -> Optional[int]:
        """Calculate temporal distance between two events"""
        event1 = None
        event2 = None
        
        for claim_id, event in self.events.items():
            if event.event_id == event1_id:
                event1 = event
            if event.event_id == event2_id:
                event2 = event
        
        if event1 and event2:
            if event1.timestamp is not None and event2.timestamp is not None:
                return abs(event1.timestamp - event2.timestamp)
        
        return None
