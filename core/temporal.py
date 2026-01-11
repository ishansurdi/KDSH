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
        
        # CRITICAL: Check age/date validation
        age_conflicts = self._check_age_date_validation()
        conflicts.extend(age_conflicts)
        
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
        
        # IMPROVED: Check all temporal constraints more strictly
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
                    # Check different relation types
                    if constraint.relation == 'before':
                        if source_event.timestamp >= target_event.timestamp:
                            conflicts.append(TemporalConflict(
                                conflict_type='ordering',
                                event1_id=source_event.event_id,
                                event2_id=target_event.event_id,
                                description=f"Event order violation: {source_event.event_id} should be before {target_event.event_id}",
                                evidence=[source_event.description[:150], target_event.description[:150]],
                                severity=0.92  # BOOSTED from 0.85, removed confidence multiplication
                            ))
                    elif constraint.relation == 'concurrent':
                        if abs(source_event.timestamp - target_event.timestamp) > 1:
                            conflicts.append(TemporalConflict(
                                conflict_type='ordering',
                                event1_id=source_event.event_id,
                                event2_id=target_event.event_id,
                                description=f"Events should be concurrent but occur at different times",
                                severity=0.88  # BOOSTED from 0.75, removed confidence multiplication
                            ))
        
        # IMPROVED: Check for circular temporal dependencies
        cycles = self.constraint_graph.detect_cycles()
        for cycle in cycles:
            if len(cycle) > 1:
                conflicts.append(TemporalConflict(
                    conflict_type='ordering',
                    event1_id=str(cycle[0]),
                    event2_id=str(cycle[-1]),
                    description=f"Circular temporal dependency detected: {' -> '.join(cycle[:3])}",
                    severity=0.95  # BOOSTED from 0.9
                ))
        
        # IMPROVED: Check for conflicting temporal markers in text
        for claim in claims:
            if claim.claim_type == 'temporal':
                text_lower = claim.text.lower()
                # Check for contradictory temporal markers in same claim
                has_before = any(word in text_lower for word in ['before', 'earlier', 'prior'])
                has_after = any(word in text_lower for word in ['after', 'later', 'following'])
                has_during = 'during' in text_lower or 'while' in text_lower
                
                # If claim has conflicting markers, flag it
                if (has_before and has_after) or (has_before and has_during and has_after):
                    conflicts.append(TemporalConflict(
                        conflict_type='ordering',
                        event1_id=claim.claim_id,
                        description=f"Claim contains conflicting temporal markers: {claim.text[:150]}",
                        severity=0.8
                    ))
        
        return conflicts
    
    def _check_duration_conflicts(self) -> List[TemporalConflict]:
        """Check for impossible event durations and AGE IMPOSSIBILITIES"""
        conflicts = []
        
        # Check impossible durations
        for event_id, event in self.events.items():
            if event.duration and event.duration < 0:
                conflicts.append(TemporalConflict(
                    conflict_type='impossibility',
                    event1_id=event.event_id,
                    description=f"Impossible duration: {event.duration}",
                    severity=0.9
                ))
        
        # AGGRESSIVE: Check AGE IMPOSSIBILITIES across ALL events
        # Extract ages from event descriptions
        age_pattern = r'(\d+)\s*(?:years?\s*old|aged|age)'
        date_pattern = r'(?:in\s+)?(1[7-9]\d{2}|20[0-2]\d)'
        
        events_with_ages = {}
        events_with_dates = {}
        life_stage_events = {}
        
        for event_id, event in self.events.items():
            text = event.description.lower()
            
            # Extract age
            age_match = re.search(age_pattern, text)
            if age_match:
                events_with_ages[event_id] = int(age_match.group(1))
            
            # Extract year
            date_match = re.search(date_pattern, text)
            if date_match:
                events_with_dates[event_id] = int(date_match.group(1))
            
            # Extract life stages
            if 'born' in text or 'birth' in text:
                life_stage_events[event_id] = ('born', 0)
            elif 'child' in text or 'childhood' in text:
                life_stage_events[event_id] = ('childhood', 5)
            elif 'graduated' in text or 'college' in text or 'university' in text:
                life_stage_events[event_id] = ('graduated', 22)
            elif 'married' in text or 'wed' in text:
                life_stage_events[event_id] = ('married', 25)
            elif 'retired' in text or 'retirement' in text:
                life_stage_events[event_id] = ('retired', 65)
            elif 'died' in text or 'death' in text:
                life_stage_events[event_id] = ('died', 70)
        
        # VALIDATION 1: Age vs life stage validation
        for event_id, age in events_with_ages.items():
            if event_id in life_stage_events:
                stage, min_age = life_stage_events[event_id]
                
                # Check impossible combinations
                if stage == 'graduated' and age < 18:
                    conflicts.append(TemporalConflict(
                        conflict_type='impossibility',
                        event1_id=event_id,
                        description=f"Cannot graduate at age {age} (too young, minimum ~18)",
                        evidence=[self.events[event_id].description],
                        severity=0.95
                    ))
                elif stage == 'married' and age < 16:
                    conflicts.append(TemporalConflict(
                        conflict_type='impossibility',
                        event1_id=event_id,
                        description=f"Unlikely to be married at age {age} (too young)",
                        evidence=[self.events[event_id].description],
                        severity=0.9
                    ))
                elif stage == 'retired' and age < 50:
                    conflicts.append(TemporalConflict(
                        conflict_type='impossibility',
                        event1_id=event_id,
                        description=f"Unlikely to retire at age {age} (too young, typical retirement 60+)",
                        evidence=[self.events[event_id].description],
                        severity=0.8
                    ))
                elif stage == 'childhood' and age > 12:
                    conflicts.append(TemporalConflict(
                        conflict_type='impossibility',
                        event1_id=event_id,
                        description=f"Age {age} is beyond childhood (should be < 13)",
                        evidence=[self.events[event_id].description],
                        severity=0.85
                    ))
        
        # VALIDATION 2: Date arithmetic (birth year + age should match event year)
        birth_year = None
        for event_id, (stage, _) in life_stage_events.items():
            if stage == 'born' and event_id in events_with_dates:
                birth_year = events_with_dates[event_id]
                break
        
        if birth_year:
            for event_id in events_with_ages:
                age = events_with_ages[event_id]
                if event_id in events_with_dates:
                    event_year = events_with_dates[event_id]
                    expected_age = event_year - birth_year
                    
                    # Check if ages match (allow ±2 year tolerance)
                    if abs(age - expected_age) > 2:
                        conflicts.append(TemporalConflict(
                            conflict_type='impossibility',
                            event1_id=event_id,
                            description=f"Age mismatch: claims age {age} in year {event_year}, but born in {birth_year} means age should be ~{expected_age}",
                            evidence=[self.events[event_id].description],
                            severity=0.95
                        ))
        
        # VALIDATION 3: Life stage ordering (can't graduate before childhood)
        life_stage_order = ['born', 'childhood', 'graduated', 'married', 'retired', 'died']
        sorted_stages = sorted(life_stage_events.items(), 
                              key=lambda x: self.events[x[0]].timestamp if self.events[x[0]].timestamp else 999)
        
        for i, (event_id1, (stage1, _)) in enumerate(sorted_stages):
            for event_id2, (stage2, _) in sorted_stages[i+1:]:
                idx1 = life_stage_order.index(stage1) if stage1 in life_stage_order else -1
                idx2 = life_stage_order.index(stage2) if stage2 in life_stage_order else -1
                
                if idx1 >= 0 and idx2 >= 0 and idx1 > idx2:
                    conflicts.append(TemporalConflict(
                        conflict_type='impossibility',
                        event1_id=event_id1,
                        event2_id=event_id2,
                        description=f"Life stage ordering violation: {stage1} cannot come before {stage2}",
                        evidence=[self.events[event_id1].description, self.events[event_id2].description],
                        severity=0.95
                    ))
        
        return conflicts
    
    def _check_anachronisms(
        self,
        evidence_map: Dict[str, List]
    ) -> List[TemporalConflict]:
        """Check for anachronisms (time period mismatches)"""
        conflicts = []
        
        # IMPROVED: More comprehensive anachronism detection
        modern_tech = {
            '20th_century': ['computer', 'internet', 'smartphone', 'email', 'television', 'tv', 'radio', 'airplane', 'plane', 'car', 'automobile', 'telephone', 'phone', 'mobile', 'cell phone', 'website', 'digital', 'laser', 'nuclear', 'satellite', 'rocket', 'jet', 'plastic', 'nylon', 'radar', 'sonar', 'video', 'dvd', 'cd', 'usb'],
            '19th_century_plus': ['telegraph', 'railroad', 'railway', 'train', 'steam engine', 'photography', 'camera', 'photograph', 'electric', 'electricity', 'lightbulb', 'phonograph'],
            '18th_century_plus': ['printing press', 'gunpowder', 'cannon', 'musket', 'rifle']
        }
        
        historical_periods = {
            'ancient': ['ancient', 'classical', 'roman empire', 'rome', 'greek', 'greece', 'bc', 'b.c.', 'antiquity', 'pharaoh', 'caesar'],
            'medieval': ['medieval', 'middle ages', 'feudal', 'knight', 'castle', 'crusade', 'plague', 'dark ages', 'gothic'],
            'renaissance': ['renaissance', '15th century', '16th century', '1400s', '1500s'],
            '17th_18th': ['17th century', '18th century', '1600s', '1700s', 'enlightenment', 'colonial', 'revolutionary war', 'napoleon'],
            '19th': ['19th century', 'victorian', '1800s', 'industrial revolution', 'civil war', 'napoleon'],
        }
        
        # Check each event
        for claim_id, event in self.events.items():
            desc_lower = event.description.lower()
            
            # Determine historical period
            detected_period = None
            for period, markers in historical_periods.items():
                if any(marker in desc_lower for marker in markers):
                    detected_period = period
                    break
            
            if detected_period:
                # Check for anachronistic technology
                for tech_era, terms in modern_tech.items():
                    # If detected period is before tech era, check for anachronistic terms
                    if detected_period in ['ancient', 'medieval', 'renaissance']:
                        # These periods can't have any modern tech
                        for term in terms:
                            if term in desc_lower:
                                conflicts.append(TemporalConflict(
                                    conflict_type='anachronism',
                                    event1_id=event.event_id,
                                    description=f"Anachronism: '{term}' in {detected_period} context",
                                    evidence=[event.description[:200]],
                                    severity=0.93  # BOOSTED from 0.85
                                ))
                    elif detected_period == '17th_18th' and tech_era == '20th_century':
                        for term in terms:
                            if term in desc_lower:
                                conflicts.append(TemporalConflict(
                                    conflict_type='anachronism',
                                    event1_id=event.event_id,
                                    description=f"Anachronism: '{term}' in {detected_period} context",
                                    evidence=[event.description[:200]],
                                    severity=0.90  # BOOSTED from 0.8
                                ))
            
            # IMPROVED: Check evidence for temporal contradictions
            evidence_list = evidence_map.get(claim_id, [])
            for evidence in evidence_list:
                ev_lower = evidence.text.lower()
                # Look for explicit dates/years in evidence
                year_pattern = r'\b(1[0-9]{3}|20[0-2][0-9])\b'
                years_in_evidence = re.findall(year_pattern, ev_lower)
                years_in_claim = re.findall(year_pattern, desc_lower)
                
                if years_in_evidence and years_in_claim:
                    # Check if years are far apart (potential inconsistency)
                    for ev_year in years_in_evidence:
                        for claim_year in years_in_claim:
                            if abs(int(ev_year) - int(claim_year)) > 50:
                                conflicts.append(TemporalConflict(
                                    conflict_type='anachronism',
                                    event1_id=event.event_id,
                                    description=f"Temporal mismatch: claim mentions {claim_year} but evidence mentions {ev_year}",
                                    evidence=[evidence.text[:200]],
                                    severity=0.88  # BOOSTED from 0.75
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
    
    def _check_age_date_validation(self) -> List[TemporalConflict]:
        """CRITICAL: Validate age vs dates vs life stages."""
        import re
        
        conflicts = []
        
        # Extract ALL age and date information
        ages_by_event = {}
        dates_by_event = {}
        life_stages_by_event = {}
        
        for event_id, event in self.events.items():
            text = event.description.lower()
            
            # Extract age
            age_pattern = r'(\d+)\s*(?:years?\s*old|aged|age)'
            age_match = re.search(age_pattern, text)
            if age_match:
                ages_by_event[event_id] = int(age_match.group(1))
            
            # Extract year
            year_pattern = r'\b(1[7-9]\d{2}|20[0-2]\d)\b'
            year_match = re.search(year_pattern, text)
            if year_match:
                dates_by_event[event_id] = int(year_match.group(1))
            
            # Detect life stages
            if 'born' in text or 'birth' in text:
                life_stages_by_event[event_id] = ('born', 0, 0)
            elif 'child' in text or 'childhood' in text:
                life_stages_by_event[event_id] = ('childhood', 5, 12)
            elif 'graduated' in text or 'university' in text or 'college' in text:
                life_stages_by_event[event_id] = ('graduated', 22, 30)
            elif 'married' in text or 'wed' in text:
                life_stages_by_event[event_id] = ('married', 18, 100)
            elif 'retired' in text:
                life_stages_by_event[event_id] = ('retired', 60, 100)
        
        # VALIDATION 1: Age vs Life Stage
        for event_id, age in ages_by_event.items():
            if event_id in life_stages_by_event:
                stage, min_age, max_age = life_stages_by_event[event_id]
                
                if age < min_age or (max_age < 100 and age > max_age):
                    conflicts.append(TemporalConflict(
                        conflict_type='impossibility',
                        event1_id=event_id,
                        description=f"Age {age} impossible for life stage '{stage}' (expected {min_age}-{max_age})",
                        evidence=[self.events[event_id].description],
                        severity=0.95
                    ))
        
        # VALIDATION 2: Infer birth year and check consistency
        birth_years = {}
        for event_id in ages_by_event:
            if event_id in dates_by_event:
                age = ages_by_event[event_id]
                year = dates_by_event[event_id]
                inferred_birth = year - age
                birth_years[event_id] = inferred_birth
        
        # Check if inferred birth years are consistent
        if len(birth_years) > 1:
            birth_year_values = list(birth_years.values())
            for i, by1 in enumerate(birth_year_values):
                for by2 in birth_year_values[i+1:]:
                    if abs(by1 - by2) > 2:  # Allow 2 year tolerance
                        conflicts.append(TemporalConflict(
                            conflict_type='impossibility',
                            event1_id=list(birth_years.keys())[i],
                            event2_id=list(birth_years.keys())[i+1],
                            description=f"Birth year inconsistent: inferred {by1} vs {by2}",
                            severity=0.90
                        ))
        
        return conflicts
