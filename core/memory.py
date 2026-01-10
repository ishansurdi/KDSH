"""
Hierarchical Narrative Memory System

Implements multi-level memory for tracking:
- Scene-level events
- Episode-level story arcs  
- Character state evolution
- Timeline and causality

Research foundations:
- Zhang & Long 2024 (Narrative gap & coherence)
- Weaving Topic Continuity 2025 (Hierarchical memory)
- Lattimer et al. 2023 (Long-doc inconsistency)
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class CharacterState:
    """Character state at a specific point in narrative"""
    character_name: str
    location: Optional[str] = None
    alive: bool = True
    relationships: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    knowledge: Set[str] = field(default_factory=set)
    commitments: List[str] = field(default_factory=list)
    timestamp: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'character_name': self.character_name,
            'location': self.location,
            'alive': self.alive,
            'relationships': self.relationships,
            'attributes': self.attributes,
            'knowledge': list(self.knowledge),
            'commitments': self.commitments,
            'timestamp': self.timestamp
        }


@dataclass
class Scene:
    """Scene-level narrative unit"""
    scene_id: str
    text: str
    characters: List[str]
    location: Optional[str] = None
    events: List[str] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    timestamp: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene_id': self.scene_id,
            'text': self.text[:200] + "..." if len(self.text) > 200 else self.text,
            'characters': self.characters,
            'location': self.location,
            'events': self.events,
            'temporal_markers': self.temporal_markers,
            'chunk_ids': self.chunk_ids,
            'timestamp': self.timestamp
        }


@dataclass
class Episode:
    """Episode-level story arc (multiple scenes)"""
    episode_id: str
    name: str
    scenes: List[str]
    main_characters: List[str]
    plot_points: List[str] = field(default_factory=list)
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'name': self.name,
            'scenes': self.scenes,
            'main_characters': self.main_characters,
            'plot_points': self.plot_points,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class HierarchicalNarrativeMemory:
    """
    Multi-level narrative memory system.
    
    Tracks narrative state at multiple granularities:
    1. Scene level: Individual events and interactions
    2. Episode level: Story arcs spanning multiple scenes
    3. Character level: State evolution over time
    4. Timeline level: Temporal ordering of events
    """
    
    def __init__(self):
        self.scenes: Dict[str, Scene] = {}
        self.episodes: Dict[str, Episode] = {}
        self.character_states: Dict[str, List[CharacterState]] = defaultdict(list)
        self.timeline: List[Tuple[int, str, str]] = []  # (timestamp, event_type, scene_id)
        self.entity_index: Dict[str, List[str]] = defaultdict(list)  # entity -> scene_ids
        
    def add_scene(self, scene: Scene):
        """Add a scene to memory"""
        self.scenes[scene.scene_id] = scene
        
        # Index characters
        for char in scene.characters:
            self.entity_index[char].append(scene.scene_id)
        
        # Index location
        if scene.location:
            self.entity_index[f"LOC:{scene.location}"].append(scene.scene_id)
        
        # Add to timeline
        if scene.timestamp is not None:
            for event in scene.events:
                self.timeline.append((scene.timestamp, event, scene.scene_id))
        
        # Sort timeline
        self.timeline.sort(key=lambda x: x[0])
    
    def add_episode(self, episode: Episode):
        """Add an episode to memory"""
        self.episodes[episode.episode_id] = episode
    
    def update_character_state(
        self, 
        character_name: str, 
        state: CharacterState
    ):
        """Update character state at a specific time"""
        self.character_states[character_name].append(state)
        # Sort by timestamp
        self.character_states[character_name].sort(
            key=lambda s: s.timestamp if s.timestamp else 0
        )
    
    def get_character_state_at(
        self, 
        character_name: str, 
        timestamp: int
    ) -> Optional[CharacterState]:
        """Get character state at specific timestamp"""
        states = self.character_states.get(character_name, [])
        
        # Find latest state before or at timestamp
        relevant_state = None
        for state in states:
            if state.timestamp is None or state.timestamp <= timestamp:
                relevant_state = state
            else:
                break
        
        return relevant_state
    
    def get_character_evolution(
        self, 
        character_name: str
    ) -> List[CharacterState]:
        """Get full evolution of character state"""
        return self.character_states.get(character_name, [])
    
    def find_scenes_with_entity(self, entity: str) -> List[Scene]:
        """Find all scenes mentioning an entity"""
        scene_ids = self.entity_index.get(entity, [])
        return [self.scenes[sid] for sid in scene_ids if sid in self.scenes]
    
    def find_scenes_with_characters(
        self, 
        characters: List[str],
        require_all: bool = False
    ) -> List[Scene]:
        """
        Find scenes with specific characters.
        
        Args:
            characters: List of character names
            require_all: If True, require all characters; else any
        """
        if not characters:
            return []
        
        # Get scenes for each character
        scene_sets = [set(self.entity_index.get(char, [])) for char in characters]
        
        # Intersect or union
        if require_all:
            common_scenes = set.intersection(*scene_sets) if scene_sets else set()
        else:
            common_scenes = set.union(*scene_sets) if scene_sets else set()
        
        return [self.scenes[sid] for sid in common_scenes if sid in self.scenes]
    
    def get_timeline_segment(
        self, 
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Tuple[int, str, str]]:
        """Get timeline events in a time range"""
        filtered = []
        for timestamp, event, scene_id in self.timeline:
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            filtered.append((timestamp, event, scene_id))
        
        return filtered
    
    def check_character_consistency(
        self, 
        character_name: str,
        claim: str
    ) -> Dict[str, Any]:
        """
        Check if a claim about a character is consistent with their evolution.
        
        Args:
            character_name: Name of character
            claim: Claim to check (e.g., "was in Paris")
        
        Returns:
            Dict with consistency check results
        """
        states = self.get_character_evolution(character_name)
        
        conflicts = []
        supporting_evidence = []
        
        # Simple pattern matching (in production, use NLP)
        claim_lower = claim.lower()
        
        for state in states:
            # Check location
            if state.location and "in" in claim_lower:
                if state.location.lower() in claim_lower:
                    supporting_evidence.append({
                        'state': state.to_dict(),
                        'reason': f'Character was in {state.location}'
                    })
                elif "not in" not in claim_lower:
                    conflicts.append({
                        'state': state.to_dict(),
                        'reason': f'Character was in {state.location}, not as claimed'
                    })
            
            # Check alive status
            if "dead" in claim_lower or "died" in claim_lower:
                if not state.alive:
                    supporting_evidence.append({
                        'state': state.to_dict(),
                        'reason': 'Character death matches claim'
                    })
                elif state.alive and "not dead" not in claim_lower:
                    conflicts.append({
                        'state': state.to_dict(),
                        'reason': 'Character alive, contradicts death claim'
                    })
        
        return {
            'character': character_name,
            'claim': claim,
            'is_consistent': len(conflicts) == 0,
            'conflicts': conflicts,
            'supporting_evidence': supporting_evidence
        }
    
    def extract_narrative_from_chunks(
        self, 
        chunks: List[Dict[str, Any]],
        novel_id: str
    ):
        """
        Build narrative memory from document chunks.
        
        Args:
            chunks: List of text chunks from novel
            novel_id: Novel identifier
        """
        from .utils import extract_entities, extract_temporal_markers
        
        scene_counter = 0
        current_timestamp = 0
        
        # Group chunks into scenes (simple heuristic: every 3-5 chunks)
        scene_size = 3
        
        for i in range(0, len(chunks), scene_size):
            scene_chunks = chunks[i:i+scene_size]
            
            # Combine text
            scene_text = " ".join([c.get('text', '') for c in scene_chunks])
            
            # Extract entities
            entities = extract_entities(scene_text)
            characters = [e['text'] for e in entities if e['type'] == 'PERSON']
            locations = [e['text'] for e in entities if e['type'] == 'LOCATION']
            
            # Extract temporal markers
            temporal = extract_temporal_markers(scene_text)
            temporal_texts = [t['text'] for t in temporal]
            
            # Create scene
            scene = Scene(
                scene_id=f"{novel_id}_scene_{scene_counter}",
                text=scene_text,
                characters=list(set(characters)),
                location=locations[0] if locations else None,
                events=[f"Event in scene {scene_counter}"],
                temporal_markers=temporal_texts,
                chunk_ids=[c.get('chunk_id', f'chunk_{i+j}') for j, c in enumerate(scene_chunks)],
                timestamp=current_timestamp
            )
            
            self.add_scene(scene)
            
            # Update character states
            for char in set(characters):
                state = CharacterState(
                    character_name=char,
                    location=locations[0] if locations else None,
                    timestamp=current_timestamp
                )
                self.update_character_state(char, state)
            
            scene_counter += 1
            current_timestamp += 1
        
        print(f"✓ Built narrative memory: {len(self.scenes)} scenes, "
              f"{len(self.character_states)} characters tracked")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_scenes': len(self.scenes),
            'total_episodes': len(self.episodes),
            'characters_tracked': len(self.character_states),
            'timeline_events': len(self.timeline),
            'indexed_entities': len(self.entity_index)
        }
    
    def export_memory(self, filepath: str):
        """Export memory to JSON"""
        memory_data = {
            'scenes': {sid: scene.to_dict() for sid, scene in self.scenes.items()},
            'episodes': {eid: ep.to_dict() for eid, ep in self.episodes.items()},
            'character_states': {
                char: [state.to_dict() for state in states]
                for char, states in self.character_states.items()
            },
            'timeline': self.timeline,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2)
        
        print(f"✓ Exported memory to {filepath}")
