# 🏆 Competition-Winning Strategy: Fix Detection System

## Current Problem
- **Expected**: ~36% inconsistent (29/80 in train)
- **Actual**: 0% inconsistent (0/60 in test)
- **Root cause**: Detection logic is too weak

## Critical Issues Identified

### 1. Constraint Extraction (constraints.py)
**Problem**: Building 0-3 constraints per backstory (should be 10-20+)
**Fix needed**: More aggressive pattern matching for:
- Temporal relations ("before", "after", "during", "while")
- Causal links ("because", "caused", "resulted in", "led to")
- Entity attributes ("was", "had", "possessed", "known for")

### 2. Temporal Conflict Detection (temporal.py)
**Problem**: 0 temporal conflicts detected across all examples
**Fix needed**:
- Check date/age contradictions
- Validate event ordering (birth → childhood → adulthood)
- Detect impossible timelines (e.g., "born 1990, graduated college 1985")

### 3. Causal Reasoning (causal.py)
**Problem**: Detecting 0-2 conflicts but not strong enough
**Fix needed**:
- Check prerequisite violations (can't graduate without attending)
- Detect circular causation
- Validate cause-effect chains

### 4. Semantic Inconsistency
**Problem**: Not checking semantic plausibility
**Fix needed**:
- Character attribute conflicts (can't be both blind and painter)
- Location impossibilities (can't be in two places simultaneously)
- Role contradictions (can't be both victim and perpetrator)

### 5. Evidence Quality Scoring
**Problem**: Weak evidence not penalized enough
**Fix needed**:
- Low relevance scores should trigger warnings
- Missing evidence for key claims is red flag
- Contradictory evidence should be weighted heavily

---

## Immediate Actions Required

### Quick Win #1: Lower Threshold (DONE ✓)
Changed from 0.5 → 0.35

### Quick Win #2: Fix Constraint Builder
Make it extract 5x more constraints

### Quick Win #3: Add Hard Rules
Implement absolute contradiction checks:
- Age/date arithmetic
- Physical impossibilities
- Logic violations

### Quick Win #4: Boost Conflict Weights
Make single conflict = high inconsistency score

---

## Implementation Priority

1. **CRITICAL** (Do first): Constraint extraction
2. **HIGH**: Temporal validation with dates/ages
3. **HIGH**: Semantic plausibility checks
4. **MEDIUM**: Causal chain validation
5. **LOW**: Fine-tune scoring weights

---

## Testing Strategy

1. Pick 5 examples from train.csv labeled "contradict"
2. Run pipeline on them manually
3. Check what signals are present but not detected
4. Add rules to catch those patterns
5. Repeat until detecting 80%+ of contradictions

---

## Expected Improvement Timeline

- **Constraint extraction fix**: +15% detection
- **Temporal validation**: +10% detection  
- **Semantic checks**: +8% detection
- **Score tuning**: +3% detection
- **Total**: ~36% detection (target!)

---

## Code Files to Modify

1. `core/constraints.py` - lines 80-150 (build_graph method)
2. `core/temporal.py` - lines 120-200 (check_temporal_consistency)
3. `core/causal.py` - lines 100-180 (check_causal_consistency)
4. `core/scorer.py` - lines 200-300 (all _score_* methods)
5. `core/classifier.py` - lines 50-80 (classify method)

---

## Ready to implement?

Choose approach:
A) **Aggressive overhaul** - rewrite detection logic (2-3 hours)
B) **Incremental fixes** - patch each module one by one (30 min each)
C) **Hybrid** - quick wins first, then deeper fixes

I recommend **B (Incremental)** - let's fix one module at a time and test!
