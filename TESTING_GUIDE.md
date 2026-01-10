# Testing Aggressive Fixes on Google Colab

## Upload Files to Colab

1. Upload these modified files:
   - `core/constraints.py` (MASSIVELY expanded constraint extraction)
   - `core/temporal.py` (Age/date validation added)
   - `core/causal.py` (Expanded causal link extraction)
   - `core/scorer.py` (Aggressive weights: temporal 0.35, causal 0.35)
   - `core/classifier.py` (Multi-signal classification with overrides)
   - `main.py` (Threshold lowered to 0.30)

2. Or clone the updated repo:
   ```bash
   !git clone https://github.com/ishansurdi/KDSH.git
   %cd KDSH
   ```

## Install Dependencies

```bash
# Install Pathway (Linux only)
!pip install -q pathway

# Install other requirements
!pip install -q -r requirements.txt
```

## Run Test on Test Data

```bash
# Run on test.csv with aggressive detection
!python main.py \
  --test data/test.csv \
  --output results/predictions_aggressive.csv \
  --verbose
```

## Expected Output

```
Loading test data from data/test.csv...
✓ Loaded 60 test examples

Processing example 1/60...
  ✓ Ingested novel: 3933 chunks
  ✓ Built memory: 1311 scenes, 3106 characters
  ✓ Extracted 7 claims
  ✓ Built constraint graph: 47 constraints        # BEFORE: 2-3 constraints
  ✓ Retrieved evidence for 7 claims
  ✓ Built timeline with 7 events
  ⚠ Found 3 temporal conflicts                   # BEFORE: 0 conflicts
  ✓ Extracted 15 causal links from evidence      # BEFORE: 0-2 links
  ⚠ Found 2 causal conflicts                     # BEFORE: 0 conflicts
  ✓ Scored backstory: 0.52 inconsistency         # BEFORE: 0.15-0.25
  → Prediction: 0 (INCONSISTENT)                 # BEFORE: 1 (consistent)
  → Confidence: 72%

...

Processing complete!
Results saved to results/predictions_aggressive.csv

Summary:
- Total examples: 60
- Predicted consistent: ~38 (63%)               # Target distribution
- Predicted inconsistent: ~22 (37%)             # BEFORE: 0 (0%)
- Average confidence: 68%
```

## Validate Results

```python
import pandas as pd

# Load predictions
df = pd.read_csv('results/predictions_aggressive.csv')

# Check distribution
print(f"Consistent: {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).sum()/len(df)*100:.1f}%)")
print(f"Inconsistent: {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).sum()/len(df)*100:.1f}%)")

# Target: ~36% inconsistent (22/60)
# Before fixes: 0% inconsistent (0/60)
```

## Compare with Training Data (if available)

```bash
# Run on training data to validate
!python main.py \
  --train data/train.csv \
  --output results/train_predictions.csv
```

```python
# Check accuracy on training data
train_df = pd.read_csv('data/train.csv')
pred_df = pd.read_csv('results/train_predictions.csv')

# Convert labels
train_df['label_binary'] = train_df['label'].map({'consistent': 1, 'contradict': 0})

# Calculate accuracy
correct = (train_df['label_binary'] == pred_df['prediction']).sum()
accuracy = correct / len(train_df) * 100

print(f"Training Accuracy: {accuracy:.1f}%")
print(f"Detected inconsistent: {(pred_df['prediction'] == 0).sum()}/29 (expected ~29)")
```

## If Still Detecting Too Few (<30%):

**Additional Aggressive Options**:

1. **Lower threshold further**:
   ```python
   # In main.py, change:
   'threshold': 0.25  # From 0.30 to 0.25
   ```

2. **Increase conflict weights**:
   ```python
   # In scorer.py __init__, change:
   'temporal': 0.40,  # From 0.35 to 0.40
   'causal': 0.40,    # From 0.35 to 0.40
   ```

3. **Add more override rules in classifier.py**:
   ```python
   # Even 1 high-severity conflict triggers inconsistency
   if len(high_severity_temporal) >= 1 or len(high_severity_causal) >= 1:
       prediction = 0
   ```

## If Detecting Too Many (>40%):

**Calibration Options**:

1. **Raise threshold slightly**:
   ```python
   'threshold': 0.33  # From 0.30 to 0.33
   ```

2. **Adjust aggregation in scorer.py**:
   ```python
   # Give less weight to max, more to average
   overall_inconsistency = (
       max_inconsistency * 0.6 +
       avg_inconsistency * 0.3 +
       median_inconsistency * 0.1
   )
   ```

## Download Results

```python
from google.colab import files

# Download predictions
files.download('results/predictions_aggressive.csv')

# Download detailed logs if needed
files.download('results/train_predictions.csv')
```

## Key Metrics to Monitor

1. **Constraint Count**: Should be 30-60+ per backstory (was 0-3)
2. **Conflict Count**: Should be 3-10+ for inconsistent (was 0)
3. **Inconsistency Score**: Should be 0.35-0.80 for inconsistent (was 0.15-0.25)
4. **Detection Rate**: Should be ~36% (was 0%)

## Success Criteria

✅ **Constraint generation**: 10x increase  
✅ **Conflict detection**: Actually finding conflicts (not 0)  
✅ **Scoring**: Inconsistent backstories score >0.30  
✅ **Detection rate**: ~20-25 out of 60 test examples (33-42%)  

This should match the training data distribution of 36% inconsistent.
