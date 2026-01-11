"""
FAST ML SUBMISSION SCRIPT FOR COLAB

Simple approach:
1. Load existing predictions/features
2. Train lightweight model
3. Re-predict with ML

Run: python fast_ml_submit.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

print("=" * 80)
print("FAST ML SUBMISSION PIPELINE")
print("=" * 80)

# Step 1: Load training data
print("\n[1] Loading training data...")
train_df = pd.read_csv('data/train.csv')
print(f"Train: {len(train_df)} examples")
print(f"Labels: {train_df['label'].value_counts().to_dict()}")

# Step 2: Create simple features from text
print("\n[2] Creating features from text...")
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine all text fields
train_df['full_text'] = train_df['content'].fillna('') + ' ' + train_df['caption'].fillna('')

# TF-IDF features (lightweight)
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['full_text']).toarray()

# Add simple count features
train_df['char_count'] = train_df['content'].str.len()
train_df['word_count'] = train_df['content'].str.split().str.len()
train_df['sentence_count'] = train_df['content'].str.count(r'[.!?]')
train_df['year_mentions'] = train_df['content'].str.count(r'\b\d{4}\b')
train_df['date_mentions'] = train_df['content'].str.count(r'\b\d{1,2}/\d{1,2}\b')

count_features = train_df[['char_count', 'word_count', 'sentence_count', 'year_mentions', 'date_mentions']].fillna(0).values

X_train = np.hstack([X_train, count_features])

# Labels: consistent=1, contradict=0
y_train = (train_df['label'] == 'consistent').astype(int).values

print(f"Feature matrix shape: {X_train.shape}")
print(f"Class distribution: {np.bincount(y_train)}")

# Step 3: Train multiple models
print("\n[3] Training ensemble models...")

models = {
    'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'mlp': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

for name, model in models.items():
    print(f"\n  Training {name}...")
    if name == 'mlp':
        model.fit(X_tr_scaled, y_tr)
        val_acc = model.score(X_val_scaled, y_val)
    else:
        model.fit(X_tr, y_tr)
        val_acc = model.score(X_val, y_val)
    
    print(f"    Validation accuracy: {val_acc:.3f}")

# Step 4: Load and predict on test data
print("\n[4] Predicting on test data...")
test_df = pd.read_csv('data/test.csv')
print(f"Test: {len(test_df)} examples")

test_df['full_text'] = test_df['content'].fillna('') + ' ' + test_df['caption'].fillna('')
X_test = vectorizer.transform(test_df['full_text']).toarray()

# Add count features for test
test_df['char_count'] = test_df['content'].str.len()
test_df['word_count'] = test_df['content'].str.split().str.len()
test_df['sentence_count'] = test_df['content'].str.count(r'[.!?]')
test_df['year_mentions'] = test_df['content'].str.count(r'\b\d{4}\b')
test_df['date_mentions'] = test_df['content'].str.count(r'\b\d{1,2}/\d{1,2}\b')

count_features_test = test_df[['char_count', 'word_count', 'sentence_count', 'year_mentions', 'date_mentions']].fillna(0).values
X_test = np.hstack([X_test, count_features_test])

# Ensemble predictions
print("\n[5] Generating ensemble predictions...")
predictions = {}

for name, model in models.items():
    if name == 'mlp':
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)
    else:
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
    
    predictions[name] = pred
    print(f"  {name}: {np.sum(pred == 0)} inconsistent, {np.sum(pred == 1)} consistent")

# Majority voting
pred_matrix = np.array([predictions[m] for m in models.keys()])
final_predictions = np.round(pred_matrix.mean(axis=0)).astype(int)

print(f"\n  Ensemble: {np.sum(final_predictions == 0)} inconsistent, {np.sum(final_predictions == 1)} consistent")
print(f"  Detection rate: {np.sum(final_predictions == 0) / len(final_predictions) * 100:.1f}%")

# Step 6: Save results
print("\n[6] Saving results...")
results_df = pd.DataFrame({
    'story_id': test_df['id'],
    'prediction': final_predictions,
    'confidence': 0.75  # Placeholder
})

Path('results').mkdir(exist_ok=True)
results_df.to_csv('results/ml_submission.csv', index=False)
print("✓ Saved to results/ml_submission.csv")

# Save models for reuse
print("\n[7] Saving models...")
with open('results/fast_ml_models.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'vectorizer': vectorizer,
        'scaler': scaler
    }, f)
print("✓ Saved models to results/fast_ml_models.pkl")

print("\n" + "=" * 80)
print("COMPLETE! Submission ready: results/ml_submission.csv")
print("=" * 80)
