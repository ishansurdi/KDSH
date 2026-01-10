# 🚀 Google Colab Setup Guide

This project is developed on **Windows** but executed on **Google Colab (Linux)** where Pathway is fully supported.

---

## 📋 Quick Start

### Step 1: Upload to Colab

**Option A: Via GitHub**
1. Push your code to GitHub
2. In Colab, clone: `!git clone https://github.com/your-username/KDSH.git`

**Option B: Via ZIP**
1. Zip your project folder
2. Upload to Colab
3. Extract: `!unzip KDSH.zip`

### Step 2: Install Dependencies

```python
# Run this in the first Colab cell
!pip install pathway numpy pandas scikit-learn scipy jupyter streamlit plotly matplotlib tqdm python-dateutil
```

### Step 3: Verify Setup

```python
import pathway as pw
print(f"✅ Pathway version: {pw.__version__}")

# Test imports
from core import PathwayDocumentStore, HierarchicalNarrativeMemory
print("✅ All core modules imported successfully")
```

---

## 📦 Complete Colab Setup Notebook

Copy this into a new Colab notebook:

```python
# Cell 1: Upload and Extract
from google.colab import files
import zipfile

# Upload ZIP file
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Extract
!unzip -q {filename}
%cd KDSH

# Cell 2: Install Dependencies
!pip install -q pathway numpy pandas scikit-learn scipy streamlit plotly matplotlib tqdm python-dateutil

# Cell 3: Verify Installation
import pathway as pw
import sys
sys.path.insert(0, '.')

from core import PathwayDocumentStore, HierarchicalNarrativeMemory, ConsistencyClassifier
print("✅ Setup complete! Ready to run.")

# Cell 4: Download Novels (Required!)
!python download_novels.py

# Alternative: Upload manually
# from google.colab import files
# print("Upload novel text files:")
# novel_files = files.upload()
# import os
# os.makedirs('data/novels', exist_ok=True)
# for filename in novel_files:
#     with open(f'data/novels/{filename}', 'wb') as f:
#         f.write(novel_files[filename])


# Cell 5: Run Pipeline
from main import NarrativeConsistencyPipeline

# Initialize
pipeline = NarrativeConsistencyPipeline()

# Ingest novel
novel_path = f"data/novels/{list(novel_files.keys())[0]}"
pipeline.ingest_novel(novel_path)

# Check backstory
backstory = "Your test backstory here..."
result = pipeline.process_example("test_001", novel_path, backstory)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Rationale: {result['rationale']}")
```

---

## 🔄 Development Workflow

### On Windows (VS Code):
1. ✅ Edit all Python files
2. ✅ Modify notebooks
3. ✅ Update documentation
4. ✅ Git commit/push
5. ❌ **Do NOT run** Pathway code

### In Google Colab:
1. ✅ Pull/upload latest code
2. ✅ Install dependencies
3. ✅ Run all notebooks
4. ✅ Test pipeline
5. ✅ Generate results
6. ✅ Download outputs

---

## 📊 Run Full Pipeline on Colab

```python
# Process entire test dataset
import pandas as pd

# Load test data
test_df = pd.read_csv('data/test.csv')

# Process each example
results = []
for idx, row in test_df.iterrows():
    result = pipeline.process_example(
        row['story_id'],
        row['novel_file'],
        row['backstory']
    )
    results.append(result)
    print(f"✅ Processed {idx+1}/{len(test_df)}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/predictions.csv', index=False)

# Download
from google.colab import files
files.download('results/predictions.csv')
```

---

## 🎯 Jupyter Notebooks in Colab

All notebooks in `notebooks/` are Colab-compatible:

1. **Upload notebooks folder** to Colab
2. **Open any .ipynb** file
3. **Run cells sequentially**

Example:
```python
# In Colab
!ls notebooks/
# 01_ingestion.ipynb
# 02_memory.ipynb
# ...
# run_pipeline.ipynb
```

---

## 📈 Generate Submission File

```python
# Final submission generation
import zipfile
import os
from datetime import datetime

# Create submission folder
submission_dir = f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(submission_dir, exist_ok=True)

# Copy required files
!cp -r core/ {submission_dir}/
!cp -r data/ {submission_dir}/
!cp -r notebooks/ {submission_dir}/
!cp -r dashboard/ {submission_dir}/
!cp main.py requirements.txt README.md {submission_dir}/
!cp results/predictions.csv {submission_dir}/

# Create ZIP
!zip -r submission.zip {submission_dir}

# Download
from google.colab import files
files.download('submission.zip')
```

---

## 🔧 Troubleshooting

### Issue: Pathway Import Error
```python
# Check Pathway installation
!pip show pathway

# Reinstall if needed
!pip install --upgrade pathway
```

### Issue: Module Not Found
```python
# Verify you're in project root
!pwd
# Should show: /content/KDSH

# Check core modules exist
!ls core/
```

### Issue: Novel File Not Found
```python
# Check novels directory
!ls data/novels/

# Upload novels manually
from google.colab import files
novel_files = files.upload()
```

---

## 💾 Save Work from Colab

```python
# Option 1: Download results
from google.colab import files
files.download('results/predictions.csv')

# Option 2: Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r results/ /content/drive/MyDrive/KDSH_results/

# Option 3: Commit to GitHub
!git config --global user.email "your@email.com"
!git config --global user.name "Your Name"
!git add results/
!git commit -m "Add results from Colab run"
!git push
```

---

## 🎓 Example Colab Session

```python
# Complete working example
import pathway as pw
from main import NarrativeConsistencyPipeline

# 1. Initialize pipeline
pipeline = NarrativeConsistencyPipeline()

# 2. Ingest novel
pipeline.ingest_novel("data/novels/sample_novel.txt")

# 3. Test backstory
backstory = """
The protagonist was born in 1990 in New York.
She moved to California in 2005 to study art.
Her father was a famous painter.
"""

result = pipeline.process_example(
    "test_001",
    "data/novels/sample_novel.txt",
    backstory
)

# 4. View results
print(f"✅ Prediction: {'CONSISTENT' if result['prediction'] == 1 else 'INCONSISTENT'}")
print(f"✅ Confidence: {result['confidence']:.2%}")
print(f"✅ Rationale:\n{result['rationale']}")
```

---

## 🏆 Submission Checklist

Before downloading final submission:

- [ ] All dependencies installed in Colab
- [ ] Pathway working correctly
- [ ] All notebooks run successfully
- [ ] Pipeline processes test data
- [ ] Results saved to results/predictions.csv
- [ ] Code cleaned and commented
- [ ] README updated
- [ ] Submission ZIP created
- [ ] ZIP downloaded from Colab

---

## 📞 Quick Commands Reference

```bash
# Install everything
!pip install pathway numpy pandas scikit-learn scipy streamlit plotly matplotlib tqdm

# Upload files
from google.colab import files; files.upload()

# Check setup
!python setup.py

# Run pipeline
!python main.py --test data/test.csv --output results/predictions.csv

# Download results
from google.colab import files; files.download('results/predictions.csv')

# Create submission
!zip -r submission.zip core/ data/ notebooks/ dashboard/ main.py requirements.txt README.md results/
```

---

**🎯 You're ready! Edit on Windows, execute on Colab, win the competition! 🏆**
