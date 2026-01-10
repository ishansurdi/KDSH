# 🎯 Development Workflow

## Overview

This project uses a **split development model** because Pathway requires Linux/macOS:

```
┌─────────────────┐         ┌─────────────────┐
│   Windows PC    │         │  Google Colab   │
│   (VS Code)     │ ──────> │    (Linux)      │
│                 │         │                 │
│  ✅ Edit code   │         │  ✅ Run code    │
│  ✅ Git commit  │  Sync   │  ✅ Test        │
│  ✅ Document    │         │  ✅ Package     │
│  ❌ No execute  │         │  ✅ Submit      │
└─────────────────┘         └─────────────────┘
```

---

## 📝 On Windows (Development)

### What You CAN Do:
✅ Edit all Python files in VS Code  
✅ Modify Jupyter notebooks  
✅ Update documentation  
✅ Manage Git (commit, push, pull)  
✅ Install dependencies (will get placeholder pathway)  
✅ Use Copilot for code generation  
✅ Plan and design architecture  

### What You CANNOT Do:
❌ Run Pathway code  
❌ Execute main.py  
❌ Test the pipeline  
❌ Generate results  
❌ Run notebooks end-to-end  

### Development Commands:
```bash
# Safe to run on Windows
git status
git add .
git commit -m "Update code"
git push

# View/edit files
code core/pathway_store.py
code notebooks/01_ingestion.ipynb
code README.md
```

---

## 🚀 On Google Colab (Execution)

### Setup (First Time):

**Method 1: From GitHub**
```python
# Cell 1: Clone repository
!git clone https://github.com/your-username/KDSH.git
%cd KDSH

# Cell 2: Install dependencies
!pip install -q pathway numpy pandas scikit-learn scipy streamlit plotly matplotlib tqdm

# Cell 3: Verify
import pathway as pw
from core import PathwayDocumentStore
print("✅ Setup complete!")
```

**Method 2: ZIP Upload**
```python
# Cell 1: Upload ZIP
from google.colab import files
uploaded = files.upload()  # Select KDSH.zip

# Cell 2: Extract
!unzip -q KDSH.zip
%cd KDSH

# Cell 3: Install & verify (same as above)
```

---

## 🔄 Sync Methods

### Option A: GitHub (Recommended)

**On Windows:**
```bash
git add .
git commit -m "Update code"
git push origin main
```

**On Colab:**
```python
!git pull origin main
```

### Option B: Manual ZIP

**On Windows:**
1. Right-click project folder
2. Send to → Compressed (zipped) folder
3. Upload to Google Drive

**On Colab:**
```python
from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/KDSH.zip
%cd KDSH
```

---

## 🧪 Testing Workflow

### 1. Make Changes on Windows
```bash
# Edit files in VS Code
code core/retriever.py

# Commit changes
git add core/retriever.py
git commit -m "Improve retrieval logic"
git push
```

### 2. Test on Colab
```python
# Pull latest code
!git pull origin main

# Run specific module test
%run notebooks/04_retrieval.ipynb

# Or test full pipeline
!python main.py --test data/test.csv --output results/predictions.csv
```

### 3. Iterate
- If tests pass → Continue development
- If tests fail → Fix on Windows, push, test again

---

## 📦 Submission Workflow

### Final Steps (All on Colab):

```python
# 1. Run full pipeline
!python main.py --train data/train.csv --calibrate
!python main.py --test data/test.csv --output results/predictions.csv

# 2. Verify results
import pandas as pd
results = pd.read_csv('results/predictions.csv')
print(results.head())
print(f"Total predictions: {len(results)}")

# 3. Create submission package
!mkdir submission
!cp -r core/ data/ notebooks/ dashboard/ main.py requirements.txt README.md results/ submission/
!zip -r submission.zip submission/

# 4. Download
from google.colab import files
files.download('submission.zip')
```

---

## 🎓 Example Daily Workflow

### Morning (Windows):
```bash
9:00 AM  - Open VS Code
9:15 AM  - Review yesterday's Colab results
9:30 AM  - Implement new feature in core/scorer.py
10:30 AM - Update corresponding notebook
11:00 AM - Commit and push changes
```

### Afternoon (Google Colab):
```python
2:00 PM  - Open Colab notebook
2:05 PM  - Pull latest code: !git pull
2:10 PM  - Run updated notebook
2:30 PM  - Review results, identify issues
3:00 PM  - Download logs for debugging
```

### Evening (Windows):
```bash
7:00 PM  - Fix bugs based on Colab logs
8:00 PM  - Push fixes
8:30 PM  - Schedule Colab run for overnight testing
```

---

## 🐛 Debugging Across Platforms

### Issue Found in Colab:

1. **Capture error details:**
```python
# In Colab
import traceback
try:
    pipeline.process_example(...)
except Exception as e:
    error_log = traceback.format_exc()
    with open('error_log.txt', 'w') as f:
        f.write(error_log)
    files.download('error_log.txt')
```

2. **Fix on Windows:**
```bash
# Read error_log.txt in VS Code
# Fix the issue
git add .
git commit -m "Fix exception in process_example"
git push
```

3. **Verify on Colab:**
```python
!git pull
# Test again
```

---

## 💾 Data Management

### Small Files (<100MB):
- Store in GitHub repository
- Sync automatically

### Large Files (novels, models):
- **Option 1:** Upload directly to Colab each session
- **Option 2:** Store in Google Drive, mount in Colab
- **Option 3:** Download from public URL in Colab

```python
# Option 2: Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/novels/*.txt data/novels/

# Option 3: Public URL
!wget https://example.com/novel.txt -O data/novels/novel.txt
```

---

## 📊 Results Management

### Download Results from Colab:
```python
# Individual file
from google.colab import files
files.download('results/predictions.csv')

# Multiple files
!zip results.zip results/*
files.download('results.zip')

# To Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r results/ /content/drive/MyDrive/KDSH_results/
```

### Analyze on Windows:
```bash
# Unzip results
# Open in Excel, Pandas, etc.
# Update visualization notebooks
```

---

## ✅ Checklist: Before Each Colab Session

- [ ] Latest code committed and pushed
- [ ] requirements.txt updated if needed
- [ ] Sample data files prepared
- [ ] Clear goal for the session (test X, run Y, etc.)

## ✅ Checklist: After Each Colab Session

- [ ] Results downloaded
- [ ] Logs saved
- [ ] Errors documented
- [ ] Next steps identified
- [ ] GitHub updated if needed

---

## 🚨 Important Reminders

1. **Never** try to run Pathway on Windows
2. **Always** commit before switching contexts
3. **Keep** Git repo clean (no large files)
4. **Test** incrementally on Colab
5. **Document** issues immediately
6. **Backup** Colab notebooks regularly

---

## 🎯 Quick Reference

| Task | Command/Location |
|------|------------------|
| Edit code | Windows VS Code |
| Run code | Google Colab |
| Commit changes | Windows Git |
| Install deps | Colab `!pip install` |
| Upload files | Colab `files.upload()` |
| Download results | Colab `files.download()` |
| Full pipeline | Colab `!python main.py` |
| Debug | Windows + Colab logs |

---

**This workflow lets you develop efficiently while respecting Pathway's platform requirements! 🎉**
