# Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Installation

```bash
# 1. Navigate to project directory
cd KDSH

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify setup
python setup.py
```

## 📖 Option 1: Interactive Dashboard (Easiest)

```bash
# Start dashboard
streamlit run dashboard/app.py

# Opens browser at http://localhost:8501
```

Then:
1. **Upload a novel** (Tab 1) - Upload your .txt file or use sample
2. **Enter backstory** (Tab 2) - Type hypothetical backstory
3. **Run analysis** - Click "Run Analysis" button
4. **View results** (Tab 3) - See detailed breakdown

## 📓 Option 2: Jupyter Notebooks (Learning)

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
```

Run notebooks in order:
1. `01_ingestion.ipynb` - Load novel
2. `02_memory.ipynb` - Build memory
3. `03_claims_constraints.ipynb` - Extract claims
4. `04_retrieval.ipynb` - Retrieve evidence
5. `05_reasoning.ipynb` - Apply reasoning
6. `06_scoring.ipynb` - Score inconsistencies
7. `07_classifier.ipynb` - Final classification
8. `08_evaluation.ipynb` - Evaluate system
9. `run_pipeline.ipynb` - **Complete pipeline** ⭐

## 🎯 Option 3: Python Script (Production)

```python
from core import *

# Setup
store = PathwayDocumentStore()

# Load novel
with open('data/novels/my_novel.txt', 'r') as f:
    novel = f.read()

store.ingest_novel(novel, 'my_novel')

# Process backstory
backstory = "Character did X before Y happened"
# ... (see run_pipeline.ipynb for complete code)
```

## 📁 Data Setup

Place your data files:

```
data/
├── novels/
│   ├── novel1.txt      # Your novel files
│   └── novel2.txt
├── train.csv           # Training examples
└── test.csv            # Test examples
```

**CSV Format:**
```csv
story_id,novel_file,backstory,label
001,novel1.txt,"Backstory text here",1
```

## ✅ Verify It Works

### Test 1: Run Sample
```bash
# Open run_pipeline.ipynb
jupyter notebook notebooks/run_pipeline.ipynb

# Run all cells (Cell → Run All)
```

### Test 2: Dashboard
```bash
streamlit run dashboard/app.py

# Upload sample novel
# Enter test backstory
# Click "Run Analysis"
```

## 🔧 Common Issues

### Import Errors
```bash
# Make sure you're in virtual environment
pip install -r requirements.txt

# Verify imports
python setup.py
```

### Missing Data
```bash
# Create sample data
python setup.py
```

### Streamlit Issues
```bash
# Upgrade streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

## 📚 Example Workflow

1. **Prepare data:**
   - Place novel in `data/novels/moby_dick.txt`
   - Create backstory: "Ishmael sailed with Ahab before"

2. **Run pipeline:**
   ```bash
   jupyter notebook notebooks/run_pipeline.ipynb
   ```

3. **Get results:**
   - Check `results/results.csv`
   - Review rationales
   - Analyze conflicts

## 🎓 Learning Path

**Day 1:** Understand the system
- Read [README.md](README.md)
- Run notebooks 01-03
- Understand claim extraction

**Day 2:** Deep dive into reasoning
- Run notebooks 04-05
- Study causal/temporal engines
- Explore evidence retrieval

**Day 3:** Production pipeline
- Run `run_pipeline.ipynb`
- Test with your own novels
- Experiment with parameters

**Day 4:** Dashboard and deployment
- Launch Streamlit dashboard
- Test interactive features
- Customize for your needs

## 💡 Tips

1. **Start small:** Use short novels (10k words) first
2. **Use dashboard:** Great for quick tests
3. **Read notebooks:** They explain each step
4. **Check outputs:** Inspect intermediate results
5. **Tune parameters:** Adjust thresholds in config

## 🏆 Production Checklist

- [ ] Install all dependencies
- [ ] Verify setup with `python setup.py`
- [ ] Place novels in `data/novels/`
- [ ] Create train.csv and test.csv
- [ ] Run notebooks 01-08
- [ ] Execute run_pipeline.ipynb
- [ ] Check results/results.csv
- [ ] Review rationales for quality
- [ ] Calibrate thresholds on train set
- [ ] Generate final predictions

## 🆘 Need Help?

1. Check error messages carefully
2. Review notebook outputs
3. Verify data format
4. Check module imports
5. Review README.md for details

## 🎯 Ready to Win!

You now have:
- ✅ Complete reasoning system
- ✅ Interactive dashboard
- ✅ Modular notebooks
- ✅ Production pipeline
- ✅ Research-backed methods

**Go win that competition! 🚀**
