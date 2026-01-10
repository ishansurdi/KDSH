# Installation Fix Guide

## Issue: Pathway Installation Error

If you see: `ERROR: No matching distribution found for pathway==0.8.0`

### ✅ Solution

**Option 1: Use Updated Requirements (Recommended)**
```bash
pip install -r requirements.txt
```
The requirements.txt has been updated to use `pathway` without version constraint.

**Option 2: Manual Installation**
```bash
# Install pathway directly
pip install pathway

# Then install remaining dependencies
pip install numpy pandas scikit-learn scipy jupyter notebook ipykernel ipywidgets streamlit plotly matplotlib tqdm python-dateutil
```

**Option 3: Without Pathway (Fallback)**
If pathway installation fails completely, you can use a simpler document store:

1. Comment out pathway in requirements.txt
2. Use the fallback implementation below

---

## 🔧 Fallback: Simple Document Store (No Pathway)

Create `core/simple_store.py`:

```python
"""Simple document store without Pathway dependency."""
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class SimpleDocumentStore:
    """Lightweight document store using basic Python."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def ingest_novel(self, novel_path: str, chunk_size: int = 1000) -> int:
        """Ingest novel from file."""
        from core.utils import chunk_text_semantic
        
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = chunk_text_semantic(text, chunk_size)
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                id=f"chunk_{i}",
                text=chunk,
                metadata={'chunk_id': i, 'file': novel_path}
            )
            self.documents.append(doc)
        
        return len(chunks)
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            doc_words = set(doc.text.lower().split())
            score = len(query_words & doc_words)
            scores.append((score, doc))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scores[:k]]
    
    def get_context_window(self, doc_id: str, window: int = 1) -> List[Document]:
        """Get surrounding chunks."""
        idx = next((i for i, d in enumerate(self.documents) if d.id == doc_id), None)
        if idx is None:
            return []
        
        start = max(0, idx - window)
        end = min(len(self.documents), idx + window + 1)
        return self.documents[start:end]

# Alias for compatibility
PathwayDocumentStore = SimpleDocumentStore
```

Then update `core/pathway_store.py`:

```python
"""Document store - uses Simple fallback if Pathway unavailable."""
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    print("⚠️ Pathway not available, using simple fallback")

if PATHWAY_AVAILABLE:
    # Original Pathway implementation
    from core.pathway_store import PathwayDocumentStore
else:
    # Use simple fallback
    from core.simple_store import SimpleDocumentStore as PathwayDocumentStore
```

---

## 🧪 Test Installation

After installation, verify everything works:

```bash
python -c "import pathway; print('Pathway:', pathway.__version__)"
python -c "import numpy, pandas, sklearn; print('✅ Core deps OK')"
python -c "from core.utils import chunk_text_semantic; print('✅ Core modules OK')"
```

---

## 📦 Minimal Installation (No Pathway)

If you want to proceed without Pathway at all:

```bash
pip install numpy pandas scikit-learn scipy jupyter streamlit matplotlib tqdm
```

Then use the SimpleDocumentStore fallback above.

---

## ✅ Verify Setup

```bash
python setup.py
```

This will check all dependencies and report any issues.

---

## 💡 Why This Happened

Pathway uses an unusual versioning scheme and may not be available on all PyPI mirrors. The system has been designed to work with or without it using fallback implementations.

---

## 🚀 Continue Setup

Once dependencies are installed:

1. **Test with dashboard**: `streamlit run dashboard/app.py`
2. **Try notebooks**: `jupyter notebook notebooks/01_ingestion.ipynb`
3. **Run pipeline**: `python main.py --help`

The system will work with either Pathway or the simple fallback!
