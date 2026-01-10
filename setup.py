"""
Setup and Installation Script for KDSH System

Run this script after installing requirements to verify setup.
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} detected. Need Python 3.8+")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    
    required = [
        'pathway', 'numpy', 'pandas', 'sklearn', 'scipy',
        'streamlit', 'jupyter', 'matplotlib', 'tqdm'
    ]
    
    missing = []
    
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    dirs = [
        'data/novels',
        'results'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_path}/")
    
    return True


def test_imports():
    """Test core module imports"""
    print("\nTesting core module imports...")
    
    try:
        from core import (
            PathwayDocumentStore, HierarchicalNarrativeMemory,
            ClaimExtractor, ConstraintBuilder, MultiHopRetriever,
            CausalReasoningEngine, TemporalReasoningEngine,
            InconsistencyScorer, ConsistencyClassifier
        )
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def create_sample_data():
    """Create sample train/test files if they don't exist"""
    print("\nCreating sample data files...")
    
    import pandas as pd
    
    # Sample train data
    if not os.path.exists('data/train.csv'):
        train_data = {
            'story_id': ['sample_001', 'sample_002'],
            'novel_file': ['sample_novel.txt', 'sample_novel.txt'],
            'backstory': [
                'The protagonist lived in London before the story began.',
                'The main character discovered a secret in 1850.'
            ],
            'label': [1, 0]
        }
        pd.DataFrame(train_data).to_csv('data/train.csv', index=False)
        print("✓ Created data/train.csv")
    else:
        print("✓ data/train.csv exists")
    
    # Sample test data
    if not os.path.exists('data/test.csv'):
        test_data = {
            'story_id': ['test_001'],
            'novel_file': ['sample_novel.txt'],
            'backstory': ['The hero fought in a battle before arriving.']
        }
        pd.DataFrame(test_data).to_csv('data/test.csv', index=False)
        print("✓ Created data/test.csv")
    else:
        print("✓ data/test.csv exists")
    
    return True


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("\n1. Run Jupyter notebooks:")
    print("   jupyter notebook")
    print("   → Navigate to notebooks/ and run 01_ingestion.ipynb")
    print("\n2. Or run Streamlit dashboard:")
    print("   streamlit run dashboard/app.py")
    print("\n3. Or run full pipeline:")
    print("   jupyter notebook notebooks/run_pipeline.ipynb")
    print("\n" + "=" * 60)


def main():
    """Main setup routine"""
    print("=" * 60)
    print("KDSH - Long-Context Narrative Consistency System")
    print("Setup and Verification Script")
    print("=" * 60)
    
    success = True
    
    # Run checks
    success &= check_python_version()
    success &= check_dependencies()
    success &= create_directories()
    success &= test_imports()
    success &= create_sample_data()
    
    if success:
        print_next_steps()
    else:
        print("\n✗ Setup incomplete. Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
