"""
Download novel files for testing.
Run this in Google Colab to get the required novels.
"""

import os
from pathlib import Path
import requests

# Create novels directory
os.makedirs('data/novels', exist_ok=True)

# Project Gutenberg URLs for public domain novels
NOVELS = {
    'The Count of Monte Cristo.txt': 'https://www.gutenberg.org/cache/epub/1184/pg1184.txt',
    'In Search of the Castaways.txt': 'https://www.gutenberg.org/cache/epub/18857/pg18857.txt'
}

print("Downloading novels from Project Gutenberg...\n")

for filename, url in NOVELS.items():
    output_path = f'data/novels/{filename}'
    
    if Path(output_path).exists():
        print(f"✓ {filename} already exists, skipping")
        continue
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        file_size = len(response.text)
        print(f"✓ Downloaded {filename} ({file_size:,} characters)\n")
        
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}\n")

print("\n" + "="*60)
print("Download complete!")
print("="*60)

# List downloaded files
print("\nNovels in data/novels/:")
for f in Path('data/novels').glob('*.txt'):
    size = f.stat().st_size
    print(f"  - {f.name} ({size:,} bytes)")
