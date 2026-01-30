"""
Quick 3D Model Setup
Downloads comprehensive 3D biological model library.
"""

from pathlib import Path
from src.ingest import Fixed3DIngestor
import logging

logging.basicConfig(level=logging.INFO)


def main():
    print("="*80)
    print("🧬 3D BIOLOGICAL MODEL LIBRARY SETUP")
    print("="*80)
    
    ingestor = Fixed3DIngestor()
    
    # Download all categories
    stats = ingestor.ingest_all_categories(
        include_proteins=True,
        include_viruses=True
    )
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE")
    print("="*80)
    print(f"📊 Total 3D models: {stats['total']}")
    print("\n📁 Categories downloaded:")
    for category, count in stats['by_category'].items():
        if count > 0:
            print(f"   • {category}: {count} models")
    
    # Optional: Search for specific topics
    print("\n🔍 Bonus: Searching for specific topics...")
    custom_keywords = [
        "blood vessel",
        "neuron",
        "muscle fiber",
        "enzyme complex"
    ]
    
    custom_results = ingestor.ingest_by_keywords(custom_keywords, max_per_keyword=10)
    
    print(f"\n✅ Downloaded {sum(len(imgs) for imgs in custom_results.values())} additional models")
    print("\n🎉 3D model library ready!")


if __name__ == "__main__":
    main()