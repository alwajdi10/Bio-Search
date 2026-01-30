"""
Quick Image Setup Script
Ingests comprehensive image dataset and indexes with CLIP.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.enh_img import EnhancedImageIngestor
from src.image_embeddings import EnhancedImageSearchEngine


def main():
    print("="*80)
    print("🖼️  BIOLOGICAL IMAGE LIBRARY SETUP")
    print("="*80)
    
    # Step 1: Ingest images
    print("\n📥 Step 1: Ingesting images from multiple sources...")
    ingestor = EnhancedImageIngestor()
    
    stats = ingestor.ingest_comprehensive_dataset(
        data_dir=Path("data/raw"),
        max_compounds=200,  # 200 compound structures
        max_proteins=100,   # 100 protein structures
        common_pathways=True  # 10 pathway diagrams
    )
    
    # Step 2: Index images
    print("\n🔍 Step 2: Indexing images with CLIP embeddings...")
    search_engine = EnhancedImageSearchEngine()
    
    stats_search = search_engine.get_statistics()
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE")
    print("="*80)
    print(f"📊 Image Library Statistics:")
    print(f"   Total images: {stats_search['total']}")
    print(f"   By type: {stats_search['by_type']}")
    print(f"   Embeddings: {'✓' if stats_search['has_embeddings'] else '✗'}")
    
    # Step 3: Test search
    print("\n🧪 Testing search...")
    test_queries = [
        "chemical compound structure",
        "protein 3d model",
        "cancer signaling pathway"
    ]
    
    for query in test_queries:
        results = search_engine.search_by_text(query, top_k=3)
        print(f"\n  Query: '{query}'")
        print(f"  Results: {len(results)}")
        for r in results[:2]:
            print(f"    - {r['image'].caption} (score: {r['score']:.3f})")
    
    print("\n" + "="*80)
    print("🎉 Image library ready!")
    print("="*80)


if __name__ == "__main__":
    main()