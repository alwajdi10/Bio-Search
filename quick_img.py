#!/usr/bin/env python3
"""
Image Features Quickstart
Demonstrates all image capabilities of the platform.

Usage:
    python quickstart_images.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.image_manager import ImageManager
from src.image_embeddings import ImageEmbeddingGenerator, ImageSearchEngine


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_compound_structures():
    """Demo: Download and manage 2D compound structures."""
    print_header("1️⃣  COMPOUND 2D STRUCTURES")
    
    manager = ImageManager()
    
    # Popular drug structures
    compounds = [
        (2244, "Aspirin"),
        (5311, "Ibuprofen"),
        (60823, "Osimertinib - EGFR inhibitor"),
        (11526198, "Gefitinib - EGFR inhibitor"),
        (135565912, "Sotorasib - KRAS inhibitor")
    ]
    
    print("Downloading compound structures...")
    cids = [c[0] for c in compounds]
    images = manager.batch_download_compound_structures(cids, size="large")
    
    print(f"\n✅ Downloaded {len(images)} structures:\n")
    for (cid, name), img in zip(compounds, images):
        print(f"  ✓ {name}")
        print(f"    CID: {cid}")
        print(f"    Size: {img.width}x{img.height}")
        print(f"    Path: {img.local_path}")
        print(f"    URL: {img.url}\n")
    
    # Create comparison grid
    print("Creating comparison grid...")
    grid_path = manager.create_comparison_grid(images, cols=3)
    print(f"✓ Grid saved to: {grid_path}")
    
    return images


def demo_protein_structures():
    """Demo: Download 3D protein structures."""
    print_header("2️⃣  PROTEIN 3D STRUCTURES")
    
    manager = ImageManager()
    
    # Important cancer-related proteins
    proteins = [
        ("6XM0", "KRAS G12C mutant"),
        ("1M17", "p53 tumor suppressor"),
        ("7BV2", "EGFR kinase domain")
    ]
    
    print("Downloading protein structures...")
    pdb_ids = [p[0] for p in proteins]
    images = manager.batch_download_protein_structures(pdb_ids)
    
    print(f"\n✅ Downloaded {len(images)} structures:\n")
    for (pdb_id, name), img in zip(proteins, images):
        print(f"  ✓ {name}")
        print(f"    PDB ID: {pdb_id}")
        print(f"    Path: {img.local_path}")
        print(f"    View 3D: https://www.rcsb.org/3d-view/{pdb_id}\n")
    
    return images


def demo_pathway_diagrams():
    """Demo: Download pathway diagrams."""
    print_header("3️⃣  PATHWAY DIAGRAMS")
    
    manager = ImageManager()
    
    # Key cancer pathways
    pathways = [
        ("04010", "MAPK signaling pathway"),
        ("04110", "Cell cycle"),
        ("04151", "PI3K-Akt signaling pathway")
    ]
    
    print("Downloading pathway diagrams...")
    images = []
    
    for pathway_id, name in pathways:
        img = manager.get_pathway_diagram(pathway_id, organism="hsa")
        if img:
            images.append(img)
            print(f"  ✓ {name}")
            print(f"    Pathway: hsa{pathway_id}")
            print(f"    Path: {img.local_path}\n")
        else:
            print(f"  ✗ Failed to download {name}\n")
    
    return images


def demo_visual_search():
    """Demo: CLIP-powered visual search."""
    print_header("4️⃣  VISUAL SEARCH WITH CLIP")
    
    # Check if transformers is available
    try:
        import transformers
    except ImportError:
        print("⚠️  transformers not installed")
        print("Install with: pip install transformers")
        print("Skipping visual search demo...")
        return False
    
    try:
        print("Initializing CLIP model...")
        clip = ImageEmbeddingGenerator()
        print(f"✓ CLIP loaded (embedding dim: {clip.embedding_dim})\n")
        
        # Initialize search engine
        print("Building image index...")
        search = ImageSearchEngine(use_clip=True)
        print(f"✓ Indexed {len(search.images)} images\n")
        
        # Text-to-image search
        print("─" * 80)
        print("TEXT-TO-IMAGE SEARCH")
        print("─" * 80 + "\n")
        
        queries = [
            "chemical compound structure",
            "protein 3D model",
            "biological pathway diagram"
        ]
        
        for query in queries:
            print(f"Query: '{query}'")
            results = search.search_by_text(query, top_k=3)
            
            print("Results:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['image'].caption}")
                print(f"     Similarity: {r['score']:.3f}")
                print(f"     Type: {r['image'].image_type}\n")
        
        # Image-to-image search
        if search.images:
            print("─" * 80)
            print("IMAGE-TO-IMAGE SEARCH")
            print("─" * 80 + "\n")
            
            query_img = search.images[0]
            print(f"Query image: {query_img.caption}")
            
            similar = search.search_similar(query_img.local_path, top_k=5)
            
            print("Similar images:")
            for i, r in enumerate(similar, 1):
                print(f"  {i}. {r['image'].caption}")
                print(f"     Similarity: {r['score']:.3f}\n")
        
        return True
        
    except ImportError:
        print("⚠️  transformers not installed")
        print("Install with: pip install transformers")
        print("Skipping visual search demo...")
        return False


def demo_image_utilities():
    """Demo: Image manipulation utilities."""
    print_header("5️⃣  IMAGE UTILITIES")
    
    manager = ImageManager()
    
    # Get a sample image
    img = manager.get_compound_2d_structure(2244, size="large")  # Aspirin
    
    if img:
        print("Original image:")
        print(f"  Size: {img.width}x{img.height}")
        print(f"  Path: {img.local_path}\n")
        
        # Resize
        print("Resizing to 400x400...")
        resized = manager.resize_image(img, max_size=(400, 400))
        print(f"✓ Resized: {resized.width}x{resized.height}")
        print(f"  Path: {resized.local_path}\n")
        
        # Convert to base64
        print("Converting to base64...")
        b64 = manager.get_image_as_base64(img)
        print(f"✓ Base64: {b64[:50]}... ({len(b64)} chars)\n")
        
        # Save metadata
        print("Saving metadata...")
        all_images = manager.get_all_cached_images()
        manager.save_metadata(all_images)
        print(f"✓ Saved metadata for {len(all_images)} images")


def demo_statistics():
    """Display statistics."""
    print_header("📊 STATISTICS")
    
    manager = ImageManager()
    images = manager.get_all_cached_images()
    
    if images:
        from collections import Counter
        
        print(f"Total images: {len(images)}\n")
        
        # By type
        type_counts = Counter(img.image_type for img in images)
        print("By type:")
        for img_type, count in type_counts.items():
            print(f"  {img_type}: {count}")
        
        # By source
        source_counts = Counter(img.source for img in images)
        print("\nBy source:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        # Storage
        total_size = sum(
            Path(img.local_path).stat().st_size 
            for img in images 
            if img.local_path and Path(img.local_path).exists()
        )
        print(f"\nTotal storage: {total_size / 1024 / 1024:.1f} MB")
        
        print(f"\nCache directory: {manager.cache_dir}")
    else:
        print("No images in cache yet.")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  🖼️  IMAGE FEATURES QUICKSTART")
    print("  Multi-Modal Visual Search for Biological Research")
    print("="*80)
    
    try:
        # Demo 1: Compound structures
        compound_images = demo_compound_structures()
        
        # Demo 2: Protein structures
        protein_images = demo_protein_structures()
        
        # Demo 3: Pathway diagrams
        pathway_images = demo_pathway_diagrams()
        
        # Demo 4: Visual search (requires CLIP)
        clip_available = demo_visual_search()
        
        # Demo 5: Utilities
        demo_image_utilities()
        
        # Statistics
        demo_statistics()
        
        # Summary
        print_header("✅ QUICKSTART COMPLETE")
        
        print("What you can do now:")
        print("  1. View downloaded images in data/images/")
        print("  2. Run the enhanced app: streamlit run app_with_images.py")
        print("  3. Use visual search in your research")
        
        if clip_available:
            print("\n🎨 CLIP visual search is enabled!")
            print("  - Search images by text description")
            print("  - Find similar structures")
            print("  - Multi-modal queries")
        else:
            print("\n💡 To enable CLIP visual search:")
            print("  pip install transformers")
        
        print("\n📚 See IMAGE_GUIDE.md for detailed documentation")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()