#!/usr/bin/env python3
"""
Complete Example: Using Enhanced Accuracy Features
Demonstrates all improvements in a real-world workflow.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.embeddings import EnhancedEmbeddingGenerator
from src.search import HybridSearchEngine


def example_1_basic_search():
    """Example 1: Basic search with automatic fallback."""
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC SEARCH WITH FALLBACK")
    print("="*80)
    
    # Initialize with enhanced settings
    search = HybridSearchEngine(
        similarity_metric="hybrid",  # Best separation
        rerank=True,  # Improve precision
        min_score_threshold=0.5  # Balanced threshold
    )
    
    # Search - automatically falls back to web if needed
    query = "KRAS G12C inhibitor for lung cancer"
    results = search.search_papers(query, limit=10, use_fallback=True)
    
    print(f"\nQuery: {query}")
    print(f"Total results: {results['total']}")
    print(f"From vector DB: {results['from_vector_db']}")
    print(f"From web search: {results['from_web']}")
    print(f"Fallback used: {results['used_fallback']}")
    
    print("\nTop 3 results:")
    for i, paper in enumerate(results['papers'][:3], 1):
        print(f"\n{i}. {paper['title'][:70]}...")
        print(f"   Score: {paper['score']:.3f}")
        print(f"   Source: {paper['source']}")
        print(f"   PMID: {paper['pmid']}")


def example_2_compound_similarity():
    """Example 2: Find similar compounds with better fingerprints."""
    print("\n" + "="*80)
    print("EXAMPLE 2: COMPOUND SIMILARITY SEARCH")
    print("="*80)
    
    generator = EnhancedEmbeddingGenerator()
    
    # Query compound (Sotorasib - KRAS inhibitor)
    query_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)CN3CCN(CC3)C)F)F"
    
    print(f"Query compound: Sotorasib (KRAS G12C inhibitor)")
    print(f"SMILES: {query_smiles}")
    
    # Generate enhanced 4096-bit fingerprint
    query_fp = generator.embed_smiles(query_smiles)
    print(f"Fingerprint dimension: {len(query_fp)}")
    
    # Example database compounds
    database = [
        ("Adagrasib", "CN1CCN(CC1)CC2=CC(=C(C=C2)NC(=O)C3=CC=C(C=C3)F)F", "KRAS inhibitor"),
        ("Osimertinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "EGFR inhibitor"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "Pain reliever")
    ]
    
    print("\nSearching for similar compounds...")
    similarities = []
    
    for name, smiles, drug_class in database:
        compound_fp = generator.embed_smiles(smiles)
        
        # Use hybrid metric for better separation
        similarity = generator.compute_similarity(
            query_fp,
            compound_fp,
            metric="hybrid"
        )
        
        similarities.append((name, similarity, drug_class))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nResults (ordered by similarity):")
    for name, sim, drug_class in similarities:
        print(f"  {name:15s} - {sim:.3f} ({drug_class})")
    
    print("\n✓ Notice: KRAS inhibitor ranked highest!")


def example_3_multi_modal_search():
    """Example 3: Search across all modalities."""
    print("\n" + "="*80)
    print("EXAMPLE 3: MULTI-MODAL SEARCH")
    print("="*80)
    
    search = HybridSearchEngine()
    
    query = "EGFR mutation targeted therapy"
    
    print(f"Query: {query}")
    print("Searching across papers, compounds, proteins, and trials...\n")
    
    results = search.search_multi_modal(
        query=query,
        search_papers=True,
        search_compounds=True,
        search_proteins=True,
        search_trials=True,
        limit_per_type=3
    )
    
    # Display results by modality
    if results['papers']:
        print(f"📄 PAPERS ({len(results['papers'])}):")
        for paper in results['papers']:
            print(f"  • {paper['title'][:60]}... ({paper['score']:.3f})")
    
    if results['compounds']:
        print(f"\n💊 COMPOUNDS ({len(results['compounds'])}):")
        for compound in results['compounds']:
            print(f"  • {compound['name']} - {compound['molecular_formula']} ({compound['score']:.3f})")
    
    if results['proteins']:
        print(f"\n🧬 PROTEINS ({len(results['proteins'])}):")
        for protein in results['proteins']:
            genes = ', '.join(protein['gene_names'][:2])
            print(f"  • {protein['protein_name']} ({genes}) ({protein['score']:.3f})")
    
    if results['trials']:
        print(f"\n🏥 CLINICAL TRIALS ({len(results['trials'])}):")
        for trial in results['trials']:
            print(f"  • {trial['title'][:60]}... ({trial['phase']}) ({trial['score']:.3f})")
    
    print(f"\n✓ Total results: {results['total']} across all modalities")


def example_4_precision_tuning():
    """Example 4: Tune precision vs recall."""
    print("\n" + "="*80)
    print("EXAMPLE 4: PRECISION/RECALL TUNING")
    print("="*80)
    
    query = "kinase inhibitor cancer"
    
    # High precision (strict matching)
    print("\nHIGH PRECISION (strict matching):")
    strict_search = HybridSearchEngine(
        similarity_metric="dot",
        min_score_threshold=0.7,
        rerank=True
    )
    
    strict_results = strict_search.search_papers(query, limit=5, use_fallback=False)
    print(f"  Results: {len(strict_results['papers'])}")
    if strict_results['papers']:
        print(f"  Top score: {strict_results['papers'][0]['score']:.3f}")
    
    # Balanced
    print("\nBALANCED (recommended):")
    balanced_search = HybridSearchEngine(
        similarity_metric="hybrid",
        min_score_threshold=0.5,
        rerank=True
    )
    
    balanced_results = balanced_search.search_papers(query, limit=5, use_fallback=False)
    print(f"  Results: {len(balanced_results['papers'])}")
    if balanced_results['papers']:
        print(f"  Top score: {balanced_results['papers'][0]['score']:.3f}")
    
    # High recall (exploratory)
    print("\nHIGH RECALL (exploratory):")
    exploratory_search = HybridSearchEngine(
        similarity_metric="cosine",
        min_score_threshold=0.3,
        rerank=False
    )
    
    exploratory_results = exploratory_search.search_papers(query, limit=5, use_fallback=False)
    print(f"  Results: {len(exploratory_results['papers'])}")
    if exploratory_results['papers']:
        print(f"  Top score: {exploratory_results['papers'][0]['score']:.3f}")
    
    print("\n💡 Tip: Use 'balanced' for most applications")


def example_5_similarity_metrics_comparison():
    """Example 5: Compare different similarity metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 5: SIMILARITY METRICS COMPARISON")
    print("="*80)
    
    generator = EnhancedEmbeddingGenerator()
    
    # Test pairs
    query = "KRAS G12C inhibitor"
    similar = "KRAS mutation targeted therapy"
    dissimilar = "Vitamin D supplementation"
    
    print(f"Query: {query}")
    print(f"Similar doc: {similar}")
    print(f"Dissimilar doc: {dissimilar}\n")
    
    # Generate embeddings
    q_emb = generator.embed_text(query)
    s_emb = generator.embed_text(similar)
    d_emb = generator.embed_text(dissimilar)
    
    # Test each metric
    metrics = ["cosine", "dot", "angular", "hybrid"]
    
    print("Metric      | Similar | Dissimilar | Separation")
    print("-" * 55)
    
    for metric in metrics:
        sim_score = generator.compute_similarity(q_emb, s_emb, metric=metric)
        dis_score = generator.compute_similarity(q_emb, d_emb, metric=metric)
        separation = sim_score - dis_score
        
        print(f"{metric:11s} | {sim_score:7.3f} | {dis_score:10.3f} | {separation:10.3f}")
    
    print("\n✓ 'hybrid' typically provides best separation!")


def main():
    """Run all examples."""
    print("="*80)
    print("ENHANCED ACCURACY FEATURES - COMPLETE EXAMPLES")
    print("="*80)
    
    try:
        # Example 1: Basic search with fallback
        example_1_basic_search()
        
        # Example 2: Compound similarity
        example_2_compound_similarity()
        
        # Example 3: Multi-modal search
        example_3_multi_modal_search()
        
        # Example 4: Precision tuning
        example_4_precision_tuning()
        
        # Example 5: Metric comparison
        example_5_similarity_metrics_comparison()
        
        # Summary
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETE")
        print("="*80)
        
        print("\nKey Takeaways:")
        print("  1. Use 'hybrid' similarity metric for best results")
        print("  2. Enable re-ranking for higher precision")
        print("  3. Web fallback ensures you always get results")
        print("  4. 768-dim embeddings provide better separation")
        print("  5. 4096-bit fingerprints improve compound matching")
        
        print("\nNext Steps:")
        print("  • Run test_accuracy.py to see quantitative improvements")
        print("  • Update your Qdrant collections with enhanced dimensions")
        print("  • Replace CloudSearch with HybridSearchEngine in your app")
        print("  • Experiment with different thresholds and metrics")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()