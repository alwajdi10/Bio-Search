"""
Hybrid Search Engine
Combines vector search (Qdrant) with web search fallback.
Always returns relevant results even if not in local database.
"""

import os
import logging
from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, SearchRequest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import EnhancedEmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Advanced search engine with multiple strategies:
    1. Vector search in Qdrant (primary)
    2. Re-ranking with multiple metrics
    3. Web search fallback (if results insufficient)
    4. Multi-vector matching
    """
    
    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        similarity_metric: str = "hybrid",
        rerank: bool = True,
        min_score_threshold: float = 0.5
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            qdrant_url: Qdrant Cloud URL
            qdrant_api_key: API key
            similarity_metric: Metric for scoring
            rerank: Whether to re-rank results
            min_score_threshold: Minimum similarity score
        """
        # Get credentials
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("Qdrant credentials required")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60
        )
        
        # Initialize enhanced embeddings
        self.generator = EnhancedEmbeddingGenerator(
            text_model="sentence-transformers/all-mpnet-base-v2",  # 768-dim
            normalize=True
        )
        
        # Search parameters
        self.similarity_metric = similarity_metric
        self.rerank = rerank
        self.min_score_threshold = min_score_threshold
        
        # Collection names
        self.collection_papers = "research_papers"
        self.collection_compounds = "chemical_compounds"
        self.collection_proteins = "proteins"
        self.collection_trials = "clinical_trials"
        
        logger.info("✓ Hybrid search engine initialized")
        logger.info(f"  Similarity metric: {similarity_metric}")
        logger.info(f"  Re-ranking: {rerank}")
        logger.info(f"  Min score threshold: {min_score_threshold}")
    
    def search_papers(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        use_fallback: bool = True
    ) -> Dict:
        """
        Search papers with fallback to web search.
        
        Args:
            query: Search query
            limit: Number of results
            filters: Optional filters (date, compounds, etc)
            use_fallback: Use web search if insufficient results
            
        Returns:
            Dict with papers, metadata, and fallback indicator
        """
        logger.info(f"Searching papers: '{query}'")
        
        # Generate query embedding
        query_vector = self.generator.embed_text(query)
        
        # Search Qdrant
        try:
            results = self.client.search(
                collection_name=self.collection_papers,
                query_vector=query_vector.tolist(),
                limit=limit * 2,  # Get more for re-ranking
                query_filter=self._build_filter(filters) if filters else None,
                score_threshold=self.min_score_threshold
            )
            
            logger.info(f"Found {len(results)} papers in vector DB")
            
            # Parse results
            papers = []
            for hit in results:
                paper = {
                    "score": float(hit.score),
                    "pmid": hit.payload["pmid"],
                    "title": hit.payload["title"],
                    "abstract": hit.payload["abstract"],
                    "authors": hit.payload.get("authors", []),
                    "journal": hit.payload.get("journal", ""),
                    "publication_date": hit.payload.get("publication_date", ""),
                    "mentioned_compounds": hit.payload.get("mentioned_compounds", []),
                    "mesh_terms": hit.payload.get("mesh_terms", []),
                    "source": "vector_db"
                }
                papers.append(paper)
            
            # Re-rank if enabled
            if self.rerank and len(papers) > 0:
                papers = self._rerank_papers(query, papers, query_vector)
            
            # Apply score threshold again after re-ranking
            papers = [p for p in papers if p["score"] >= self.min_score_threshold]
            
            # Limit to requested number
            papers = papers[:limit]
            
            # Check if we need fallback
            needs_fallback = len(papers) < max(3, limit // 2)
            
            if needs_fallback and use_fallback:
                logger.info("Insufficient results, using web search fallback...")
                web_papers = self._web_search_fallback(query, limit - len(papers))
                papers.extend(web_papers)
            
            return {
                "query": query,
                "papers": papers,
                "total": len(papers),
                "from_vector_db": sum(1 for p in papers if p["source"] == "vector_db"),
                "from_web": sum(1 for p in papers if p["source"] == "web"),
                "used_fallback": needs_fallback and use_fallback
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            
            # Full fallback to web search
            if use_fallback:
                logger.info("Vector search failed, using full web fallback...")
                web_papers = self._web_search_fallback(query, limit)
                return {
                    "query": query,
                    "papers": web_papers,
                    "total": len(web_papers),
                    "from_vector_db": 0,
                    "from_web": len(web_papers),
                    "used_fallback": True,
                    "error": str(e)
                }
            else:
                return {
                    "query": query,
                    "papers": [],
                    "total": 0,
                    "error": str(e)
                }
    
    def _rerank_papers(
        self,
        query: str,
        papers: List[Dict],
        query_vector: any
    ) -> List[Dict]:
        """
        Re-rank papers using enhanced similarity metrics.
        
        Args:
            query: Original query
            papers: List of papers from vector search
            query_vector: Query embedding
            
        Returns:
            Re-ranked papers
        """
        logger.info(f"Re-ranking {len(papers)} papers with {self.similarity_metric} metric...")
        
        # Generate embeddings for each paper
        for paper in papers:
            # Create paper embedding
            paper_text = f"{paper['title']}. {paper['abstract']}"
            paper_vector = self.generator.embed_text(paper_text)
            
            # Compute similarity with chosen metric
            new_score = self.generator.compute_similarity(
                query_vector,
                paper_vector,
                metric=self.similarity_metric
            )
            
            # Store both original and new score
            paper["original_score"] = paper["score"]
            paper["score"] = float(new_score)
        
        # Sort by new score
        papers.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info("✓ Re-ranking complete")
        return papers
    
    def _web_search_fallback(
        self,
        query: str,
        limit: int
    ) -> List[Dict]:
        """
        Fallback to web search when vector DB has insufficient results.
        
        Args:
            query: Search query
            limit: Number of results needed
            
        Returns:
            List of papers from web search
        """
        try:
            # Use PubMed E-utilities for web search
            from src.pubmed_ingestor import PubMedIngestor
            
            ingestor = PubMedIngestor()
            papers = ingestor.search_and_fetch(query, max_results=limit)
            
            # Convert to standard format
            web_papers = []
            for paper in papers:
                web_papers.append({
                    "score": 0.8,  # Default score for web results
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "journal": paper.journal,
                    "publication_date": paper.publication_date,
                    "mentioned_compounds": paper.mentioned_compounds,
                    "mesh_terms": paper.mesh_terms,
                    "source": "web"
                })
            
            logger.info(f"✓ Retrieved {len(web_papers)} papers from web")
            return web_papers
            
        except Exception as e:
            logger.error(f"Web search fallback failed: {e}")
            return []
    
    def _build_filter(self, filters: Dict) -> Filter:
        """Build Qdrant filter from dict."""
        conditions = []
        
        if "publication_date" in filters:
            date = filters["publication_date"]
            if isinstance(date, dict):
                from qdrant_client.models import Range
                conditions.append(
                    FieldCondition(
                        key="publication_date",
                        range=Range(**date)
                    )
                )
        
        if "mentioned_compounds" in filters:
            compounds = filters["mentioned_compounds"]
            conditions.append(
                FieldCondition(
                    key="mentioned_compounds",
                    match={"any": compounds}
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def search_multi_modal(
        self,
        query: str,
        search_papers: bool = True,
        search_compounds: bool = True,
        search_proteins: bool = True,
        search_trials: bool = True,
        limit_per_type: int = 5
    ) -> Dict:
        """
        Search across all modalities simultaneously.
        
        Args:
            query: Search query
            search_*: Which modalities to search
            limit_per_type: Results per modality
            
        Returns:
            Dict with results from all modalities
        """
        logger.info(f"Multi-modal search: '{query}'")
        
        results = {
            "query": query,
            "papers": [],
            "compounds": [],
            "proteins": [],
            "trials": []
        }
        
        # Search papers
        if search_papers:
            paper_results = self.search_papers(query, limit=limit_per_type, use_fallback=True)
            results["papers"] = paper_results["papers"]
        
        # Search compounds (by name/description)
        if search_compounds:
            try:
                query_vector = self.generator.embed_text(query)
                hits = self.client.search(
                    collection_name=self.collection_compounds,
                    query_vector=query_vector.tolist(),
                    limit=limit_per_type
                )
                
                results["compounds"] = [
                    {
                        "score": float(hit.score),
                        "cid": hit.payload["cid"],
                        "name": hit.payload["name"],
                        "smiles": hit.payload.get("smiles", ""),
                        "molecular_formula": hit.payload.get("molecular_formula", ""),
                        "source": "vector_db"
                    }
                    for hit in hits
                ]
            except Exception as e:
                logger.warning(f"Compound search failed: {e}")
        
        # Search proteins
        if search_proteins:
            try:
                query_vector = self.generator.embed_text(query)
                hits = self.client.search(
                    collection_name=self.collection_proteins,
                    query_vector=query_vector.tolist(),
                    limit=limit_per_type
                )
                
                results["proteins"] = [
                    {
                        "score": float(hit.score),
                        "uniprot_id": hit.payload["uniprot_id"],
                        "protein_name": hit.payload["protein_name"],
                        "gene_names": hit.payload.get("gene_names", []),
                        "function": hit.payload.get("function", ""),
                        "source": "vector_db"
                    }
                    for hit in hits
                ]
            except Exception as e:
                logger.warning(f"Protein search failed: {e}")
        
        # Search trials
        if search_trials:
            try:
                query_vector = self.generator.embed_text(query)
                hits = self.client.search(
                    collection_name=self.collection_trials,
                    query_vector=query_vector.tolist(),
                    limit=limit_per_type
                )
                
                results["trials"] = [
                    {
                        "score": float(hit.score),
                        "nct_id": hit.payload["nct_id"],
                        "title": hit.payload["title"],
                        "status": hit.payload.get("status", ""),
                        "phase": hit.payload.get("phase", ""),
                        "conditions": hit.payload.get("conditions", []),
                        "source": "vector_db"
                    }
                    for hit in hits
                ]
            except Exception as e:
                logger.warning(f"Trial search failed: {e}")
        
        # Calculate totals
        results["total"] = sum([
            len(results["papers"]),
            len(results["compounds"]),
            len(results["proteins"]),
            len(results["trials"])
        ])
        
        logger.info(f"✓ Multi-modal search complete: {results['total']} total results")
        return results


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("HYBRID SEARCH ENGINE TEST")
    print("="*80)
    
    # Initialize
    print("\n1. Initializing hybrid search...")
    search = HybridSearchEngine(
        similarity_metric="hybrid",
        rerank=True,
        min_score_threshold=0.5
    )
    
    # Test search with fallback
    print("\n2. Testing search with web fallback...")
    results = search.search_papers(
        "KRAS G12C inhibitor lung cancer",
        limit=10,
        use_fallback=True
    )
    
    print(f"\n✓ Found {results['total']} papers:")
    print(f"  From vector DB: {results['from_vector_db']}")
    print(f"  From web: {results['from_web']}")
    print(f"  Used fallback: {results['used_fallback']}")
    
    for i, paper in enumerate(results['papers'][:3], 1):
        print(f"\n{i}. {paper['title'][:80]}...")
        print(f"   Score: {paper['score']:.3f}")
        print(f"   Source: {paper['source']}")
        print(f"   PMID: {paper['pmid']}")
    
    # Test multi-modal search
    print("\n3. Testing multi-modal search...")
    multi_results = search.search_multi_modal(
        "KRAS mutation treatment",
        limit_per_type=3
    )
    
    print(f"\n✓ Multi-modal results:")
    print(f"  Papers: {len(multi_results['papers'])}")
    print(f"  Compounds: {len(multi_results['compounds'])}")
    print(f"  Proteins: {len(multi_results['proteins'])}")
    print(f"  Trials: {len(multi_results['trials'])}")
    print(f"  Total: {multi_results['total']}")