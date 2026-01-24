"""
Cloud Search Engine
Search using Qdrant Cloud.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from .embeddings import EmbeddingGenerator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudSearch:
    """Search engine using Qdrant Cloud."""
    
    def __init__(self):
        """Initialize cloud search."""
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if not url or not api_key:
            raise ValueError("Missing Qdrant Cloud credentials in .env")
        
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.generator = EmbeddingGenerator()
        self.collection_papers = "research_papers"
        self.collection_compounds = "chemical_compounds"
        
        logger.info("Connected to Qdrant Cloud")
    
    def search_by_text(self, query: str, limit: int = 10) -> Dict:
        """Search papers by text query."""
        logger.info(f"Searching: {query}")
        
        # Generate embedding
        query_vector = self.generator.embed_text(query)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_papers,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        papers = []
        compound_names = set()
        
        for hit in results:
            paper = {
                "score": hit.score,
                "pmid": hit.payload["pmid"],
                "title": hit.payload["title"],
                "abstract": hit.payload["abstract"],
                "journal": hit.payload.get("journal", ""),
                "publication_date": hit.payload.get("publication_date", ""),
                "mentioned_compounds": hit.payload.get("mentioned_compounds", [])
            }
            papers.append(paper)
            compound_names.update(hit.payload.get("mentioned_compounds", []))
        
        # Get compound details
        compounds = []
        for name in list(compound_names)[:10]:
            try:
                results = self.client.scroll(
                    collection_name=self.collection_compounds,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="name",
                                match={"value": name}
                            )
                        ]
                    ),
                    limit=1
                )
                if results[0]:
                    hit = results[0][0]
                    compounds.append({
                        "cid": hit.payload["cid"],
                        "name": hit.payload["name"],
                        "smiles": hit.payload.get("smiles", ""),
                        "molecular_formula": hit.payload.get("molecular_formula", "")
                    })
            except:
                pass
        
        return {
            "query": query,
            "papers": papers,
            "compounds": compounds
        }
    
    def search_by_smiles(self, smiles: str, limit: int = 10) -> Dict:
        """Search compounds by structure."""
        logger.info(f"Searching SMILES: {smiles}")
        
        # Generate fingerprint
        query_vector = self.generator.embed_smiles(smiles)
        
        if query_vector is None:
            return {"error": "Invalid SMILES"}
        
        # Search
        results = self.client.search(
            collection_name=self.collection_compounds,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        compounds = []
        pmids = set()
        
        for hit in results:
            compound = {
                "score": hit.score,
                "cid": hit.payload["cid"],
                "name": hit.payload["name"],
                "smiles": hit.payload.get("smiles", ""),
                "molecular_formula": hit.payload.get("molecular_formula", ""),
                "source_pmids": hit.payload.get("source_pmids", [])
            }
            compounds.append(compound)
            pmids.update(hit.payload.get("source_pmids", []))
        
        # Get papers
        papers = []
        for pmid in list(pmids)[:10]:
            try:
                results = self.client.scroll(
                    collection_name=self.collection_papers,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="pmid",
                                match={"value": pmid}
                            )
                        ]
                    ),
                    limit=1
                )
                if results[0]:
                    hit = results[0][0]
                    papers.append({
                        "pmid": hit.payload["pmid"],
                        "title": hit.payload["title"],
                        "abstract": hit.payload.get("abstract", "")[:200] + "..."
                    })
            except:
                pass
        
        return {
            "query": smiles,
            "compounds": compounds,
            "papers": papers
        }


if __name__ == "__main__":
    search = CloudSearch()
    
    # Test
    results = search.search_by_text("KRAS inhibitor", limit=5)
    print(f"\nFound {len(results['papers'])} papers")
    for p in results['papers']:
        print(f"- {p['title']}")