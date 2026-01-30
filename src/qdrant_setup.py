"""
Enhanced Qdrant Configuration
Supports higher-dimensional embeddings for better accuracy.
"""

import os
import json
from pathlib import Path
from typing import List
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings import EnhancedEmbeddingGenerator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQdrantConfig:
    """
    Enhanced Qdrant setup with higher-dimensional embeddings.
    """
    
    # Enhanced collection configurations
    COLLECTIONS = {
        "research_papers": {
            "dim": 768,
            "type": "text",
            "distance": Distance.DOT,
            "description": "Research papers with high-accuracy embeddings"
        },
        "chemical_compounds": {
            "dim": 4096,
            "type": "fingerprint",
            "distance": Distance.DOT,
            "description": "Chemical compounds with enhanced fingerprints"
        },
        "proteins": {
            "dim": 768,
            "type": "text",
            "distance": Distance.DOT,
            "description": "Protein data with enhanced embeddings"
        },
        "clinical_trials": {
            "dim": 768,
            "type": "text",
            "distance": Distance.DOT,
            "description": "Clinical trials with enhanced embeddings"
        },
        "bio_images": {
            "dim": 512,
            "type": "image",
            "distance": Distance.DOT,
            "description": "Biological images with CLIP embeddings"
        }
    }
    
    def __init__(self):
        """Initialize enhanced Qdrant manager."""
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if not url or not api_key:
            raise ValueError("Qdrant credentials required in .env")
        
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.generator = EnhancedEmbeddingGenerator(normalize=True)
        
        logger.info(f"✓ Connected to Qdrant Cloud: {url}")
        logger.info(f"✓ Using enhanced embeddings:")
        logger.info(f"  Text: {self.generator.text_dim}-dim")
        logger.info(f"  Fingerprints: {self.generator.fp_nbits}-bit")
    
    def create_all_collections(self, recreate: bool = False):
        """Create all enhanced collections."""
        logger.info("Setting up enhanced collections...")
        
        for name, config in self.COLLECTIONS.items():
            if recreate:
                try:
                    self.client.delete_collection(name)
                    logger.info(f"  Deleted old: {name}")
                except:
                    pass
            
            try:
                self.client.get_collection(name)
                logger.info(f"  ✓ Exists: {name} ({config['dim']}-dim)")
            except:
                logger.info(f"  Creating: {name} ({config['dim']}-dim)")
                
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=config["dim"],
                        distance=config["distance"]
                    )
                )
                
                logger.info(f"  ✓ Created: {name}")
        
        logger.info("✅ All enhanced collections ready")
    
    def populate_all(self, data_dir: Path):
        """Populate all collections from data directory."""
        logger.info(f"Populating from: {data_dir}")
        
        # Papers
        paper_files = list(data_dir.glob("*_papers.json"))
        if paper_files:
            all_papers = []
            for pf in paper_files:
                with open(pf) as f:
                    all_papers.extend(json.load(f))
            
            if all_papers:
                merged = data_dir / "merged_papers.json"
                with open(merged, 'w') as f:
                    json.dump(all_papers, f)
                self.populate_papers_enhanced(merged)
        
        # Compounds
        compound_files = list(data_dir.glob("*_compounds.json"))
        if compound_files:
            compounds_by_cid = {}
            for cf in compound_files:
                with open(cf) as f:
                    for c in json.load(f):
                        compounds_by_cid[c['cid']] = c
            
            if compounds_by_cid:
                merged = data_dir / "merged_compounds.json"
                with open(merged, 'w') as f:
                    json.dump(list(compounds_by_cid.values()), f)
                self.populate_compounds_enhanced(merged)
        
        logger.info("✅ All data populated")
    
    def print_stats(self):
        """Print collection statistics."""
        logger.info("📊 Collection Statistics:")
        
        for name in self.COLLECTIONS.keys():
            try:
                info = self.client.get_collection(name)
                logger.info(f"  {name}: {info.points_count} points")
            except:
                logger.info(f"  {name}: Collection not found")
    
    def populate_papers_enhanced(self, papers_file: Path):
        """Upload papers with enhanced 768-dim embeddings."""
        logger.info(f"Loading papers from: {papers_file}")
        
        with open(papers_file, 'r') as f:
            papers = json.load(f)
        
        logger.info(f"Generating 768-dim embeddings for {len(papers)} papers...")
        embeddings = self.generator.batch_embed_papers(papers, batch_size=16)
        
        logger.info("Uploading to Qdrant...")
        points = []
        
        for i, (paper, embedding) in enumerate(zip(papers, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "authors": paper.get("authors", []),
                    "journal": paper.get("journal", ""),
                    "publication_date": paper.get("publication_date", ""),
                    "mentioned_compounds": paper.get("mentioned_compounds", []),
                    "mesh_terms": paper.get("mesh_terms", [])
                }
            )
            points.append(point)
        
        # Batch upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name="research_papers", points=batch)
            logger.info(f"  Uploaded {min(i+batch_size, len(points))}/{len(points)}")
        
        logger.info(f"✅ Uploaded {len(papers)} papers")
    
    def populate_compounds_enhanced(self, compounds_file: Path):
        """Upload compounds with enhanced 4096-bit fingerprints."""
        logger.info(f"Loading compounds from: {compounds_file}")
        
        with open(compounds_file, 'r') as f:
            compounds = json.load(f)
        
        logger.info(f"Generating 4096-bit fingerprints for {len(compounds)} compounds...")
        embeddings = self.generator.batch_embed_compounds(compounds, batch_size=50)
        
        logger.info("Uploading to Qdrant...")
        points = []
        
        for i, (compound, embedding) in enumerate(zip(compounds, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "cid": compound["cid"],
                    "name": compound["name"],
                    "smiles": compound.get("smiles", ""),
                    "canonical_smiles": compound.get("canonical_smiles", ""),
                    "molecular_formula": compound.get("molecular_formula", ""),
                    "molecular_weight": compound.get("molecular_weight", 0.0),
                    "source_pmids": compound.get("source_pmids", []),
                    "synonyms": compound.get("synonyms", [])
                }
            )
            points.append(point)
        
        # Batch upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name="chemical_compounds", points=batch)
            logger.info(f"  Uploaded {min(i+batch_size, len(points))}/{len(points)}")
        
        logger.info(f"✅ Uploaded {len(compounds)} compounds")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--populate", type=str)
    parser.add_argument("--stats", action="store_true")
    
    args = parser.parse_args()
    
    config = EnhancedQdrantConfig()
    
    if args.create or args.recreate:
        config.create_all_collections(recreate=args.recreate)
    
    if args.populate:
        data_dir = Path(args.populate)
        config.populate_all(data_dir)
    
    if args.stats:
        config.print_stats()