"""
Enhanced Qdrant Cloud Manager
Manages multiple collections for multi-modal biological data.
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embeddings import EmbeddingGenerator

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQdrantManager:
    """
    Enhanced manager for multi-modal Qdrant collections.
    
    Collections:
    - research_papers: Scientific papers from PubMed
    - chemical_compounds: Compounds from PubChem
    - proteins: Proteins from UniProt
    - clinical_trials: Clinical trials from ClinicalTrials.gov
    """
    
    COLLECTIONS = {
        "research_papers": {"dim": 384, "type": "text"},
        "chemical_compounds": {"dim": 2048, "type": "fingerprint"},
        "proteins": {"dim": 384, "type": "text"},
        "clinical_trials": {"dim": 384, "type": "text"}
    }
    
    def __init__(self):
        """Initialize Qdrant client."""
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if not url or not api_key:
            raise ValueError(
                "Missing Qdrant credentials! Set QDRANT_URL and QDRANT_API_KEY in .env"
            )
        
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.generator = EmbeddingGenerator()
        
        logger.info(f"✓ Connected to Qdrant Cloud: {url}")
    
    def create_all_collections(self, recreate: bool = False):
        """Create all collections."""
        logger.info("Creating collections...")
        
        for name, config in self.COLLECTIONS.items():
            if recreate:
                try:
                    self.client.delete_collection(name)
                    logger.info(f"  Deleted old collection: {name}")
                except:
                    pass
            
            try:
                self.client.get_collection(name)
                logger.info(f"  ✓ Collection exists: {name}")
            except:
                logger.info(f"  Creating: {name} (dim={config['dim']})")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=config["dim"],
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"  ✓ Created: {name}")
        
        logger.info("✅ All collections ready")
    
    def populate_papers(self, papers_file: Path):
        """Upload papers to Qdrant."""
        logger.info(f"Loading papers from: {papers_file}")
        
        with open(papers_file, 'r') as f:
            papers = json.load(f)
        
        logger.info(f"Generating embeddings for {len(papers)} papers...")
        embeddings = self.generator.batch_embed_papers(papers, batch_size=32)
        
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
        
        self._batch_upload(points, "research_papers")
        logger.info(f"✅ Uploaded {len(papers)} papers")
    
    def populate_compounds(self, compounds_file: Path):
        """Upload compounds to Qdrant."""
        logger.info(f"Loading compounds from: {compounds_file}")
        
        with open(compounds_file, 'r') as f:
            compounds = json.load(f)
        
        logger.info(f"Generating embeddings for {len(compounds)} compounds...")
        embeddings = self.generator.batch_embed_compounds(compounds, batch_size=100)
        
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
        
        self._batch_upload(points, "chemical_compounds")
        logger.info(f"✅ Uploaded {len(compounds)} compounds")
    
    def populate_proteins(self, proteins_file: Path):
        """Upload proteins to Qdrant."""
        logger.info(f"Loading proteins from: {proteins_file}")
        
        with open(proteins_file, 'r') as f:
            proteins = json.load(f)
        
        logger.info(f"Generating embeddings for {len(proteins)} proteins...")
        
        # Embed using protein name + function
        texts = []
        for protein in proteins:
            text = f"{protein['protein_name']}. {protein.get('function', '')}"
            texts.append(text)
        
        embeddings = self.generator.embed_text(texts)
        
        logger.info("Uploading to Qdrant...")
        points = []
        for i, (protein, embedding) in enumerate(zip(proteins, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "uniprot_id": protein["uniprot_id"],
                    "protein_name": protein["protein_name"],
                    "gene_names": protein.get("gene_names", []),
                    "organism": protein.get("organism", ""),
                    "function": protein.get("function", ""),
                    "sequence_length": protein.get("sequence_length", 0),
                    "source_pmids": protein.get("source_pmids", []),
                    "go_terms": protein.get("go_terms", [])
                }
            )
            points.append(point)
        
        self._batch_upload(points, "proteins")
        logger.info(f"✅ Uploaded {len(proteins)} proteins")
    
    def populate_trials(self, trials_file: Path):
        """Upload clinical trials to Qdrant."""
        logger.info(f"Loading trials from: {trials_file}")
        
        with open(trials_file, 'r') as f:
            trials = json.load(f)
        
        logger.info(f"Generating embeddings for {len(trials)} trials...")
        
        # Embed using title + summary
        texts = []
        for trial in trials:
            text = f"{trial['title']}. {trial.get('summary', '')}"
            texts.append(text)
        
        embeddings = self.generator.embed_text(texts)
        
        logger.info("Uploading to Qdrant...")
        points = []
        for i, (trial, embedding) in enumerate(zip(trials, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "nct_id": trial["nct_id"],
                    "title": trial["title"],
                    "status": trial.get("status", ""),
                    "phase": trial.get("phase", ""),
                    "conditions": trial.get("conditions", []),
                    "interventions": trial.get("interventions", []),
                    "sponsor": trial.get("sponsor", ""),
                    "summary": trial.get("summary", ""),
                    "related_pmids": trial.get("related_pmids", [])
                }
            )
            points.append(point)
        
        self._batch_upload(points, "clinical_trials")
        logger.info(f"✅ Uploaded {len(trials)} trials")
    
    def _batch_upload(self, points: List[PointStruct], collection: str):
        """Upload points in batches."""
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(collection_name=collection, points=batch)
            logger.info(f"  Uploaded {min(i+batch_size, len(points))}/{len(points)}")
    
    def populate_all(self, data_dir: Path):
        """
        Populate all collections from a data directory.
        Automatically finds and merges files.
        """
        logger.info(f"Populating from: {data_dir}")
        
        # Find and merge papers
        paper_files = list(data_dir.glob("*_papers.json"))
        if paper_files:
            all_papers = []
            for pf in paper_files:
                with open(pf) as f:
                    all_papers.extend(json.load(f))
            
            merged_papers = data_dir / "merged_papers.json"
            with open(merged_papers, 'w') as f:
                json.dump(all_papers, f)
            
            self.populate_papers(merged_papers)
        
        # Find and merge compounds
        compound_files = list(data_dir.glob("*_compounds.json"))
        if compound_files:
            compounds_by_cid = {}
            for cf in compound_files:
                with open(cf) as f:
                    for c in json.load(f):
                        compounds_by_cid[c['cid']] = c
            
            merged_compounds = data_dir / "merged_compounds.json"
            with open(merged_compounds, 'w') as f:
                json.dump(list(compounds_by_cid.values()), f)
            
            self.populate_compounds(merged_compounds)
        
        # Find and merge proteins
        protein_files = list(data_dir.glob("*_proteins.json"))
        if protein_files:
            proteins_by_id = {}
            for pf in protein_files:
                with open(pf) as f:
                    for p in json.load(f):
                        proteins_by_id[p['uniprot_id']] = p
            
            merged_proteins = data_dir / "merged_proteins.json"
            with open(merged_proteins, 'w') as f:
                json.dump(list(proteins_by_id.values()), f)
            
            self.populate_proteins(merged_proteins)
        
        # Find and merge trials
        trial_files = list(data_dir.glob("*_trials.json"))
        if trial_files:
            trials_by_nct = {}
            for tf in trial_files:
                with open(tf) as f:
                    for t in json.load(f):
                        trials_by_nct[t['nct_id']] = t
            
            merged_trials = data_dir / "merged_trials.json"
            with open(merged_trials, 'w') as f:
                json.dump(list(trials_by_nct.values()), f)
            
            self.populate_trials(merged_trials)
        
        logger.info("✅ All data populated")
    
    def get_stats(self) -> Dict:
        """Get statistics for all collections."""
        stats = {}
        
        for name in self.COLLECTIONS.keys():
            try:
                info = self.client.get_collection(name)
                stats[name] = {
                    'count': info.points_count,
                    'vector_size': info.config.params.vectors.size
                }
            except:
                stats[name] = {'error': 'Collection not found'}
        
        return stats
    
    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("QDRANT COLLECTIONS STATISTICS")
        print("="*80)
        
        for name, info in stats.items():
            if 'error' in info:
                print(f"\n❌ {name}: {info['error']}")
            else:
                print(f"\n✅ {name}")
                print(f"   Points: {info['count']:,}")
                print(f"   Vector Dim: {info['vector_size']}")
        
        print("\n" + "="*80)


# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Qdrant Manager")
    parser.add_argument("--create", action="store_true", help="Create collections")
    parser.add_argument("--recreate", action="store_true", help="Recreate collections")
    parser.add_argument("--populate", type=str, help="Populate from data directory")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    manager = EnhancedQdrantManager()
    
    if args.create or args.recreate:
        manager.create_all_collections(recreate=args.recreate)
    
    if args.populate:
        data_dir = Path(args.populate)
        manager.populate_all(data_dir)
    
    if args.stats or not any([args.create, args.recreate, args.populate]):
        manager.print_stats()