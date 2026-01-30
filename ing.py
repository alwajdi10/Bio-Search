"""
Scalable Ingestion Manager - Optimized for Qdrant Free Tier
Maximizes 1GB RAM / 4GB disk space efficiently.

Strategy:
- Batch processing to minimize memory
- Incremental uploads (never load everything at once)
- Smart filtering (only ingest high-quality data)
- Compression-friendly metadata
- Progress tracking with resume capability
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from collections import defaultdict
import sys
from datetime import datetime
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pubmed_ingestor import PubMedIngestor, PubMedPaper
from src.pubchem_ingestor import PubChemIngestor, ChemicalCompound
from src.uniprot_ingestor import UniProtIngestor, Protein
from src.clinical_trials_ingestor import ClinicalTrialsIngestor, ClinicalTrial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalableIngestionManager:
    """
    Optimized ingestion for Qdrant free tier (1GB RAM, 4GB disk).
    
    Key optimizations:
    - Processes in small batches (never loads everything)
    - Filters low-quality data early
    - Saves incrementally (can resume if interrupted)
    - Tracks progress to avoid duplicates
    - Minimal memory footprint
    """
    
    # FREE TIER LIMITS
    MAX_TOTAL_VECTORS = 100000  # Safe limit for free tier
    MAX_BATCH_SIZE = 50  # Process 50 items at a time
    
    # QUALITY THRESHOLDS (filter early to save space)
    MIN_ABSTRACT_LENGTH = 100  # Skip papers with very short abstracts
    MIN_COMPOUND_ATOMS = 5  # Skip very small molecules
    MIN_PROTEIN_LENGTH = 50  # Skip protein fragments
    
    def __init__(self, output_dir: str = "data/raw", cache_dir: str = "data/cache"):
        """
        Initialize scalable ingestion manager.
        
        Args:
            output_dir: Where to save final data
            cache_dir: Where to save progress (for resuming)
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ingestors
        self.pubmed = PubMedIngestor()
        self.pubchem = PubChemIngestor()
        self.uniprot = UniProtIngestor()
        self.clinical_trials = ClinicalTrialsIngestor()
        
        # Progress tracking
        self.progress_file = self.cache_dir / "ingestion_progress.json"
        self.progress = self._load_progress()
        
        logger.info(f"Initialized ScalableIngestionManager")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Cache: {self.cache_dir}")
    
    def _load_progress(self) -> Dict:
        """Load progress from previous runs."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                logger.info(f"Loaded progress: {progress.get('completed_topics', 0)} topics completed")
                return progress
        return {
            "completed_topics": [],
            "processed_pmids": set(),
            "processed_cids": set(),
            "last_update": None
        }
    
    def _save_progress(self):
        """Save progress for resuming."""
        self.progress["last_update"] = datetime.now().isoformat()
        # Convert sets to lists for JSON
        progress_copy = self.progress.copy()
        progress_copy["processed_pmids"] = list(self.progress["processed_pmids"])
        progress_copy["processed_cids"] = list(self.progress["processed_cids"])
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_copy, f, indent=2)
    
    def ingest_large_dataset(
        self,
        topics: List[Dict],
        papers_per_topic: int = 100,
        max_total_papers: int = 5000,
        quality_filter: bool = True
    ) -> Dict:
        """
        Ingest large dataset efficiently.
        
        Args:
            topics: List of research topics (e.g., [{"query": "KRAS inhibitor", ...}])
            papers_per_topic: Papers to fetch per topic
            max_total_papers: Total paper limit (to fit in free tier)
            quality_filter: Whether to filter low-quality data
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting large-scale ingestion:")
        logger.info(f"  Topics: {len(topics)}")
        logger.info(f"  Papers per topic: {papers_per_topic}")
        logger.info(f"  Max total papers: {max_total_papers}")
        logger.info(f"  Quality filter: {quality_filter}")
        
        total_papers = 0
        total_compounds = 0
        total_proteins = 0
        total_trials = 0
        
        for i, topic in enumerate(topics, 1):
            # Check if already completed
            if topic["query"] in self.progress["completed_topics"]:
                logger.info(f"Skipping completed topic: {topic['query']}")
                continue
            
            # Check if we've hit limits
            if total_papers >= max_total_papers:
                logger.warning(f"Reached max papers limit ({max_total_papers})")
                break
            
            logger.info(f"\n{'='*80}")
            logger.info(f"TOPIC {i}/{len(topics)}: {topic['query']}")
            logger.info(f"{'='*80}")
            
            # Adjust papers_per_topic if close to limit
            remaining = max_total_papers - total_papers
            papers_to_fetch = min(papers_per_topic, remaining)
            
            try:
                # Ingest this topic (incremental)
                stats = self._ingest_topic_incremental(
                    query=topic["query"],
                    max_papers=papers_to_fetch,
                    min_date=topic.get("min_date"),
                    quality_filter=quality_filter
                )
                
                total_papers += stats["papers"]
                total_compounds += stats["compounds"]
                total_proteins += stats.get("proteins", 0)
                total_trials += stats.get("trials", 0)
                
                # Mark as completed
                self.progress["completed_topics"].append(topic["query"])
                self._save_progress()
                
                logger.info(f"✓ Topic complete: {stats}")
                
            except Exception as e:
                logger.error(f"Error on topic '{topic['query']}': {e}")
                continue
        
        # Final statistics
        final_stats = {
            "total_papers": total_papers,
            "total_compounds": total_compounds,
            "total_proteins": total_proteins,
            "total_trials": total_trials,
            "topics_completed": len(self.progress["completed_topics"])
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"INGESTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total papers: {total_papers}")
        logger.info(f"Total compounds: {total_compounds}")
        logger.info(f"Total proteins: {total_proteins}")
        logger.info(f"Total trials: {total_trials}")
        
        return final_stats
    
    def _ingest_topic_incremental(
        self,
        query: str,
        max_papers: int,
        min_date: Optional[str] = None,
        quality_filter: bool = True
    ) -> Dict:
        """
        Ingest a single topic incrementally (never loads all at once).
        
        Returns:
            Stats for this topic
        """
        # Fetch papers in batches
        logger.info(f"Fetching papers for: {query}")
        papers = self.pubmed.search_and_fetch(
            query=query,
            max_results=max_papers,
            min_date=min_date
        )
        
        if not papers:
            return {"papers": 0, "compounds": 0}
        
        # Filter quality (before processing)
        if quality_filter:
            papers = self._filter_papers(papers)
            logger.info(f"After quality filter: {len(papers)} papers")
        
        # Remove duplicates (check against progress)
        papers = [p for p in papers if p.pmid not in self.progress["processed_pmids"]]
        logger.info(f"After dedup: {len(papers)} new papers")
        
        if not papers:
            return {"papers": 0, "compounds": 0}
        
        # Extract compounds (incremental)
        compound_names = self._extract_unique_compounds(papers)
        logger.info(f"Extracted {len(compound_names)} unique compounds")
        
        # Fetch compounds (batch by batch)
        compounds = self._fetch_compounds_incremental(compound_names)
        logger.info(f"Fetched {len(compounds)} compounds")
        
        # Create links
        papers, compounds = self._create_links(papers, compounds)
        
        # Save incrementally (append to existing files)
        self._save_incremental(query, papers, compounds)
        
        # Update progress
        for paper in papers:
            self.progress["processed_pmids"].add(paper.pmid)
        for compound in compounds:
            self.progress["processed_cids"].add(compound.cid)
        
        return {
            "papers": len(papers),
            "compounds": len(compounds)
        }
    
    def _filter_papers(self, papers: List[PubMedPaper]) -> List[PubMedPaper]:
        """Filter low-quality papers to save space."""
        filtered = []
        
        for paper in papers:
            # Skip papers without abstracts
            if not paper.abstract or len(paper.abstract) < self.MIN_ABSTRACT_LENGTH:
                continue
            
            # Skip papers without publication date
            if not paper.publication_date:
                continue
            
            # Keep high-quality papers
            filtered.append(paper)
        
        return filtered
    
    def _extract_unique_compounds(self, papers: List[PubMedPaper]) -> Set[str]:
        """Extract unique compound names."""
        compound_names = set()
        
        for paper in papers:
            compound_names.update(paper.mentioned_compounds)
        
        # Filter short names
        return {name for name in compound_names if len(name) >= 5}
    
    def _fetch_compounds_incremental(
        self, 
        compound_names: Set[str],
        batch_size: int = 20
    ) -> List[ChemicalCompound]:
        """Fetch compounds in batches to minimize memory."""
        compounds_by_cid = {}
        names_list = list(compound_names)
        
        for i in range(0, len(names_list), batch_size):
            batch = names_list[i:i+batch_size]
            
            for name in batch:
                # Skip if already processed
                if name in self.progress.get("processed_compound_names", set()):
                    continue
                
                try:
                    cids = self.pubchem.search_by_name(name, max_results=1)
                    
                    if cids:
                        cid = cids[0]
                        
                        # Skip if already processed
                        if cid in self.progress["processed_cids"]:
                            continue
                        
                        if cid not in compounds_by_cid:
                            compound = self.pubchem.fetch_by_cid(cid)
                            
                            if compound and self._is_quality_compound(compound):
                                compounds_by_cid[cid] = compound
                
                except Exception as e:
                    logger.debug(f"Error fetching '{name}': {e}")
            
            logger.info(f"Processed compounds {i+1}-{min(i+batch_size, len(names_list))}/{len(names_list)}")
        
        return list(compounds_by_cid.values())
    
    def _is_quality_compound(self, compound: ChemicalCompound) -> bool:
        """Filter low-quality compounds."""
        # Skip very small molecules
        if compound.num_atoms < self.MIN_COMPOUND_ATOMS:
            return False
        
        # Must have valid SMILES
        if not compound.canonical_smiles:
            return False
        
        return True
    
    def _create_links(
        self,
        papers: List[PubMedPaper],
        compounds: List[ChemicalCompound]
    ) -> tuple:
        """Create bidirectional links (memory-efficient)."""
        # Create lookup
        name_to_cid = {}
        for c in compounds:
            name_to_cid[c.name.lower()] = c.cid
            for syn in c.synonyms[:5]:  # Limit synonyms to save memory
                name_to_cid[syn.lower()] = c.cid
        
        cid_to_compound = {c.cid: c for c in compounds}
        cid_to_pmids = defaultdict(list)
        
        # Link papers → compounds
        for paper in papers:
            validated = []
            for name in paper.mentioned_compounds:
                cid = name_to_cid.get(name.lower())
                if cid:
                    validated.append(name)
                    cid_to_pmids[cid].append(paper.pmid)
            paper.mentioned_compounds = validated
        
        # Link compounds → papers
        for cid, pmids in cid_to_pmids.items():
            if cid in cid_to_compound:
                cid_to_compound[cid].source_pmids = list(set(pmids))
        
        return papers, compounds
    
    def _save_incremental(
        self,
        query: str,
        papers: List[PubMedPaper],
        compounds: List[ChemicalCompound]
    ):
        """Save incrementally (append mode)."""
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
        
        # Papers - append to existing file
        papers_file = self.output_dir / f"{safe_query}_papers.json"
        existing_papers = []
        
        if papers_file.exists():
            with open(papers_file, 'r') as f:
                existing_papers = json.load(f)
        
        all_papers = existing_papers + [p.model_dump() for p in papers]
        
        with open(papers_file, 'w') as f:
            json.dump(all_papers, f, indent=2)
        
        # Compounds - append
        compounds_file = self.output_dir / f"{safe_query}_compounds.json"
        existing_compounds = {}
        
        if compounds_file.exists():
            with open(compounds_file, 'r') as f:
                for c in json.load(f):
                    existing_compounds[c['cid']] = c
        
        for c in compounds:
            existing_compounds[c.cid] = c.model_dump()
        
        with open(compounds_file, 'w') as f:
            json.dump(list(existing_compounds.values()), f, indent=2)
        
        logger.info(f"Saved incrementally to {safe_query}_*.json")


# ============================================================================
# PREDEFINED TOPIC SETS FOR LARGE INGESTION
# ============================================================================

def get_cancer_topics(papers_per_topic: int = 100) -> List[Dict]:
    """Get comprehensive cancer research topics."""
    return [
        # KRAS pathway
        {"query": "KRAS inhibitor", "min_date": "2020/01/01"},
        {"query": "KRAS G12C mutation", "min_date": "2020/01/01"},
        {"query": "KRAS G12D mutation", "min_date": "2020/01/01"},
        
        # EGFR pathway
        {"query": "EGFR tyrosine kinase inhibitor", "min_date": "2019/01/01"},
        {"query": "EGFR mutation lung cancer", "min_date": "2020/01/01"},
        
        # CDK inhibitors
        {"query": "CDK4/6 inhibitor breast cancer", "min_date": "2019/01/01"},
        {"query": "palbociclib resistance", "min_date": "2020/01/01"},
        
        # Immunotherapy
        {"query": "PD-1 checkpoint inhibitor", "min_date": "2019/01/01"},
        {"query": "PD-L1 expression cancer", "min_date": "2020/01/01"},
        {"query": "CTLA-4 blockade", "min_date": "2019/01/01"},
        
        # PARP inhibitors
        {"query": "PARP inhibitor BRCA", "min_date": "2019/01/01"},
        {"query": "olaparib ovarian cancer", "min_date": "2020/01/01"},
        
        # MEK/BRAF
        {"query": "BRAF V600E inhibitor", "min_date": "2019/01/01"},
        {"query": "MEK inhibitor melanoma", "min_date": "2020/01/01"},
        
        # PI3K/AKT/mTOR
        {"query": "PI3K inhibitor", "min_date": "2019/01/01"},
        {"query": "mTOR inhibitor cancer", "min_date": "2019/01/01"},
        
        # ALK/ROS1
        {"query": "ALK inhibitor lung cancer", "min_date": "2019/01/01"},
        {"query": "ROS1 fusion", "min_date": "2020/01/01"},
        
        # Emerging targets
        {"query": "PROTAC degrader", "min_date": "2020/01/01"},
        {"query": "antibody-drug conjugate", "min_date": "2019/01/01"},
    ]


def get_diverse_topics(papers_per_topic: int = 50) -> List[Dict]:
    """Get diverse biomedical topics."""
    return [
        # Cancer (10 topics)
        *get_cancer_topics(papers_per_topic)[:10],
        
        # Neuroscience (5 topics)
        {"query": "Alzheimer disease treatment", "min_date": "2020/01/01"},
        {"query": "Parkinson disease therapy", "min_date": "2020/01/01"},
        {"query": "depression antidepressant", "min_date": "2020/01/01"},
        {"query": "schizophrenia antipsychotic", "min_date": "2020/01/01"},
        {"query": "epilepsy anticonvulsant", "min_date": "2020/01/01"},
        
        # Infectious disease (5 topics)
        {"query": "COVID-19 treatment", "min_date": "2020/01/01"},
        {"query": "antibiotic resistance", "min_date": "2020/01/01"},
        {"query": "antiviral therapy", "min_date": "2020/01/01"},
        {"query": "vaccine development", "min_date": "2020/01/01"},
        {"query": "fungal infection treatment", "min_date": "2020/01/01"},
        
        # Cardiovascular (3 topics)
        {"query": "heart failure treatment", "min_date": "2020/01/01"},
        {"query": "hypertension medication", "min_date": "2020/01/01"},
        {"query": "anticoagulant therapy", "min_date": "2020/01/01"},
        
        # Diabetes (2 topics)
        {"query": "type 2 diabetes treatment", "min_date": "2020/01/01"},
        {"query": "GLP-1 agonist", "min_date": "2020/01/01"},
    ]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Large-scale data ingestion")
    parser.add_argument("--mode", choices=["cancer", "diverse", "custom"], default="cancer")
    parser.add_argument("--papers-per-topic", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=5000)
    parser.add_argument("--resume", action="store_true", help="Resume previous ingestion")
    
    args = parser.parse_args()
    
    print("="*80)
    print("LARGE-SCALE DATA INGESTION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Papers per topic: {args.papers_per_topic}")
    print(f"Max total papers: {args.max_total}")
    print(f"Resume mode: {args.resume}")
    print()
    
    # Initialize manager
    manager = ScalableIngestionManager()
    
    # Select topics
    if args.mode == "cancer":
        topics = get_cancer_topics(args.papers_per_topic)
        print(f"Selected {len(topics)} cancer research topics")
    elif args.mode == "diverse":
        topics = get_diverse_topics(args.papers_per_topic)
        print(f"Selected {len(topics)} diverse biomedical topics")
    else:
        print("Custom mode - define your own topics!")
        topics = []
    
    if not topics:
        print("No topics defined. Exiting.")
        exit(0)
    
    # Run ingestion
    stats = manager.ingest_large_dataset(
        topics=topics,
        papers_per_topic=args.papers_per_topic,
        max_total_papers=args.max_total,
        quality_filter=True
    )
    
    print("\n" + "="*80)
    print("✅ INGESTION COMPLETE")
    print("="*80)
    print(f"Total papers: {stats['total_papers']}")
    print(f"Total compounds: {stats['total_compounds']}")
    print(f"Topics completed: {stats['topics_completed']}")
    print(f"\nData saved to: data/raw/")