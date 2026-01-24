"""
Enhanced Ingestion Manager
Orchestrates multi-modal data collection: papers, compounds, proteins, genes, trials.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pubmed_ingestor import PubMedIngestor, PubMedPaper
from src.pubchem_ingestor import PubChemIngestor, ChemicalCompound
from src.uniprot_ingestor import UniProtIngestor, Protein
from src.clinical_trials_ingestor import ClinicalTrialsIngestor, ClinicalTrial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedIngestionManager:
    """
    Enhanced ingestion manager for multi-modal biological data.
    
    Supports:
    - Papers (PubMed)
    - Compounds (PubChem)
    - Proteins (UniProt)
    - Clinical Trials (ClinicalTrials.gov)
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize all ingestors."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ingestors
        self.pubmed = PubMedIngestor()
        self.pubchem = PubChemIngestor()
        self.uniprot = UniProtIngestor()
        self.clinical_trials = ClinicalTrialsIngestor()
        
        logger.info(f"Initialized EnhancedIngestionManager with output dir: {self.output_dir}")
    
    def ingest_comprehensive(
        self,
        query: str,
        max_papers: int = 30,
        max_trials: int = 10,
        min_date: Optional[str] = None,
        include_proteins: bool = True,
        include_trials: bool = True
    ) -> Dict:
        """
        Comprehensive ingestion across all modalities.
        
        Pipeline:
        1. Fetch papers from PubMed
        2. Extract compound mentions → fetch from PubChem
        3. Extract protein/gene mentions → fetch from UniProt
        4. Search clinical trials for condition/intervention
        5. Link all entities bidirectionally
        6. Save structured data
        
        Args:
            query: Research topic
            max_papers: Max papers to fetch
            max_trials: Max clinical trials to fetch
            min_date: Minimum publication date
            include_proteins: Whether to fetch protein data
            include_trials: Whether to fetch clinical trials
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting comprehensive ingestion for: '{query}'")
        
        # Step 1: Fetch papers
        logger.info("=" * 80)
        logger.info("STEP 1: Fetching papers from PubMed")
        logger.info("=" * 80)
        
        papers = self.pubmed.search_and_fetch(
            query=query,
            max_results=max_papers,
            min_date=min_date
        )
        
        if not papers:
            logger.warning("No papers found")
            return {"papers": 0, "compounds": 0, "proteins": 0, "trials": 0}
        
        logger.info(f"✓ Retrieved {len(papers)} papers")
        
        # Step 2: Extract and fetch compounds
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Extracting compound mentions")
        logger.info("=" * 80)
        
        compound_names = self._extract_unique_compounds(papers)
        logger.info(f"Found {len(compound_names)} unique compound mentions")
        
        compounds = self._fetch_compounds(compound_names)
        logger.info(f"✓ Fetched {len(compounds)} compounds")
        
        # Step 3: Extract and fetch proteins (if enabled)
        proteins = []
        if include_proteins:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Extracting protein/gene mentions")
            logger.info("=" * 80)
            
            protein_names = self._extract_protein_mentions(papers)
            logger.info(f"Found {len(protein_names)} unique protein/gene mentions")
            
            proteins = self._fetch_proteins(protein_names)
            logger.info(f"✓ Fetched {len(proteins)} proteins")
        
        # Step 4: Fetch clinical trials (if enabled)
        trials = []
        if include_trials:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Searching clinical trials")
            logger.info("=" * 80)
            
            trials = self.clinical_trials.search_and_fetch(
                query=query,
                max_results=max_trials
            )
            logger.info(f"✓ Retrieved {len(trials)} clinical trials")
        
        # Step 5: Create cross-modal links
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Creating cross-modal links")
        logger.info("=" * 80)
        
        papers, compounds, proteins, trials = self._create_cross_modal_links(
            papers, compounds, proteins, trials
        )
        
        # Step 6: Save all data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Saving data")
        logger.info("=" * 80)
        
        stats = self._save_comprehensive_data(
            query, papers, compounds, proteins, trials
        )
        
        logger.info(f"\n✅ Ingestion complete: {stats}")
        return stats
    
    def _extract_unique_compounds(self, papers: List[PubMedPaper]) -> set:
        """Extract unique compound names from papers."""
        compound_names = set()
        for paper in papers:
            compound_names.update(paper.mentioned_compounds)
        
        # Filter short names (likely false positives)
        return {name for name in compound_names if len(name) >= 5}
    
    def _extract_protein_mentions(self, papers: List[PubMedPaper]) -> set:
        """
        Extract protein/gene mentions from papers.
        Uses MeSH terms and simple pattern matching.
        """
        protein_names = set()
        
        # Common protein patterns
        protein_patterns = [
            "KRAS", "EGFR", "TP53", "BRAF", "PIK3CA", "ALK", "ROS1",
            "CDK4", "CDK6", "PARP", "PD-1", "PD-L1", "CTLA-4",
            "HER2", "VEGF", "mTOR", "JAK", "STAT"
        ]
        
        for paper in papers:
            # Extract from MeSH terms
            for mesh_term in paper.mesh_terms:
                mesh_lower = mesh_term.lower()
                if any(keyword in mesh_lower for keyword in ["protein", "gene", "receptor"]):
                    protein_names.add(mesh_term)
            
            # Simple pattern matching in title/abstract
            text = (paper.title + " " + paper.abstract).upper()
            for pattern in protein_patterns:
                if pattern in text:
                    protein_names.add(pattern)
        
        return protein_names
    
    def _fetch_compounds(self, compound_names: set) -> List[ChemicalCompound]:
        """Fetch compound data from PubChem."""
        compounds_by_cid = {}
        
        for name in compound_names:
            try:
                cids = self.pubchem.search_by_name(name, max_results=1)
                
                if cids:
                    cid = cids[0]
                    if cid not in compounds_by_cid:
                        compound = self.pubchem.fetch_by_cid(cid)
                        if compound:
                            compounds_by_cid[cid] = compound
            except Exception as e:
                logger.debug(f"Error fetching compound '{name}': {e}")
        
        return list(compounds_by_cid.values())
    
    def _fetch_proteins(self, protein_names: set) -> List[Protein]:
        """Fetch protein data from UniProt."""
        all_proteins = []
        
        for name in list(protein_names)[:20]:  # Limit to avoid rate limits
            try:
                proteins = self.uniprot.search_and_fetch([name], max_per_name=1)
                all_proteins.extend(proteins)
            except Exception as e:
                logger.debug(f"Error fetching protein '{name}': {e}")
        
        return all_proteins
    
    def _create_cross_modal_links(
        self,
        papers: List[PubMedPaper],
        compounds: List[ChemicalCompound],
        proteins: List[Protein],
        trials: List[ClinicalTrial]
    ) -> tuple:
        """
        Create bidirectional links between all entities.
        
        Links:
        - Papers ↔ Compounds
        - Papers ↔ Proteins (via PMIDs)
        - Trials ↔ Compounds (via intervention names)
        - Trials ↔ Papers (via PMIDs)
        """
        # Create lookups
        compound_name_to_cid = {}
        for c in compounds:
            compound_name_to_cid[c.name.lower()] = c.cid
            for syn in c.synonyms:
                compound_name_to_cid[syn.lower()] = c.cid
        
        cid_to_compound = {c.cid: c for c in compounds}
        pmid_to_paper = {p.pmid: p for p in papers}
        uniprot_to_protein = {p.uniprot_id: p for p in proteins}
        
        # Link papers ↔ compounds
        cid_to_pmids = defaultdict(list)
        for paper in papers:
            validated_compounds = []
            for mentioned in paper.mentioned_compounds:
                cid = compound_name_to_cid.get(mentioned.lower())
                if cid:
                    validated_compounds.append(mentioned)
                    cid_to_pmids[cid].append(paper.pmid)
            paper.mentioned_compounds = validated_compounds
        
        for cid, pmids in cid_to_pmids.items():
            if cid in cid_to_compound:
                cid_to_compound[cid].source_pmids = list(set(pmids))
        
        # Link trials ↔ compounds (by intervention names)
        for trial in trials:
            for intervention in trial.interventions:
                cid = compound_name_to_cid.get(intervention.lower())
                if cid and cid in cid_to_compound:
                    # Add trial reference to compound
                    if not hasattr(cid_to_compound[cid], 'trial_ncts'):
                        cid_to_compound[cid].trial_ncts = []
                    if trial.nct_id not in cid_to_compound[cid].trial_ncts:
                        cid_to_compound[cid].trial_ncts.append(trial.nct_id)
        
        logger.info(f"Created links: "
                   f"{sum(len(c.source_pmids) for c in compounds)} paper-compound, "
                   f"{len([p for p in proteins if p.source_pmids])} paper-protein")
        
        return papers, compounds, proteins, trials
    
    def _save_comprehensive_data(
        self,
        query: str,
        papers: List[PubMedPaper],
        compounds: List[ChemicalCompound],
        proteins: List[Protein],
        trials: List[ClinicalTrial]
    ) -> Dict:
        """Save all data to JSON files."""
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
        
        # Save papers
        if papers:
            papers_file = self.output_dir / f"{safe_query}_papers.json"
            with open(papers_file, 'w') as f:
                json.dump([p.model_dump() for p in papers], f, indent=2)
            logger.info(f"✓ Saved {len(papers)} papers")
        
        # Save compounds
        if compounds:
            compounds_file = self.output_dir / f"{safe_query}_compounds.json"
            with open(compounds_file, 'w') as f:
                json.dump([c.model_dump() for c in compounds], f, indent=2)
            logger.info(f"✓ Saved {len(compounds)} compounds")
        
        # Save proteins
        if proteins:
            proteins_file = self.output_dir / f"{safe_query}_proteins.json"
            with open(proteins_file, 'w') as f:
                json.dump([p.model_dump() for p in proteins], f, indent=2)
            logger.info(f"✓ Saved {len(proteins)} proteins")
        
        # Save trials
        if trials:
            trials_file = self.output_dir / f"{safe_query}_trials.json"
            with open(trials_file, 'w') as f:
                json.dump([t.model_dump() for t in trials], f, indent=2)
            logger.info(f"✓ Saved {len(trials)} trials")
        
        # Save summary
        summary = {
            "query": query,
            "counts": {
                "papers": len(papers),
                "compounds": len(compounds),
                "proteins": len(proteins),
                "trials": len(trials)
            },
            "top_compounds": [
                {"name": c.name, "cid": c.cid, "papers": len(c.source_pmids)}
                for c in sorted(compounds, key=lambda x: len(x.source_pmids), reverse=True)[:10]
            ],
            "top_proteins": [
                {"name": p.protein_name, "id": p.uniprot_id, "gene": p.gene_names}
                for p in proteins[:10]
            ]
        }
        
        summary_file = self.output_dir / f"{safe_query}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary["counts"]


# Example usage
if __name__ == "__main__":
    manager = EnhancedIngestionManager()
    
    print("\n" + "="*80)
    print("ENHANCED MULTI-MODAL INGESTION TEST")
    print("="*80)
    
    stats = manager.ingest_comprehensive(
        query="KRAS inhibitor lung cancer",
        max_papers=15,
        max_trials=5,
        min_date="2020/01/01",
        include_proteins=True,
        include_trials=True
    )
    
    print(f"\n📊 Final Statistics:")
    print(f"  Papers: {stats['papers']}")
    print(f"  Compounds: {stats['compounds']}")
    print(f"  Proteins: {stats['proteins']}")
    print(f"  Clinical Trials: {stats['trials']}")