"""
Enhanced Image Ingestor
Downloads biological images from multiple high-quality sources.
Includes validation and quality filtering.
"""

import os
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
from PIL import Image
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Fix the import issue
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Now import after path is set
from src.image_manager import BioImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedImageIngestor:
    """
    Downloads biological images from multiple sources:
    - PubChem: 2D/3D compound structures
    - RCSB PDB: Protein 3D structures  
    - KEGG: Pathway diagrams
    - ChEBI: Chemical entity structures
    - DrugBank: Drug structures and interactions
    """
    
    def __init__(self, cache_dir: str = "data/images"):
        """Initialize enhanced image ingestor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "compounds_2d").mkdir(exist_ok=True)
        (self.cache_dir / "compounds_3d").mkdir(exist_ok=True)
        (self.cache_dir / "proteins").mkdir(exist_ok=True)
        (self.cache_dir / "pathways").mkdir(exist_ok=True)
        (self.cache_dir / "interactions").mkdir(exist_ok=True)
        
        # Session for persistent connections
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BiologicalResearch/1.0 (Educational Use)'
        })
        
        logger.info(f"✓ Enhanced image ingestor initialized")
    
    # ============================================================
    # 1. PUBCHEM - COMPOUND STRUCTURES (2D & 3D)
    # ============================================================
    
    def fetch_pubchem_2d_structures(
        self, 
        cids: List[int],
        size: str = "large",
        max_workers: int = 5
    ) -> List[BioImage]:
        """
        Fetch 2D structures from PubChem in parallel.
        
        Args:
            cids: List of PubChem CIDs
            size: 'small' or 'large'
            max_workers: Parallel download threads
            
        Returns:
            List of BioImage objects
        """
        logger.info(f"Fetching {len(cids)} 2D structures from PubChem...")
        
        images = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_pubchem_2d, cid, size): cid 
                for cid in cids
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    images.append(result)
        
        logger.info(f"✓ Downloaded {len(images)}/{len(cids)} structures")
        return images
    
    def _download_pubchem_2d(self, cid: int, size: str) -> Optional[BioImage]:
        """Download single 2D structure from PubChem."""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
            if size == "large":
                url += "?image_size=large"
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            if img.size[0] < 50 or img.size[1] < 50:  # Too small
                return None
            
            # Save
            image_id = f"pubchem_2d_{cid}"
            filename = f"{image_id}.png"
            filepath = self.cache_dir / "compounds_2d" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Get compound name
            name = self._get_compound_name(cid)
            
            return BioImage(
                image_id=image_id,
                image_type="structure_2d",
                source="pubchem",
                source_id=str(cid),
                url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                local_path=str(filepath),
                caption=f"2D structure of {name} (CID: {cid})",
                width=img.size[0],
                height=img.size[1],
                format="PNG",
                metadata={"cid": cid, "name": name, "size": size}
            )
            
        except Exception as e:
            logger.debug(f"Failed to download CID {cid}: {e}")
            return None
    
    def _get_compound_name(self, cid: int) -> str:
        """Get compound name from PubChem."""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data['PropertyTable']['Properties'][0]['Title']
        except:
            pass
        return f"Compound {cid}"
    
    # ============================================================
    # 2. RCSB PDB - PROTEIN STRUCTURES
    # ============================================================
    
    def fetch_protein_structures(
        self, 
        pdb_ids: List[str],
        max_workers: int = 5
    ) -> List[BioImage]:
        """
        Fetch protein 3D structure images from RCSB PDB.
        
        Args:
            pdb_ids: List of PDB IDs (e.g., ["6XM0", "1M17"])
            max_workers: Parallel threads
            
        Returns:
            List of BioImage objects
        """
        logger.info(f"Fetching {len(pdb_ids)} protein structures from PDB...")
        
        images = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_pdb_structure, pdb_id): pdb_id 
                for pdb_id in pdb_ids
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    images.append(result)
        
        logger.info(f"✓ Downloaded {len(images)}/{len(pdb_ids)} protein structures")
        return images
    
    def _download_pdb_structure(self, pdb_id: str) -> Optional[BioImage]:
        """Download single protein structure."""
        try:
            pdb_lower = pdb_id.lower()
            
            # Try assembly image first
            url = f"https://cdn.rcsb.org/images/structures/{pdb_lower}_assembly-1.jpeg"
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                # Try model image
                url = f"https://cdn.rcsb.org/images/structures/{pdb_lower}_model-1.jpeg"
                response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            
            # Save
            image_id = f"pdb_{pdb_lower}"
            filename = f"{image_id}.jpeg"
            filepath = self.cache_dir / "proteins" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Get protein name
            protein_name = self._get_protein_name(pdb_id)
            
            return BioImage(
                image_id=image_id,
                image_type="structure_3d",
                source="pdb",
                source_id=pdb_id.upper(),
                url=f"https://www.rcsb.org/structure/{pdb_id.upper()}",
                local_path=str(filepath),
                caption=f"3D structure of {protein_name} (PDB: {pdb_id.upper()})",
                width=img.size[0],
                height=img.size[1],
                format="JPEG",
                metadata={"pdb_id": pdb_id.upper(), "protein_name": protein_name}
            )
            
        except Exception as e:
            logger.debug(f"Failed to download PDB {pdb_id}: {e}")
            return None
    
    def _get_protein_name(self, pdb_id: str) -> str:
        """Get protein name from PDB."""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                title = data.get('struct', {}).get('title', '')
                return title if title else f"Protein {pdb_id}"
        except:
            pass
        return f"Protein {pdb_id}"
    
    # ============================================================
    # 3. KEGG - PATHWAY DIAGRAMS
    # ============================================================
    
    def fetch_pathway_diagrams(
        self,
        pathway_ids: List[str],
        organism: str = "hsa"
    ) -> List[BioImage]:
        """
        Fetch pathway diagrams from KEGG.
        
        Args:
            pathway_ids: List of pathway IDs (e.g., ["04010", "04151"])
            organism: Organism code (hsa=human, mmu=mouse, etc.)
            
        Returns:
            List of BioImage objects
        """
        logger.info(f"Fetching {len(pathway_ids)} pathway diagrams from KEGG...")
        
        images = []
        
        for pathway_id in pathway_ids:
            img = self._download_kegg_pathway(pathway_id, organism)
            if img:
                images.append(img)
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"✓ Downloaded {len(images)}/{len(pathway_ids)} pathways")
        return images
    
    def _download_kegg_pathway(self, pathway_id: str, organism: str) -> Optional[BioImage]:
        """Download single pathway diagram."""
        try:
            full_id = f"{organism}{pathway_id}"
            url = f"https://rest.kegg.jp/get/{full_id}/image"
            
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            
            # Save
            image_id = f"kegg_{full_id}"
            filename = f"{image_id}.png"
            filepath = self.cache_dir / "pathways" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Get pathway name
            pathway_name = self._get_pathway_name(full_id)
            
            return BioImage(
                image_id=image_id,
                image_type="pathway",
                source="kegg",
                source_id=full_id,
                url=f"https://www.kegg.jp/pathway/{full_id}",
                local_path=str(filepath),
                caption=f"KEGG pathway: {pathway_name} ({full_id})",
                width=img.size[0],
                height=img.size[1],
                format="PNG",
                metadata={"pathway_id": full_id, "name": pathway_name}
            )
            
        except Exception as e:
            logger.debug(f"Failed to download pathway {pathway_id}: {e}")
            return None
    
    def _get_pathway_name(self, pathway_id: str) -> str:
        """Get pathway name from KEGG."""
        try:
            url = f"https://rest.kegg.jp/get/{pathway_id}"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    if line.startswith('NAME'):
                        return line.split('NAME')[1].strip()
        except:
            pass
        return pathway_id
    
    # ============================================================
    # 4. AUTO-INGESTION FROM EXISTING DATA
    # ============================================================
    
    def auto_ingest_from_compounds(
        self,
        compounds_file: Path,
        max_compounds: int = 100
    ) -> List[BioImage]:
        """
        Auto-ingest images from existing compound data.
        
        Args:
            compounds_file: Path to compounds JSON
            max_compounds: Maximum to download
            
        Returns:
            List of BioImage objects
        """
        logger.info(f"Auto-ingesting from {compounds_file}...")
        
        with open(compounds_file) as f:
            compounds = json.load(f)
        
        # Get CIDs
        cids = [c['cid'] for c in compounds[:max_compounds]]
        
        # Download structures
        images = self.fetch_pubchem_2d_structures(cids, size="large")
        
        logger.info(f"✓ Auto-ingested {len(images)} images")
        return images
    
    def auto_ingest_from_papers(
        self,
        papers_file: Path,
        max_proteins: int = 50
    ) -> List[BioImage]:
        """
        Auto-ingest protein structures from papers mentioning PDB IDs.
        
        Args:
            papers_file: Path to papers JSON
            max_proteins: Maximum to download
            
        Returns:
            List of BioImage objects
        """
        logger.info(f"Auto-ingesting proteins from {papers_file}...")
        
        with open(papers_file) as f:
            papers = json.load(f)
        
        # Extract PDB IDs from abstracts
        import re
        pdb_pattern = r'\b[1-9][A-Z0-9]{3}\b'
        
        pdb_ids = set()
        for paper in papers:
            abstract = paper.get('abstract', '')
            matches = re.findall(pdb_pattern, abstract.upper())
            pdb_ids.update(matches)
        
        pdb_ids = list(pdb_ids)[:max_proteins]
        
        if not pdb_ids:
            logger.warning("No PDB IDs found in papers")
            return []
        
        # Download structures
        images = self.fetch_protein_structures(pdb_ids)
        
        logger.info(f"✓ Auto-ingested {len(images)} protein structures")
        return images
    
    # ============================================================
    # 5. COMPREHENSIVE INGESTION
    # ============================================================
    
    def ingest_comprehensive_dataset(
        self,
        data_dir: Path = Path("data/raw"),
        max_compounds: int = 200,
        max_proteins: int = 100,
        common_pathways: bool = True
    ) -> Dict[str, int]:
        """
        Ingest comprehensive image dataset.
        
        Args:
            data_dir: Directory with existing data
            max_compounds: Max compound structures
            max_proteins: Max protein structures
            common_pathways: Include common pathways
            
        Returns:
            Statistics dict
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE IMAGE INGESTION")
        logger.info("="*80)
        
        stats = {
            "compounds_2d": 0,
            "proteins_3d": 0,
            "pathways": 0,
            "total": 0
        }
        
        # 1. Compounds from existing data
        compound_files = list(data_dir.glob("*_compounds.json"))
        if compound_files:
            logger.info("\n1. Ingesting compound structures...")
            for cf in compound_files:
                images = self.auto_ingest_from_compounds(cf, max_compounds // len(compound_files))
                stats["compounds_2d"] += len(images)
        
        # 2. Proteins from papers
        paper_files = list(data_dir.glob("*_papers.json"))
        if paper_files:
            logger.info("\n2. Ingesting protein structures...")
            for pf in paper_files:
                images = self.auto_ingest_from_papers(pf, max_proteins // len(paper_files))
                stats["proteins_3d"] += len(images)
        
        # 3. Common pathways
        if common_pathways:
            logger.info("\n3. Ingesting pathway diagrams...")
            common_pathway_ids = [
                "04010",  # MAPK signaling
                "04151",  # PI3K-Akt signaling
                "04110",  # Cell cycle
                "04210",  # Apoptosis
                "04620",  # Toll-like receptor
                "04630",  # JAK-STAT signaling
                "04668",  # TNF signaling
                "05200",  # Pathways in cancer
                "05219",  # Bladder cancer
                "05223"   # Non-small cell lung cancer
            ]
            
            images = self.fetch_pathway_diagrams(common_pathway_ids)
            stats["pathways"] = len(images)
        
        stats["total"] = sum(stats.values())
        
        logger.info("\n" + "="*80)
        logger.info("INGESTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Compound 2D structures: {stats['compounds_2d']}")
        logger.info(f"Protein 3D structures: {stats['proteins_3d']}")
        logger.info(f"Pathway diagrams: {stats['pathways']}")
        logger.info(f"Total images: {stats['total']}")
        
        return stats


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Image Ingestion")
    parser.add_argument("--compounds", type=int, default=200, help="Max compounds")
    parser.add_argument("--proteins", type=int, default=100, help="Max proteins")
    parser.add_argument("--pathways", action="store_true", help="Include pathways")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    
    args = parser.parse_args()
    
    ingestor = EnhancedImageIngestor()
    
    stats = ingestor.ingest_comprehensive_dataset(
        data_dir=Path(args.data_dir),
        max_compounds=args.compounds,
        max_proteins=args.proteins,
        common_pathways=args.pathways
    )
    
    print("\n✅ Ingestion complete!")
    print(f"📁 Images saved to: data/images/")