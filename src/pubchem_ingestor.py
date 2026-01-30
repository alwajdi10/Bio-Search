"""
PubChem Data Ingestor
Fetches chemical compound data from NCBI PubChem using REST API.
"""

import time
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemicalCompound(BaseModel):
    """Structured model for a chemical compound."""
    cid: int
    name: str
    smiles: str = ""
    canonical_smiles: str = ""
    iupac_name: str = ""
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    inchi: str = ""
    inchi_key: str = ""
    synonyms: List[str] = Field(default_factory=list)
    source_pmids: List[str] = Field(default_factory=list)
    trial_ncts: List[str] = Field(default_factory=list)
    # Additional computed properties
    num_atoms: int = 0
    num_bonds: int = 0
    num_rings: int = 0
    logp: float = 0.0


class PubChemIngestor:
    """
    Ingestor for PubChem chemical compound data.
    Uses PubChem REST API with rate limiting and error handling.
    """
    
    def __init__(self, rate_limit: float = 0.25):
        """
        Initialize the PubChem ingestor.
        
        Args:
            rate_limit: Seconds to wait between requests (default: 0.25s = 4 req/sec)
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search_by_name(self, name: str, max_results: int = 5) -> List[int]:
        """
        Search for compounds by name and return CIDs.
        
        Args:
            name: Compound name
            max_results: Maximum number of CIDs to return
            
        Returns:
            List of PubChem Compound IDs (CIDs)
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Searching PubChem for compound: {name}")
            
            # Use pubchempy for name search (handles synonyms better)
            compounds = pcp.get_compounds(name, 'name', listkey_count=max_results)
            cids = [c.cid for c in compounds]
            
            logger.info(f"Found {len(cids)} compounds for '{name}'")
            return cids[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching by name '{name}': {e}")
            return []
    
    def search_by_smiles(self, smiles: str) -> Optional[int]:
        """
        Search for a compound by SMILES string.
        
        Args:
            smiles: SMILES representation
            
        Returns:
            PubChem CID or None if not found
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Searching PubChem by SMILES: {smiles}")
            
            # Validate SMILES first
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # Search using pubchempy
            compounds = pcp.get_compounds(smiles, 'smiles')
            
            if compounds:
                cid = compounds[0].cid
                logger.info(f"Found CID {cid} for SMILES")
                return cid
            else:
                logger.info(f"No compound found for SMILES: {smiles}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching by SMILES: {e}")
            return None
    
    def fetch_by_cid(self, cid: int) -> Optional[ChemicalCompound]:
        """
        Fetch detailed compound information by CID.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            ChemicalCompound object or None
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Fetching compound details for CID: {cid}")
            
            # Get compound using pubchempy
            compound = pcp.Compound.from_cid(cid)
            
            if not compound:
                logger.warning(f"No data found for CID {cid}")
                return None
            
            # Get additional synonyms
            synonyms = self._fetch_synonyms(cid)
            
            # Compute additional properties using RDKit
            rdkit_props = self._compute_rdkit_properties(compound.canonical_smiles)
            
            return ChemicalCompound(
                cid=compound.cid,
                name=compound.iupac_name or synonyms[0] if synonyms else f"CID_{cid}",
                smiles=compound.isomeric_smiles or compound.canonical_smiles,
                canonical_smiles=compound.canonical_smiles,
                iupac_name=compound.iupac_name or "",
                molecular_formula=compound.molecular_formula or "",
                molecular_weight=compound.molecular_weight or 0.0,
                inchi=compound.inchi or "",
                inchi_key=compound.inchikey or "",
                synonyms=synonyms[:10],  # Limit to 10 synonyms
                source_pmids=[],  # Will be populated later
                **rdkit_props
            )
            
        except Exception as e:
            logger.error(f"Error fetching CID {cid}: {e}")
            return None
    
    def _fetch_synonyms(self, cid: int, max_synonyms: int = 10) -> List[str]:
        """Fetch synonyms for a compound."""
        self._wait_for_rate_limit()
        
        try:
            url = f"{self.base_url}/compound/cid/{cid}/synonyms/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                synonyms = data.get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])
                return synonyms[:max_synonyms]
            else:
                return []
        except Exception as e:
            logger.debug(f"Could not fetch synonyms for CID {cid}: {e}")
            return []
    
    def _compute_rdkit_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Compute additional molecular properties using RDKit.
        
        Args:
            smiles: Canonical SMILES string
            
        Returns:
            Dictionary of computed properties
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    'num_atoms': 0,
                    'num_bonds': 0,
                    'num_rings': 0,
                    'logp': 0.0
                }
            
            return {
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'logp': round(Descriptors.MolLogP(mol), 2)
            }
        except Exception as e:
            logger.debug(f"Error computing RDKit properties: {e}")
            return {
                'num_atoms': 0,
                'num_bonds': 0,
                'num_rings': 0,
                'logp': 0.0
            }
    
    def fetch_multiple(self, cids: List[int]) -> List[ChemicalCompound]:
        """
        Fetch multiple compounds by CID.
        
        Args:
            cids: List of PubChem CIDs
            
        Returns:
            List of ChemicalCompound objects
        """
        compounds = []
        
        for cid in cids:
            compound = self.fetch_by_cid(cid)
            if compound:
                compounds.append(compound)
        
        logger.info(f"Successfully fetched {len(compounds)}/{len(cids)} compounds")
        return compounds
    
    def search_and_fetch(
        self, 
        names: List[str], 
        max_per_name: int = 1
    ) -> List[ChemicalCompound]:
        """
        Convenience method to search by names and fetch details.
        
        Args:
            names: List of compound names
            max_per_name: Maximum compounds to fetch per name
            
        Returns:
            List of ChemicalCompound objects
        """
        all_compounds = []
        
        for name in names:
            cids = self.search_by_name(name, max_results=max_per_name)
            compounds = self.fetch_multiple(cids)
            all_compounds.extend(compounds)
        
        return all_compounds
    
    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string using RDKit.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_2d_structure_url(self, cid: int, width: int = 300, height: int = 300) -> str:
        """
        Get URL for 2D structure image.
        
        Args:
            cid: PubChem CID
            width: Image width
            height: Image height
            
        Returns:
            URL to structure image
        """
        return f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={cid}&t=l"


# Example usage
if __name__ == "__main__":
    ingestor = PubChemIngestor()
    
    # Test 1: Search by name
    print("\n=== Test 1: Search by name ===")
    compound_names = ["Sotorasib", "Aspirin", "Ibuprofen"]
    compounds = ingestor.search_and_fetch(compound_names, max_per_name=1)
    
    for compound in compounds:
        print(f"\nCID: {compound.cid}")
        print(f"Name: {compound.name}")
        print(f"IUPAC: {compound.iupac_name}")
        print(f"Formula: {compound.molecular_formula}")
        print(f"Weight: {compound.molecular_weight}")
        print(f"SMILES: {compound.canonical_smiles}")
        print(f"Synonyms: {', '.join(compound.synonyms[:3])}...")
        print(f"Atoms: {compound.num_atoms}, Rings: {compound.num_rings}, LogP: {compound.logp}")
        print(f"2D Image: {ingestor.get_2d_structure_url(compound.cid)}")
    
    # Test 2: Search by SMILES
    print("\n=== Test 2: Search by SMILES ===")
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    cid = ingestor.search_by_smiles(aspirin_smiles)
    if cid:
        compound = ingestor.fetch_by_cid(cid)
        print(f"Found: {compound.name} (CID: {compound.cid})")
    
    # Test 3: Validate SMILES
    print("\n=== Test 3: Validate SMILES ===")
    test_smiles = [
        ("CC(=O)Oc1ccccc1C(=O)O", "Valid"),
        ("invalid_smiles", "Invalid")
    ]
    for smiles, expected in test_smiles:
        is_valid = ingestor.validate_smiles(smiles)
        print(f"{smiles}: {is_valid} (expected: {expected})")