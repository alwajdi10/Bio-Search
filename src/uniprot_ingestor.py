"""
UniProt Protein Ingestor
Fetches protein data from UniProt REST API.
"""

import time
import requests
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Protein(BaseModel):
    """Structured model for a protein."""
    uniprot_id: str
    entry_name: str
    protein_name: str
    gene_names: List[str] = Field(default_factory=list)
    organism: str = ""
    sequence: str = ""
    sequence_length: int = 0
    function: str = ""
    subcellular_location: List[str] = Field(default_factory=list)
    disease_involvement: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    source_pmids: List[str] = Field(default_factory=list)
    go_terms: List[str] = Field(default_factory=list)


class UniProtIngestor:
    """
    Ingestor for UniProt protein data.
    Uses UniProt REST API.
    """
    
    def __init__(self, rate_limit: float = 0.5):
        """
        Initialize UniProt ingestor.
        
        Args:
            rate_limit: Seconds between requests
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.base_url = "https://rest.uniprot.org/uniprotkb"
    
    def _wait_for_rate_limit(self):
        """Ensure rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search_by_name(self, name: str, max_results: int = 10) -> List[str]:
        """
        Search for proteins by name/gene.
        
        Args:
            name: Protein or gene name
            max_results: Max results
            
        Returns:
            List of UniProt IDs
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Searching UniProt for: {name}")
            
            params = {
                "query": f"(protein_name:{name}) OR (gene:{name})",
                "format": "json",
                "size": max_results,
                "fields": "accession"
            }
            
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                ids = [entry["primaryAccession"] for entry in data.get("results", [])]
                logger.info(f"Found {len(ids)} proteins")
                return ids
            else:
                logger.warning(f"Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def fetch_by_id(self, uniprot_id: str) -> Optional[Protein]:
        """
        Fetch protein details by UniProt ID.
        
        Args:
            uniprot_id: UniProt accession
            
        Returns:
            Protein object or None
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Fetching protein: {uniprot_id}")
            
            response = requests.get(
                f"{self.base_url}/{uniprot_id}.json",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_protein(data)
            else:
                logger.warning(f"Fetch failed for {uniprot_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {uniprot_id}: {e}")
            return None
    
    def _parse_protein(self, data: Dict) -> Protein:
        """Parse UniProt JSON to Protein model."""
        try:
            # Basic info
            uniprot_id = data.get("primaryAccession", "")
            entry_name = data.get("uniProtkbId", "")
            
            # Protein name
            protein_name = ""
            if "proteinDescription" in data:
                rec_name = data["proteinDescription"].get("recommendedName", {})
                protein_name = rec_name.get("fullName", {}).get("value", "")
            
            # Gene names
            gene_names = []
            if "genes" in data:
                for gene in data["genes"]:
                    if "geneName" in gene:
                        gene_names.append(gene["geneName"].get("value", ""))
            
            # Organism
            organism = ""
            if "organism" in data:
                organism = data["organism"].get("scientificName", "")
            
            # Sequence
            sequence = ""
            sequence_length = 0
            if "sequence" in data:
                sequence = data["sequence"].get("value", "")
                sequence_length = data["sequence"].get("length", 0)
            
            # Function
            function = ""
            if "comments" in data:
                for comment in data["comments"]:
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            function = texts[0].get("value", "")
                        break
            
            # Subcellular location
            subcellular = []
            if "comments" in data:
                for comment in data["comments"]:
                    if comment.get("commentType") == "SUBCELLULAR LOCATION":
                        locations = comment.get("subcellularLocations", [])
                        for loc in locations:
                            if "location" in loc:
                                subcellular.append(loc["location"].get("value", ""))
            
            # Disease involvement
            diseases = []
            if "comments" in data:
                for comment in data["comments"]:
                    if comment.get("commentType") == "DISEASE":
                        disease_info = comment.get("disease", {})
                        disease_name = disease_info.get("diseaseId", "")
                        if disease_name:
                            diseases.append(disease_name)
            
            # Extract PMIDs from references
            pmids = []
            if "references" in data:
                for ref in data["references"]:
                    citation = ref.get("citation", {})
                    if "citationCrossReferences" in citation:
                        for xref in citation["citationCrossReferences"]:
                            if xref.get("database") == "PubMed":
                                pmids.append(xref.get("id", ""))
            
            # GO terms
            go_terms = []
            if "uniProtKBCrossReferences" in data:
                for xref in data["uniProtKBCrossReferences"]:
                    if xref.get("database") == "GO":
                        go_terms.append(xref.get("id", ""))
            
            return Protein(
                uniprot_id=uniprot_id,
                entry_name=entry_name,
                protein_name=protein_name,
                gene_names=gene_names,
                organism=organism,
                sequence=sequence,
                sequence_length=sequence_length,
                function=function,
                subcellular_location=subcellular,
                disease_involvement=diseases,
                source_pmids=pmids[:10],
                go_terms=go_terms[:10]
            )
            
        except Exception as e:
            logger.error(f"Error parsing protein: {e}")
            return None
    
    def search_and_fetch(
        self, 
        names: List[str], 
        max_per_name: int = 3
    ) -> List[Protein]:
        """
        Search and fetch proteins.
        
        Args:
            names: Protein/gene names
            max_per_name: Max proteins per name
            
        Returns:
            List of Protein objects
        """
        all_proteins = []
        
        for name in names:
            ids = self.search_by_name(name, max_results=max_per_name)
            
            for uid in ids:
                protein = self.fetch_by_id(uid)
                if protein:
                    all_proteins.append(protein)
        
        return all_proteins


# Example usage
if __name__ == "__main__":
    ingestor = UniProtIngestor()
    
    # Test search
    print("\n=== Test: Search Proteins ===")
    proteins = ingestor.search_and_fetch(["KRAS", "EGFR", "TP53"], max_per_name=2)
    
    for protein in proteins:
        print(f"\n{protein.protein_name}")
        print(f"  UniProt ID: {protein.uniprot_id}")
        print(f"  Gene: {', '.join(protein.gene_names)}")
        print(f"  Organism: {protein.organism}")
        print(f"  Length: {protein.sequence_length} aa")
        print(f"  Function: {protein.function[:100]}...")
        print(f"  PMIDs: {len(protein.source_pmids)}")
        print(f"  URL: https://www.uniprot.org/uniprotkb/{protein.uniprot_id}")