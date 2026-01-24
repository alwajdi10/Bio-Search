"""
PubMed Data Ingestor
Fetches scientific paper abstracts from NCBI PubMed using E-utilities API.
"""

import time
import re
from typing import List, Optional, Dict
from datetime import datetime
from Bio import Entrez
from pydantic import BaseModel, Field
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure NCBI Entrez
Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY", None)


class PubMedPaper(BaseModel):
    """Structured model for a PubMed paper."""
    pmid: str
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    journal: str = ""
    publication_date: str = ""
    mesh_terms: List[str] = Field(default_factory=list)
    doi: Optional[str] = None
    mentioned_compounds: List[str] = Field(default_factory=list)


class PubMedIngestor:
    """
    Ingestor for PubMed scientific literature.
    Uses NCBI E-utilities API with rate limiting and error handling.
    """
    
    def __init__(self, rate_limit: float = 0.34):
        """
        Initialize the PubMed ingestor.
        
        Args:
            rate_limit: Seconds to wait between requests (default: 0.34s = ~3 req/sec)
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self, 
        query: str, 
        max_results: int = 100,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs.
        
        Args:
            query: Search query (e.g., "KRAS inhibitor")
            max_results: Maximum number of results to return
            min_date: Minimum date filter (YYYY/MM/DD)
            max_date: Maximum date filter (YYYY/MM/DD)
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Searching PubMed for: {query}")
            
            # Build search parameters
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'sort': 'relevance'
            }
            
            if min_date:
                params['mindate'] = min_date
            if max_date:
                params['maxdate'] = max_date
                
            # Execute search
            handle = Entrez.esearch(**params)
            record = Entrez.read(handle)
            handle.close()
            
            pmids = record['IdList']
            logger.info(f"Found {len(pmids)} papers")
            
            return pmids
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[PubMedPaper]:
        """
        Fetch detailed information for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PubMedPaper objects
        """
        if not pmids:
            return []
        
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Fetching details for {len(pmids)} papers")
            
            # Fetch records
            handle = Entrez.efetch(
                db='pubmed',
                id=','.join(pmids),
                retmode='xml'
            )
            records = Entrez.read(handle)
            handle.close()
            
            papers = []
            for article in records['PubmedArticle']:
                try:
                    paper = self._parse_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching details: {e}")
            return []
    
    def _parse_article(self, article: Dict) -> Optional[PubMedPaper]:
        """Parse a single PubMed article record."""
        try:
            medline = article['MedlineCitation']
            article_data = medline['Article']
            
            # Extract PMID
            pmid = str(medline['PMID'])
            
            # Extract title
            title = article_data.get('ArticleTitle', '')
            
            # Extract abstract
            abstract_parts = article_data.get('Abstract', {}).get('AbstractText', [])
            if isinstance(abstract_parts, list):
                abstract = ' '.join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)
            
            # Skip if no abstract
            if not abstract or abstract == '':
                logger.debug(f"Skipping PMID {pmid}: No abstract")
                return None
            
            # Extract authors
            authors = []
            author_list = article_data.get('AuthorList', [])
            for author in author_list:
                if 'LastName' in author and 'ForeName' in author:
                    authors.append(f"{author['ForeName']} {author['LastName']}")
            
            # Extract journal
            journal = article_data.get('Journal', {}).get('Title', '')
            
            # Extract publication date
            pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            pub_date_str = self._format_date(pub_date)
            
            # Extract MeSH terms
            mesh_terms = []
            mesh_list = medline.get('MeshHeadingList', [])
            for mesh in mesh_list:
                descriptor = mesh.get('DescriptorName', '')
                if descriptor:
                    mesh_terms.append(str(descriptor))
            
            # Extract DOI
            doi = None
            article_ids = article.get('PubmedData', {}).get('ArticleIdList', [])
            for aid in article_ids:
                if aid.attributes.get('IdType') == 'doi':
                    doi = str(aid)
            
            # Extract compound mentions (simple regex-based extraction)
            mentioned_compounds = self._extract_compound_mentions(abstract + ' ' + title)
            
            return PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date_str,
                mesh_terms=mesh_terms,
                doi=doi,
                mentioned_compounds=mentioned_compounds
            )
            
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    def _format_date(self, pub_date: Dict) -> str:
        """Format publication date from PubMed format."""
        try:
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '01')
            day = pub_date.get('Day', '01')
            
            # Convert month name to number if needed
            month_map = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
            }
            
            if month in month_map:
                month = month_map[month]
            elif len(month) == 1:
                month = f'0{month}'
            
            if len(day) == 1:
                day = f'0{day}'
            
            return f"{year}-{month}-{day}"
        except:
            return ""
    
    def _extract_compound_mentions(self, text: str) -> List[str]:
        """
        Extract potential compound mentions from text.
        Simple regex-based approach. Can be improved with NER.
        """
        compounds = set()
        
        # Pattern for common drug/compound naming conventions
        patterns = [
            r'\b[A-Z][a-z]+[a-z]{4,}nib\b',  # -nib suffix (kinase inhibitors)
            r'\b[A-Z][a-z]+[a-z]{4,}mab\b',  # -mab suffix (monoclonal antibodies)
            r'\b[A-Z][a-z]+[a-z]{4,}tinib\b',  # -tinib suffix
            r'\b[A-Z]{2,}[-\s][0-9]{2,}\b',  # Codes like KRAS-G12C, PD-1
            r'\b[A-Z][a-z]{2,}[-][A-Z][0-9]+[A-Z]?\b',  # Codes like Drug-A123
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            compounds.update(matches)
        
        return list(compounds)
    
    def search_and_fetch(
        self,
        query: str,
        max_results: int = 50,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[PubMedPaper]:
        """
        Convenience method to search and fetch in one call.
        
        Args:
            query: Search query
            max_results: Maximum papers to retrieve
            min_date: Minimum date filter
            max_date: Maximum date filter
            
        Returns:
            List of PubMedPaper objects
        """
        pmids = self.search(query, max_results, min_date, max_date)
        if not pmids:
            return []
        
        return self.fetch_details(pmids)


# Example usage
if __name__ == "__main__":
    ingestor = PubMedIngestor()
    
    # Test search
    papers = ingestor.search_and_fetch("KRAS inhibitor", max_results=5)
    
    print(f"\nRetrieved {len(papers)} papers:\n")
    for paper in papers:
        print(f"PMID: {paper.pmid}")
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}...")
        print(f"Journal: {paper.journal}")
        print(f"Date: {paper.publication_date}")
        print(f"Compounds mentioned: {paper.mentioned_compounds}")
        print(f"Abstract preview: {paper.abstract[:200]}...")
        print("-" * 80)