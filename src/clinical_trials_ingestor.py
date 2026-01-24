"""
Clinical Trials Ingestor
Fetches clinical trial data from ClinicalTrials.gov API.
"""

import time
import requests
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalTrial(BaseModel):
    """Structured model for a clinical trial."""
    nct_id: str
    title: str
    status: str = ""
    phase: str = ""
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    sponsor: str = ""
    start_date: str = ""
    completion_date: str = ""
    enrollment: int = 0
    summary: str = ""
    outcome_measures: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    related_pmids: List[str] = Field(default_factory=list)


class ClinicalTrialsIngestor:
    """
    Ingestor for ClinicalTrials.gov data.
    Uses ClinicalTrials.gov API v2.
    """
    
    def __init__(self, rate_limit: float = 0.5):
        """
        Initialize ingestor.
        
        Args:
            rate_limit: Seconds between requests
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    def _wait_for_rate_limit(self):
        """Rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self, 
        query: str, 
        max_results: int = 20,
        status: Optional[str] = None
    ) -> List[str]:
        """
        Search for clinical trials.
        
        Args:
            query: Search query (condition, drug, etc)
            max_results: Max trials to return
            status: Filter by status (e.g., "RECRUITING", "COMPLETED")
            
        Returns:
            List of NCT IDs
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Searching ClinicalTrials.gov: {query}")
            
            params = {
                "query.term": query,
                "pageSize": max_results,
                "format": "json"
            }
            
            if status:
                params["filter.overallStatus"] = status
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                studies = data.get("studies", [])
                nct_ids = [s["protocolSection"]["identificationModule"]["nctId"] 
                          for s in studies]
                logger.info(f"Found {len(nct_ids)} trials")
                return nct_ids
            else:
                logger.warning(f"Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def fetch_by_nct(self, nct_id: str) -> Optional[ClinicalTrial]:
        """
        Fetch trial details by NCT ID.
        
        Args:
            nct_id: NCT identifier (e.g., NCT12345678)
            
        Returns:
            ClinicalTrial object or None
        """
        self._wait_for_rate_limit()
        
        try:
            logger.info(f"Fetching trial: {nct_id}")
            
            response = requests.get(
                f"{self.base_url}/{nct_id}",
                params={"format": "json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_trial(data)
            else:
                logger.warning(f"Fetch failed for {nct_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {nct_id}: {e}")
            return None
    
    def _parse_trial(self, data: Dict) -> Optional[ClinicalTrial]:
        """Parse ClinicalTrials.gov JSON."""
        try:
            protocol = data.get("protocolSection", {})
            
            # Identification
            id_module = protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId", "")
            title = id_module.get("briefTitle", "")
            
            # Status
            status_module = protocol.get("statusModule", {})
            status = status_module.get("overallStatus", "")
            start_date = status_module.get("startDateStruct", {}).get("date", "")
            completion_date = status_module.get("completionDateStruct", {}).get("date", "")
            
            # Design
            design_module = protocol.get("designModule", {})
            phases = design_module.get("phases", [])
            phase = ", ".join(phases) if phases else "N/A"
            
            # Conditions
            conditions_module = protocol.get("conditionsModule", {})
            conditions = conditions_module.get("conditions", [])
            
            # Interventions
            interventions = []
            arms_module = protocol.get("armsInterventionsModule", {})
            for intervention in arms_module.get("interventions", []):
                interventions.append(intervention.get("name", ""))
            
            # Sponsor
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            sponsor = sponsor_module.get("leadSponsor", {}).get("name", "")
            
            # Enrollment
            design_module = protocol.get("designModule", {})
            enrollment = design_module.get("enrollmentInfo", {}).get("count", 0)
            
            # Description/Summary
            desc_module = protocol.get("descriptionModule", {})
            summary = desc_module.get("briefSummary", "")
            
            # Outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            outcome_measures = []
            for outcome in outcomes_module.get("primaryOutcomes", []):
                outcome_measures.append(outcome.get("measure", ""))
            
            # Locations
            contacts_module = protocol.get("contactsLocationsModule", {})
            locations = []
            for location in contacts_module.get("locations", []):
                city = location.get("city", "")
                country = location.get("country", "")
                if city and country:
                    locations.append(f"{city}, {country}")
            
            # References (PMIDs)
            refs_module = protocol.get("referencesModule", {})
            pmids = []
            for ref in refs_module.get("references", []):
                if "pmid" in ref:
                    pmids.append(ref["pmid"])
            
            return ClinicalTrial(
                nct_id=nct_id,
                title=title,
                status=status,
                phase=phase,
                conditions=conditions,
                interventions=interventions,
                sponsor=sponsor,
                start_date=start_date,
                completion_date=completion_date,
                enrollment=enrollment,
                summary=summary,
                outcome_measures=outcome_measures,
                locations=locations[:5],
                related_pmids=pmids
            )
            
        except Exception as e:
            logger.error(f"Error parsing trial: {e}")
            return None
    
    def search_and_fetch(
        self, 
        query: str, 
        max_results: int = 10,
        status: Optional[str] = None
    ) -> List[ClinicalTrial]:
        """
        Search and fetch trials.
        
        Args:
            query: Search query
            max_results: Max trials
            status: Filter by status
            
        Returns:
            List of ClinicalTrial objects
        """
        nct_ids = self.search(query, max_results, status)
        
        trials = []
        for nct_id in nct_ids:
            trial = self.fetch_by_nct(nct_id)
            if trial:
                trials.append(trial)
        
        logger.info(f"Fetched {len(trials)} trials")
        return trials


# Example usage
if __name__ == "__main__":
    ingestor = ClinicalTrialsIngestor()
    
    # Test search
    print("\n=== Test: Search Clinical Trials ===")
    trials = ingestor.search_and_fetch(
        "KRAS inhibitor lung cancer", 
        max_results=5
    )
    
    for trial in trials:
        print(f"\n{trial.title}")
        print(f"  NCT ID: {trial.nct_id}")
        print(f"  Status: {trial.status}")
        print(f"  Phase: {trial.phase}")
        print(f"  Conditions: {', '.join(trial.conditions)}")
        print(f"  Interventions: {', '.join(trial.interventions[:3])}")
        print(f"  Sponsor: {trial.sponsor}")
        print(f"  Enrollment: {trial.enrollment}")
        print(f"  URL: https://clinicaltrials.gov/study/{trial.nct_id}")