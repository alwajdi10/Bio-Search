"""
LLM-Powered Query Understanding
Converts natural language queries to structured filters and enhanced searches.
"""

import os
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from groq import Groq
import instructor
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchIntent(BaseModel):
    """Structured representation of user search intent."""
    refined_query: str = Field(description="Refined search query optimized for vector search")
    compound_names: List[str] = Field(default_factory=list, description="Specific compound names mentioned")
    diseases: List[str] = Field(default_factory=list, description="Disease/condition names mentioned")
    targets: List[str] = Field(default_factory=list, description="Biological targets mentioned (e.g., KRAS, EGFR)")
    date_filter: Optional[str] = Field(default=None, description="Date filter like 'last 3 years' or specific year")
    search_type: str = Field(default="text", description="Type of search: 'text', 'compound', or 'both'")


class LLMQueryProcessor:
    """
    Processes natural language queries using LLM.
    Extracts search intent and generates structured filters.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM processor.
        
        Args:
            api_key: Groq API key (defaults to env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No GROQ_API_KEY found. LLM features will be disabled.")
            self.client = None
        else:
            # Initialize Groq client with instructor
            self.client = instructor.from_groq(
                Groq(api_key=self.api_key),
                mode=instructor.Mode.TOOLS
            )
            logger.info("LLM processor initialized with Groq")
    
    def process_query(self, query: str) -> SearchIntent:
        """
        Process a natural language query into structured intent.
        
        Args:
            query: User's natural language query
            
        Returns:
            SearchIntent object with extracted information
        """
        if not self.client:
            # Fallback: return query as-is
            return SearchIntent(refined_query=query)
        
        try:
            logger.info(f"Processing query: {query}")
            
            response = self.client.chat.completions.create(
                 model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a biomedical research assistant that helps users search scientific literature and chemical compounds.

Your task: Extract structured information from user queries about biological research.

Guidelines:
- refined_query: Create a clear, concise search query (2-10 words) optimized for semantic search
- compound_names: Extract specific drug/compound names mentioned
- diseases: Extract disease/condition names
- targets: Extract biological targets (proteins, genes) like KRAS, EGFR, CDK4
- date_filter: Extract time constraints (e.g., "2023", "last 3 years", "recent")
- search_type: Determine if this is a 'text' (paper), 'compound' (structure), or 'both' search

Examples:
- "KRAS inhibitors for pancreatic cancer" → refined_query: "KRAS inhibitor pancreatic cancer", targets: ["KRAS"], diseases: ["pancreatic cancer"]
- "Find papers about Sotorasib from 2023" → refined_query: "Sotorasib", compound_names: ["Sotorasib"], date_filter: "2023"
- "Recent EGFR tyrosine kinase inhibitors in lung cancer" → refined_query: "EGFR tyrosine kinase inhibitor lung cancer", targets: ["EGFR"], diseases: ["lung cancer"], date_filter: "recent"
"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                response_model=SearchIntent,
                max_tokens=500
            )
            
            logger.info(f"Extracted intent: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Fallback
            return SearchIntent(refined_query=query)
    
    def summarize_results(
        self,
        query: str,
        papers: List[Dict],
        compounds: List[Dict],
        max_length: int = 300
    ) -> str:
        """
        Generate a summary of search results using LLM.
        
        Args:
            query: Original query
            papers: List of paper results
            compounds: List of compound results
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if not self.client:
            return f"Found {len(papers)} papers and {len(compounds)} compounds."
        
        try:
            # Prepare context
            papers_text = "\n".join([
                f"- {p['title']} ({p.get('publication_date', 'unknown date')})"
                for p in papers[:5]
            ])
            
            compounds_text = "\n".join([
                f"- {c['name']} (CID: {c['cid']})"
                for c in compounds[:5]
            ])
            
            # Create prompt
            prompt = f"""Provide a concise summary of these search results for the query: "{query}"

Papers found ({len(papers)} total):
{papers_text}

Compounds found ({len(compounds)} total):
{compounds_text}

Write a 2-3 sentence summary highlighting the key findings and relationships. Be specific about compounds and therapeutic areas."""
            
            # Get completion (not using instructor for simple text)
            from groq import Groq
            groq = Groq(api_key=self.api_key)
            
            response = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a biomedical research assistant. Provide clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Found {len(papers)} papers and {len(compounds)} compounds related to {query}."
    
    def suggest_related_queries(self, query: str, num_suggestions: int = 3) -> List[str]:
        """
        Suggest related search queries.
        
        Args:
            query: Original query
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of suggested queries
        """
        if not self.client:
            return []
        
        try:
            from groq import Groq
            groq = Groq(api_key=self.api_key)
            
            prompt = f"""Given this biomedical research query: "{query}"

Suggest {num_suggestions} related queries that would help explore this topic further. Format as a JSON array of strings.

Focus on:
- Alternative therapeutic approaches
- Related biological targets
- Different disease contexts
- Mechanistic studies

Respond ONLY with a JSON array like ["query 1", "query 2", "query 3"]"""
            
            response = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a biomedical research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            # Parse JSON
            suggestions = json.loads(content)
            return suggestions[:num_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []


# Example usage
if __name__ == "__main__":
    processor = LLMQueryProcessor()
    
    # Test queries
    queries = [
        "KRAS inhibitors for pancreatic cancer from the last 3 years",
        "Find papers about Sotorasib and Adagrasib",
        "EGFR tyrosine kinase inhibitors in lung cancer",
        "Recent PROTACs targeting CDK4/6 in breast cancer"
    ]
    
    print("=== Query Processing Tests ===\n")
    
    for query in queries:
        print(f"Query: {query}")
        intent = processor.process_query(query)
        print(f"Refined: {intent.refined_query}")
        print(f"Compounds: {intent.compound_names}")
        print(f"Targets: {intent.targets}")
        print(f"Diseases: {intent.diseases}")
        print(f"Date: {intent.date_filter}")
        print(f"Type: {intent.search_type}")
        print("-" * 80 + "\n")
    
    # Test suggestions
    print("\n=== Related Query Suggestions ===\n")
    query = "KRAS G12C inhibitors"
    suggestions = processor.suggest_related_queries(query)
    print(f"Original: {query}")
    print(f"Suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")