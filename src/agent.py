"""
AI Research Agent
Orchestrates multi-modal search and LLM reasoning for biological research queries.
Responds intelligently even when data isn't in the vector database.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from groq import Groq
import instructor
from pathlib import Path
import sys

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search import CloudSearch
from app.llm_filter import LLMQueryProcessor, SearchIntent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Structured response from the agent."""
    answer: str = Field(description="Main answer to the user's query")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used (papers, compounds, etc)")
    related_queries: List[str] = Field(default_factory=list, description="Suggested follow-up queries")
    data_found: bool = Field(default=False, description="Whether data was found in vector DB")
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning process")


class BiologicalResearchAgent:
    """
    Intelligent agent for biological research queries.
    
    Features:
    - Searches vector database (Qdrant) for relevant data
    - Uses LLM (Groq) to answer queries even without DB data
    - Provides source links and citations
    - Suggests related queries
    - Handles multi-modal data (papers, compounds, proteins, genes, trials)
    """
    
    def __init__(self):
        """Initialize the agent."""
        # Initialize search engine
        try:
            self.search_engine = CloudSearch()
            logger.info("✓ Connected to Qdrant Cloud")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            self.search_engine = None
        
        # Initialize LLM processor
        self.llm_processor = LLMQueryProcessor()
        
        # Initialize Groq client
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.groq_client = Groq(api_key=groq_key)
        self.instructor_client = instructor.from_groq(
            self.groq_client,
            mode=instructor.Mode.TOOLS
        )
        
        logger.info("✓ Agent initialized")
    
    def query(self, user_query: str, search_limit: int = 10) -> AgentResponse:
        """
        Process a user query and return intelligent response.
        
        Args:
            user_query: Natural language query from user
            search_limit: Number of results to retrieve from vector DB
            
        Returns:
            AgentResponse with answer, sources, and suggestions
        """
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Extract search intent using LLM
        intent = self.llm_processor.process_query(user_query)
        logger.info(f"Extracted intent: {intent.refined_query}")
        
        # Step 2: Search vector database
        db_results = None
        if self.search_engine:
            try:
                db_results = self.search_engine.search_by_text(
                    intent.refined_query, 
                    limit=search_limit
                )
                logger.info(f"Found {len(db_results.get('papers', []))} papers, "
                          f"{len(db_results.get('compounds', []))} compounds")
            except Exception as e:
                logger.error(f"Search error: {e}")
                db_results = None
        
        # Step 3: Generate response based on available data
        if db_results and (db_results.get('papers') or db_results.get('compounds')):
            # We have data - generate answer with citations
            response = self._generate_response_with_data(
                user_query, 
                intent, 
                db_results
            )
        else:
            # No data - use LLM general knowledge
            response = self._generate_response_without_data(
                user_query, 
                intent
            )
        
        # Step 4: Add related query suggestions
        try:
            suggestions = self.llm_processor.suggest_related_queries(user_query, num_suggestions=3)
            response.related_queries = suggestions
        except Exception as e:
            logger.warning(f"Could not generate suggestions: {e}")
        
        return response
    
    def _generate_response_with_data(
        self, 
        query: str, 
        intent: SearchIntent,
        db_results: Dict
    ) -> AgentResponse:
        """Generate response when vector DB has relevant data."""
        
        papers = db_results.get('papers', [])
        compounds = db_results.get('compounds', [])
        
        # Build context from search results
        context = self._build_context(papers, compounds)
        
        # Generate answer using LLM + context
        prompt = f"""You are a biomedical research assistant. Answer the user's query using the provided research data.

User Query: {query}

Available Data:
{context}

Instructions:
1. Provide a comprehensive answer based on the data
2. Cite specific papers using [PMID: XXXXX] format
3. Mention specific compounds by name and CID
4. If data is limited, acknowledge this
5. Keep response clear and scientifically accurate
6. Include PubMed links: https://pubmed.ncbi.nlm.nih.gov/PMID
7. Include PubChem links: https://pubchem.ncbi.nlm.nih.gov/compound/CID

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Build sources list
            sources = []
            
            for paper in papers[:5]:
                sources.append({
                    "type": "paper",
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}",
                    "relevance": paper.get("score", 0)
                })
            
            for compound in compounds[:5]:
                sources.append({
                    "type": "compound",
                    "cid": compound["cid"],
                    "name": compound["name"],
                    "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound['cid']}",
                    "formula": compound.get("molecular_formula", "")
                })
            
            return AgentResponse(
                answer=answer,
                sources=sources,
                data_found=True,
                reasoning=f"Found {len(papers)} relevant papers and {len(compounds)} compounds in database"
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback
            return AgentResponse(
                answer=f"Found {len(papers)} relevant papers and {len(compounds)} compounds. "
                       f"Error generating detailed response: {e}",
                sources=[],
                data_found=True
            )
    
    def _generate_response_without_data(
        self, 
        query: str, 
        intent: SearchIntent
    ) -> AgentResponse:
        """Generate response when no vector DB data is available."""
        
        prompt = f"""You are a biomedical research assistant. The user asked: "{query}"

Our database doesn't contain specific data for this query, but you can provide helpful information using your general knowledge.

Instructions:
1. Acknowledge that you don't have specific papers/compounds in the database
2. Provide accurate general information about the topic
3. Explain what type of research exists in this area
4. Suggest what the user should look for
5. Recommend authoritative sources (PubMed, ClinicalTrials.gov, etc.)
6. Keep response factual and helpful

Response:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert biomedical research assistant with broad knowledge."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.5
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add helpful external links
            sources = [
                {
                    "type": "external",
                    "name": "PubMed Search",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/?term={intent.refined_query.replace(' ', '+')}",
                    "description": "Search all of PubMed"
                },
                {
                    "type": "external", 
                    "name": "ClinicalTrials.gov",
                    "url": f"https://clinicaltrials.gov/search?term={intent.refined_query.replace(' ', '+')}",
                    "description": "Search clinical trials"
                }
            ]
            
            return AgentResponse(
                answer=answer,
                sources=sources,
                data_found=False,
                reasoning="No data in local database - using general biomedical knowledge"
            )
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return AgentResponse(
                answer="I don't have specific data for this query in my database. "
                       "Please try searching PubMed or ClinicalTrials.gov directly.",
                sources=[],
                data_found=False
            )
    
    def _build_context(self, papers: List[Dict], compounds: List[Dict]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        if papers:
            context_parts.append("=== PAPERS ===")
            for i, paper in enumerate(papers[:5], 1):
                context_parts.append(
                    f"{i}. [PMID: {paper['pmid']}] {paper['title']}\n"
                    f"   Journal: {paper.get('journal', 'Unknown')}\n"
                    f"   Date: {paper.get('publication_date', 'Unknown')}\n"
                    f"   Abstract: {paper.get('abstract', '')[:300]}...\n"
                )
        
        if compounds:
            context_parts.append("\n=== COMPOUNDS ===")
            for i, compound in enumerate(compounds[:5], 1):
                context_parts.append(
                    f"{i}. {compound['name']} (CID: {compound['cid']})\n"
                    f"   Formula: {compound.get('molecular_formula', 'Unknown')}\n"
                    f"   Mentioned in PMIDs: {compound.get('source_pmids', [])}\n"
                )
        
        return "\n".join(context_parts)
    
    def chat(self, message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Conversational interface for the agent.
        Maintains context across multiple turns.
        
        Args:
            message: User's message
            conversation_history: Previous messages in format [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Dict with response and updated history
        """
        if conversation_history is None:
            conversation_history = []
        
        # Get agent response
        response = self.query(message)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response.answer})
        
        return {
            "response": response,
            "history": conversation_history
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("BIOLOGICAL RESEARCH AGENT - TEST")
    print("="*80)
    
    agent = BiologicalResearchAgent()
    
    # Test queries
    test_queries = [
        "What are KRAS inhibitors for lung cancer?",
        "Tell me about Sotorasib",
        "What are the latest treatments for pancreatic cancer?",  # May not be in DB
        "How do CDK4/6 inhibitors work in breast cancer?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        response = agent.query(query)
        
        print(f"\n📊 Data Found: {response.data_found}")
        print(f"\n💡 Answer:\n{response.answer}")
        
        if response.sources:
            print(f"\n📚 Sources ({len(response.sources)}):")
            for source in response.sources[:3]:
                if source["type"] == "paper":
                    print(f"  • Paper: {source['title'][:60]}...")
                    print(f"    {source['url']}")
                elif source["type"] == "compound":
                    print(f"  • Compound: {source['name']} ({source['formula']})")
                    print(f"    {source['url']}")
                elif source["type"] == "external":
                    print(f"  • {source['name']}: {source['url']}")
        
        if response.related_queries:
            print(f"\n🔍 Related Queries:")
            for rq in response.related_queries:
                print(f"  • {rq}")
        
        print()