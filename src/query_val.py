"""
Query Validator
Ensures queries are within biological/biomedical scope.
"""

from typing import Tuple
from groq import Groq
import os
import logging

logger = logging.getLogger(__name__)


class BiologicalQueryValidator:
    """Validates that queries are biology/medicine related."""
    
    BIOLOGICAL_KEYWORDS = {
        # Core biology
        "gene", "protein", "dna", "rna", "cell", "molecule", "enzyme",
        "mutation", "sequence", "genome", "chromosome", "allele",
        # Medicine/disease
        "cancer", "disease", "tumor", "therapy", "treatment", "drug",
        "clinical", "patient", "diagnosis", "symptom", "syndrome",
        "infection", "virus", "bacteria", "pathogen",
        # Compounds
        "compound", "chemical", "inhibitor", "agonist", "antagonist",
        "pharmaceutical", "medication", "molecule",
        # Research
        "trial", "study", "research", "pubmed", "publication",
        # Specific terms
        "kras", "egfr", "antibody", "receptor", "pathway", "kinase"
    }
    
    def __init__(self):
        """Initialize validator with LLM for nuanced checking."""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        else:
            self.groq_client = None
            logger.warning("No GROQ_API_KEY - using keyword-only validation")
    
    def is_biological_query(self, query: str) -> Tuple[bool, str]:
        """
        Check if query is biology/medicine related.
        
        Args:
            query: User's query
            
        Returns:
            (is_valid, reason)
        """
        query_lower = query.lower()
        
        # Quick keyword check first
        has_bio_keyword = any(kw in query_lower for kw in self.BIOLOGICAL_KEYWORDS)
        
        if has_bio_keyword:
            return True, "Contains biological keywords"
        
        # If no keywords, use LLM for nuanced check
        if self.groq_client:
            return self._llm_validate(query)
        else:
            # Strict fallback without LLM
            return False, "No biological keywords found"
    
    def _llm_validate(self, query: str) -> Tuple[bool, str]:
        """Use LLM to validate query scope."""
        
        prompt = f"""Is this query related to biology, medicine, biochemistry, or biomedical research?

Query: "{query}"

Respond with ONLY "YES" or "NO" followed by a brief reason.

Examples:
- "What are KRAS inhibitors?" → YES - asks about biological molecules
- "What is the weather?" → NO - not biology related
- "How does aspirin work?" → YES - asks about drug mechanism
- "What is the capital of France?" → NO - geography question
- "Tell me about cancer treatments" → YES - medical topic

Your response (YES/NO + reason):"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a strict classifier. Only respond with YES or NO and a brief reason."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Parse response
            if answer.upper().startswith("YES"):
                return True, answer
            else:
                return False, answer
                
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            # Fail closed - reject if we can't validate
            return False, "Could not validate query scope"


# Quick test
if __name__ == "__main__":
    validator = BiologicalQueryValidator()
    
    test_queries = [
        "What are KRAS inhibitors?",
        "What is the weather today?",
        "Tell me about cancer",
        "Who won the Super Bowl?",
        "How does aspirin work?",
        "What is the capital of France?",
        "EGFR mutations in lung cancer"
    ]
    
    print("Testing Query Validator:\n")
    for query in test_queries:
        is_valid, reason = validator.is_biological_query(query)
        status = "✅" if is_valid else "❌"
        print(f"{status} '{query}'")
        print(f"   → {reason}\n")