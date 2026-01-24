"""
Embedding Generator
Converts text and chemical structures to vector embeddings.
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text and chemical structures.
    - Text: Uses sentence-transformers
    - Chemicals: Uses RDKit Morgan fingerprints
    """
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize embedding generators.
        
        Args:
            text_model: HuggingFace model name for text embeddings
            device: 'cuda', 'cpu', or None for auto-detect
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load text embedding model
        logger.info(f"Loading text model: {text_model}")
        self.text_model = SentenceTransformer(text_model, device=self.device)
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        
        # Chemical fingerprint parameters
        self.fp_radius = 2
        self.fp_nbits = 2048
        
        logger.info(f"Text embedding dimension: {self.text_dim}")
        logger.info(f"Chemical fingerprint dimension: {self.fp_nbits}")
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            numpy array of shape (n, text_dim) or (text_dim,) for single text
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # Generate embeddings
        embeddings = self.text_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10
        )
        
        if return_single:
            return embeddings[0]
        return embeddings
    
    def embed_abstract(self, title: str, abstract: str) -> np.ndarray:
        """
        Embed a paper by combining title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Embedding vector
        """
        # Combine with title weighted more heavily
        combined_text = f"{title}. {title}. {abstract}"
        return self.embed_text(combined_text)
    
    def embed_smiles(self, smiles: Union[str, List[str]]) -> Optional[np.ndarray]:
        """
        Generate Morgan fingerprint embeddings for SMILES string(s).
        
        Args:
            smiles: Single SMILES string or list of SMILES
            
        Returns:
            numpy array of shape (n, fp_nbits) or (fp_nbits,) for single SMILES
            Returns None if SMILES is invalid
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
            return_single = True
        else:
            smiles_list = smiles
            return_single = False
        
        fingerprints = []
        
        for smi in smiles_list:
            fp = self._smiles_to_fingerprint(smi)
            if fp is not None:
                fingerprints.append(fp)
            else:
                logger.warning(f"Invalid SMILES: {smi}")
                # Return zero vector for invalid SMILES
                fingerprints.append(np.zeros(self.fp_nbits, dtype=np.float32))
        
        fingerprints = np.array(fingerprints, dtype=np.float32)
        
        if return_single:
            return fingerprints[0]
        return fingerprints
    
    def _smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert a single SMILES string to Morgan fingerprint.
        
        Args:
            smiles: SMILES string
            
        Returns:
            numpy array of fingerprint bits as floats, or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.fp_radius,
                nBits=self.fp_nbits
            )
            
            # Convert to numpy array
            arr = np.zeros((self.fp_nbits,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
            
        except Exception as e:
            logger.error(f"Error converting SMILES to fingerprint: {e}")
            return None
    
    def batch_embed_papers(
        self,
        papers: List[dict],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Batch embed multiple papers efficiently.
        
        Args:
            papers: List of paper dicts with 'title' and 'abstract'
            batch_size: Number of papers to process at once
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            # Combine titles and abstracts
            texts = [
                f"{p['title']}. {p['title']}. {p['abstract']}" 
                for p in batch
            ]
            
            # Generate embeddings for batch
            batch_embeddings = self.embed_text(texts)
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Embedded papers {i+1}-{min(i+batch_size, len(papers))}/{len(papers)}")
        
        return embeddings
    
    def batch_embed_compounds(
        self,
        compounds: List[dict],
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Batch embed multiple compounds.
        
        Args:
            compounds: List of compound dicts with 'canonical_smiles'
            batch_size: Number of compounds to process at once
            
        Returns:
            List of fingerprint vectors
        """
        embeddings = []
        
        for i in range(0, len(compounds), batch_size):
            batch = compounds[i:i+batch_size]
            
            # Extract SMILES
            smiles_list = [c['canonical_smiles'] for c in batch]
            
            # Generate fingerprints
            batch_fps = self.embed_smiles(smiles_list)
            embeddings.extend(batch_fps)
            
            logger.info(f"Embedded compounds {i+1}-{min(i+batch_size, len(compounds))}/{len(compounds)}")
        
        return embeddings
    
    def similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: 'cosine' or 'euclidean'
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        elif metric == 'euclidean':
            return -np.linalg.norm(vec1 - vec2)
        else:
            raise ValueError(f"Unknown metric: {metric}")


# Example usage
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Test text embedding
    print("\n=== Text Embedding Test ===")
    text = "KRAS mutations are found in pancreatic cancer"
    embedding = generator.embed_text(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:10]}")
    
    # Test SMILES embedding
    print("\n=== SMILES Embedding Test ===")
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    fp = generator.embed_smiles(smiles)
    print(f"SMILES: {smiles}")
    print(f"Fingerprint shape: {fp.shape}")
    print(f"Fingerprint bits set: {np.sum(fp)}")
    
    # Test similarity
    print("\n=== Similarity Test ===")
    text1 = "KRAS inhibitor for cancer treatment"
    text2 = "KRAS G12C mutation in lung cancer"
    text3 = "Aspirin for pain relief"
    
    emb1 = generator.embed_text(text1)
    emb2 = generator.embed_text(text2)
    emb3 = generator.embed_text(text3)
    
    sim_12 = generator.similarity(emb1, emb2)
    sim_13 = generator.similarity(emb1, emb3)
    
    print(f"Similarity (KRAS texts): {sim_12:.3f}")
    print(f"Similarity (KRAS vs Aspirin): {sim_13:.3f}")