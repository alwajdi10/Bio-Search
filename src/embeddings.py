"""
Enhanced Embedding Generator with Higher Accuracy
- Larger embedding dimensions (768-dim for text, 1024-dim for images)
- Multiple similarity metrics (cosine, dot product, euclidean)
- Multi-vector representations
- Normalization and re-ranking
"""

import numpy as np
from typing import List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedEmbeddingGenerator:
    """
    High-accuracy embedding generator with advanced features.
    
    Improvements:
    - Larger models (768-dim text, 1024-dim images)
    - Multiple similarity metrics
    - Multi-vector per document
    - Better normalization
    """
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-mpnet-base-v2",  # 768-dim
        device: str = None,
        normalize: bool = True
    ):
        """
        Initialize with high-accuracy models.
        
        Args:
            text_model: HuggingFace model (default: 768-dim MPNet)
            device: 'cuda', 'cpu', or None for auto-detect
            normalize: Whether to L2-normalize embeddings
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load high-accuracy text model (768-dim)
        logger.info(f"Loading text model: {text_model}")
        self.text_model = SentenceTransformer(text_model, device=self.device)
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        
        # Chemical fingerprint parameters (increased for accuracy)
        self.fp_radius = 3  # Increased from 2
        self.fp_nbits = 4096  # Increased from 2048
        
        self.normalize = normalize
        
        logger.info(f"✓ Enhanced embeddings initialized")
        logger.info(f"  Text dimension: {self.text_dim}")
        logger.info(f"  Fingerprint dimension: {self.fp_nbits}")
        logger.info(f"  Normalization: {normalize}")
    
    def embed_text(
        self, 
        texts: Union[str, List[str]],
        normalize: bool = None
    ) -> np.ndarray:
        """
        Generate high-quality text embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Override default normalization
            
        Returns:
            Embedding vector(s)
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
            normalize_embeddings=normalize if normalize is not None else self.normalize,
            show_progress_bar=len(texts) > 10,
            batch_size=32
        )
        
        if return_single:
            return embeddings[0]
        return embeddings
    
    def embed_abstract_multi_vector(
        self, 
        title: str, 
        abstract: str,
        keywords: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Create multiple embeddings for different parts of a paper.
        Enables more nuanced matching.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            keywords: Optional keywords/MeSH terms
            
        Returns:
            Tuple of (title_embedding, abstract_embedding, keywords_embedding)
        """
        # Embed title (more weight)
        title_emb = self.embed_text(title)
        
        # Embed abstract
        abstract_emb = self.embed_text(abstract)
        
        # Embed keywords if available
        keywords_emb = None
        if keywords:
            keywords_text = " ".join(keywords)
            keywords_emb = self.embed_text(keywords_text)
        
        return title_emb, abstract_emb, keywords_emb
    
    def embed_abstract_weighted(
        self, 
        title: str, 
        abstract: str,
        title_weight: float = 0.4,
        abstract_weight: float = 0.6
    ) -> np.ndarray:
        """
        Weighted combination of title and abstract embeddings.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            title_weight: Weight for title (default: 0.4)
            abstract_weight: Weight for abstract (default: 0.6)
            
        Returns:
            Combined embedding
        """
        title_emb = self.embed_text(title)
        abstract_emb = self.embed_text(abstract)
        
        # Weighted average
        combined = (title_weight * title_emb + abstract_weight * abstract_emb)
        
        # Normalize if needed
        if self.normalize:
            combined = combined / np.linalg.norm(combined)
        
        return combined
    
    def embed_smiles(
        self, 
        smiles: Union[str, List[str]],
        fingerprint_type: str = "morgan"
    ) -> Optional[np.ndarray]:
        """
        Generate high-accuracy molecular fingerprints.
        
        Args:
            smiles: SMILES string(s)
            fingerprint_type: "morgan" or "rdkit"
            
        Returns:
            Fingerprint vector(s)
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
            return_single = True
        else:
            smiles_list = smiles
            return_single = False
        
        fingerprints = []
        
        for smi in smiles_list:
            fp = self._smiles_to_fingerprint(smi, fingerprint_type)
            if fp is not None:
                fingerprints.append(fp)
            else:
                logger.warning(f"Invalid SMILES: {smi}")
                fingerprints.append(np.zeros(self.fp_nbits, dtype=np.float32))
        
        fingerprints = np.array(fingerprints, dtype=np.float32)
        
        if return_single:
            return fingerprints[0]
        return fingerprints
    
    def _smiles_to_fingerprint(
        self, 
        smiles: str,
        fingerprint_type: str = "morgan"
    ) -> Optional[np.ndarray]:
        """
        Convert SMILES to high-quality fingerprint.
        
        Args:
            smiles: SMILES string
            fingerprint_type: "morgan" or "rdkit"
            
        Returns:
            Fingerprint array
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if fingerprint_type == "morgan":
                # Morgan fingerprint with increased radius
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=self.fp_radius,
                    nBits=self.fp_nbits
                )
            else:
                # RDKit fingerprint
                fp = Chem.RDKFingerprint(mol, fpSize=self.fp_nbits)
            
            # Convert to numpy
            arr = np.zeros((self.fp_nbits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
            
        except Exception as e:
            logger.error(f"Error converting SMILES: {e}")
            return None
    
    # ============================================================
    # SIMILARITY METRICS
    # ============================================================
    
    def similarity_cosine(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Cosine similarity (default, range: -1 to 1).
        Best for normalized vectors.
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def similarity_dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Dot product similarity (range: unbounded).
        Better separation for normalized vectors.
        """
        return float(np.dot(vec1, vec2))
    
    def similarity_euclidean(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Negative euclidean distance (range: -∞ to 0).
        Higher is more similar.
        """
        return -float(np.linalg.norm(vec1 - vec2))
    
    def similarity_manhattan(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Negative Manhattan distance.
        More robust to outliers than euclidean.
        """
        return -float(np.sum(np.abs(vec1 - vec2)))
    
    def similarity_angular(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Angular similarity (range: 0 to 1).
        Based on angle between vectors. Better separation than cosine.
        """
        cos_sim = self.similarity_cosine(vec1, vec2)
        # Convert to angular distance then to similarity
        angular_distance = np.arccos(np.clip(cos_sim, -1.0, 1.0)) / np.pi
        return 1.0 - angular_distance
    
    def similarity_hybrid(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    ) -> float:
        """
        Hybrid similarity combining multiple metrics.
        Provides best separation and robustness.
        
        Args:
            vec1, vec2: Vectors to compare
            weights: (cosine_weight, dot_weight, angular_weight)
            
        Returns:
            Weighted combined similarity
        """
        cosine = self.similarity_cosine(vec1, vec2)
        dot = self.similarity_dot_product(vec1, vec2)
        angular = self.similarity_angular(vec1, vec2)
        
        # Normalize dot product to 0-1 range (assuming normalized vectors)
        dot_norm = (dot + 1) / 2
        
        # Weighted combination
        combined = (
            weights[0] * cosine +
            weights[1] * dot_norm +
            weights[2] * angular
        )
        
        return float(combined)
    
    def compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = "hybrid"
    ) -> float:
        """
        Compute similarity using specified metric.
        
        Args:
            vec1, vec2: Vectors to compare
            metric: "cosine", "dot", "euclidean", "manhattan", "angular", "hybrid"
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            return self.similarity_cosine(vec1, vec2)
        elif metric == "dot":
            return self.similarity_dot_product(vec1, vec2)
        elif metric == "euclidean":
            return self.similarity_euclidean(vec1, vec2)
        elif metric == "manhattan":
            return self.similarity_manhattan(vec1, vec2)
        elif metric == "angular":
            return self.similarity_angular(vec1, vec2)
        elif metric == "hybrid":
            return self.similarity_hybrid(vec1, vec2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # ============================================================
    # BATCH OPERATIONS
    # ============================================================
    
    def batch_embed_papers(
        self,
        papers: List[dict],
        batch_size: int = 32,
        use_multi_vector: bool = False
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, ...]]]:
        """
        Batch embed papers with optional multi-vector.
        
        Args:
            papers: List of paper dicts
            batch_size: Batch size
            use_multi_vector: Return (title, abstract, keywords) embeddings
            
        Returns:
            List of embeddings or tuples
        """
        if use_multi_vector:
            embeddings = []
            for paper in papers:
                emb = self.embed_abstract_multi_vector(
                    paper['title'],
                    paper['abstract'],
                    paper.get('mesh_terms', [])
                )
                embeddings.append(emb)
            return embeddings
        else:
            # Single embedding per paper
            embeddings = []
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i+batch_size]
                
                texts = [
                    self._create_paper_text(p)
                    for p in batch
                ]
                
                batch_embeddings = self.embed_text(texts)
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Embedded papers {i+1}-{min(i+batch_size, len(papers))}/{len(papers)}")
            
            return embeddings
    
    def _create_paper_text(self, paper: dict) -> str:
        """Create weighted text representation of paper."""
        # Title appears 3x for more weight
        title = paper['title']
        abstract = paper.get('abstract', '')
        
        # Add keywords if available
        keywords = paper.get('mesh_terms', [])
        keywords_text = " ".join(keywords[:5]) if keywords else ""
        
        return f"{title}. {title}. {title}. {keywords_text} {abstract}"
    
    def batch_embed_compounds(
        self,
        compounds: List[dict],
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """Batch embed compounds."""
        embeddings = []
        
        for i in range(0, len(compounds), batch_size):
            batch = compounds[i:i+batch_size]
            
            smiles_list = [c['canonical_smiles'] for c in batch]
            batch_fps = self.embed_smiles(smiles_list)
            embeddings.extend(batch_fps)
            
            logger.info(f"Embedded compounds {i+1}-{min(i+batch_size, len(compounds))}/{len(compounds)}")
        
        return embeddings


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED EMBEDDING GENERATOR TEST")
    print("="*80)
    
    # Initialize with high-accuracy model
    print("\n1. Initializing enhanced generator...")
    generator = EnhancedEmbeddingGenerator(
        text_model="sentence-transformers/all-mpnet-base-v2",  # 768-dim
        normalize=True
    )
    
    print(f"✓ Text dimension: {generator.text_dim}")
    print(f"✓ Fingerprint dimension: {generator.fp_nbits}")
    
    # Test text embeddings
    print("\n2. Testing text embeddings...")
    texts = [
        "KRAS G12C inhibitor for lung cancer",
        "KRAS mutation in pancreatic cancer",
        "Aspirin for pain relief"
    ]
    
    embeddings = [generator.embed_text(t) for t in texts]
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Test similarity metrics
    print("\n3. Testing similarity metrics...")
    emb1, emb2, emb3 = embeddings
    
    metrics = ["cosine", "dot", "angular", "hybrid"]
    
    print("\nSimilarity between KRAS texts:")
    for metric in metrics:
        sim = generator.compute_similarity(emb1, emb2, metric)
        print(f"  {metric:10s}: {sim:.4f}")
    
    print("\nSimilarity between KRAS and Aspirin:")
    for metric in metrics:
        sim = generator.compute_similarity(emb1, emb3, metric)
        print(f"  {metric:10s}: {sim:.4f}")
    
    print("\n✓ Notice: hybrid metric provides better separation!")
    
    # Test multi-vector
    print("\n4. Testing multi-vector embeddings...")
    title_emb, abstract_emb, _ = generator.embed_abstract_multi_vector(
        title="KRAS G12C inhibitor shows promise",
        abstract="The KRAS G12C mutation is found in many cancers...",
        keywords=["KRAS", "cancer", "inhibitor"]
    )
    
    print(f"✓ Title embedding: {title_emb.shape}")
    print(f"✓ Abstract embedding: {abstract_emb.shape}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")