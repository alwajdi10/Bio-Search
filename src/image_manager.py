"""
Image Manager for Biological Research
Handles multiple image modalities: structures, proteins, microscopy, diagrams.
"""

import os
import requests
import io
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioImage(BaseModel):
    """Unified model for biological images."""
    image_id: str
    image_type: str  # "structure_2d", "structure_3d", "microscopy", "pathway", "figure"
    source: str  # "pubchem", "pdb", "paper", "kegg"
    source_id: str  # CID, PDB ID, PMID, etc.
    url: str
    local_path: Optional[str] = None
    caption: str = ""
    width: int = 0
    height: int = 0
    format: str = ""
    metadata: Dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None  # CLIP embedding for search


class ImageManager:
    """
    Manages biological images from multiple sources.
    Downloads, stores, and enables visual search.
    """
    
    def __init__(self, cache_dir: str = "data/images"):
        """
        Initialize image manager.
        
        Args:
            cache_dir: Directory to cache downloaded images
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "structures_2d").mkdir(exist_ok=True)
        (self.cache_dir / "structures_3d").mkdir(exist_ok=True)
        (self.cache_dir / "proteins").mkdir(exist_ok=True)
        (self.cache_dir / "pathways").mkdir(exist_ok=True)
        (self.cache_dir / "figures").mkdir(exist_ok=True)
        
        logger.info(f"ImageManager initialized. Cache: {self.cache_dir}")
    
    # ============================================================
    # 1. COMPOUND 2D STRUCTURES
    # ============================================================
    
    def get_compound_2d_structure(
        self, 
        cid: int, 
        size: str = "large"
    ) -> Optional[BioImage]:
        """
        Get 2D structure image from PubChem.
        
        Args:
            cid: PubChem Compound ID
            size: 'small' (300x300) or 'large' (600x600)
            
        Returns:
            BioImage object
        """
        try:
            # PubChem image URL
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
            if size == "large":
                url += "?image_size=large"
            
            logger.info(f"Downloading 2D structure for CID {cid}")
            
            # Download image
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to download image for CID {cid}")
                return None
            
            # Save to cache
            image_id = f"compound_2d_{cid}"
            filename = f"{image_id}.png"
            filepath = self.cache_dir / "structures_2d" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Get image dimensions
            img = Image.open(filepath)
            width, height = img.size
            
            return BioImage(
                image_id=image_id,
                image_type="structure_2d",
                source="pubchem",
                source_id=str(cid),
                url=url,
                local_path=str(filepath),
                caption=f"2D structure of compound CID {cid}",
                width=width,
                height=height,
                format="PNG",
                metadata={"cid": cid, "size": size}
            )
            
        except Exception as e:
            logger.error(f"Error fetching 2D structure: {e}")
            return None
    
    # ============================================================
    # 2. PROTEIN 3D STRUCTURES
    # ============================================================
    
    def get_protein_3d_structure(
        self, 
        pdb_id: str,
        representation: str = "cartoon"
    ) -> Optional[BioImage]:
        """
        Get 3D protein structure image from RCSB PDB.
        
        Args:
            pdb_id: PDB identifier (e.g., "6XM0")
            representation: "cartoon", "surface", "ball-stick"
            
        Returns:
            BioImage object
        """
        try:
            # RCSB PDB image URL
            url = f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg"
            
            logger.info(f"Downloading 3D structure for PDB {pdb_id}")
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to download image for PDB {pdb_id}")
                return None
            
            # Save to cache
            image_id = f"protein_3d_{pdb_id.lower()}"
            filename = f"{image_id}.jpeg"
            filepath = self.cache_dir / "structures_3d" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            img = Image.open(filepath)
            width, height = img.size
            
            return BioImage(
                image_id=image_id,
                image_type="structure_3d",
                source="pdb",
                source_id=pdb_id.upper(),
                url=url,
                local_path=str(filepath),
                caption=f"3D structure of protein {pdb_id}",
                width=width,
                height=height,
                format="JPEG",
                metadata={"pdb_id": pdb_id, "representation": representation}
            )
            
        except Exception as e:
            logger.error(f"Error fetching 3D structure: {e}")
            return None
    
    # ============================================================
    # 3. PATHWAY DIAGRAMS
    # ============================================================
    
    def get_pathway_diagram(
        self, 
        pathway_id: str,
        organism: str = "hsa"
    ) -> Optional[BioImage]:
        """
        Get pathway diagram from KEGG.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "04010" for MAPK)
            organism: Organism code (hsa=human, mmu=mouse)
            
        Returns:
            BioImage object
        """
        try:
            # KEGG pathway image URL
            full_id = f"{organism}{pathway_id}"
            url = f"https://rest.kegg.jp/get/{full_id}/image"
            
            logger.info(f"Downloading pathway diagram for {full_id}")
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to download pathway {full_id}")
                return None
            
            # Save to cache
            image_id = f"pathway_{full_id}"
            filename = f"{image_id}.png"
            filepath = self.cache_dir / "pathways" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            img = Image.open(filepath)
            width, height = img.size
            
            return BioImage(
                image_id=image_id,
                image_type="pathway",
                source="kegg",
                source_id=full_id,
                url=url,
                local_path=str(filepath),
                caption=f"KEGG pathway {full_id}",
                width=width,
                height=height,
                format="PNG",
                metadata={"pathway_id": pathway_id, "organism": organism}
            )
            
        except Exception as e:
            logger.error(f"Error fetching pathway diagram: {e}")
            return None
    
    # ============================================================
    # 4. BATCH OPERATIONS
    # ============================================================
    
    def batch_download_compound_structures(
        self, 
        cids: List[int],
        size: str = "large"
    ) -> List[BioImage]:
        """
        Download 2D structures for multiple compounds.
        
        Args:
            cids: List of PubChem CIDs
            size: Image size
            
        Returns:
            List of BioImage objects
        """
        images = []
        
        for cid in cids:
            img = self.get_compound_2d_structure(cid, size)
            if img:
                images.append(img)
        
        logger.info(f"Downloaded {len(images)}/{len(cids)} compound structures")
        return images
    
    def batch_download_protein_structures(
        self, 
        pdb_ids: List[str]
    ) -> List[BioImage]:
        """Download 3D structures for multiple proteins."""
        images = []
        
        for pdb_id in pdb_ids:
            img = self.get_protein_3d_structure(pdb_id)
            if img:
                images.append(img)
        
        logger.info(f"Downloaded {len(images)}/{len(pdb_ids)} protein structures")
        return images
    
    # ============================================================
    # 5. IMAGE UTILITIES
    # ============================================================
    
    def get_image_as_base64(self, image: BioImage) -> str:
        """Convert image to base64 for web display."""
        if not image.local_path or not Path(image.local_path).exists():
            return ""
        
        with open(image.local_path, 'rb') as f:
            img_data = f.read()
            b64 = base64.b64encode(img_data).decode()
            return f"data:image/{image.format.lower()};base64,{b64}"
    
    def resize_image(
        self, 
        image: BioImage, 
        max_size: tuple = (800, 800)
    ) -> BioImage:
        """Resize image while maintaining aspect ratio."""
        if not image.local_path:
            return image
        
        img = Image.open(image.local_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save resized version
        resized_path = Path(image.local_path).with_stem(
            Path(image.local_path).stem + "_resized"
        )
        img.save(resized_path)
        
        image.local_path = str(resized_path)
        image.width, image.height = img.size
        
        return image
    
    def create_comparison_grid(
        self, 
        images: List[BioImage],
        cols: int = 3,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a grid comparison of multiple images.
        
        Args:
            images: List of BioImage objects
            cols: Number of columns
            output_path: Where to save grid (optional)
            
        Returns:
            Path to grid image
        """
        if not images:
            return ""
        
        # Load images
        pil_images = []
        for bio_img in images:
            if bio_img.local_path and Path(bio_img.local_path).exists():
                pil_images.append(Image.open(bio_img.local_path))
        
        if not pil_images:
            return ""
        
        # Calculate grid dimensions
        rows = (len(pil_images) + cols - 1) // cols
        
        # Get max dimensions
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        
        # Create grid
        grid_width = cols * max_width
        grid_height = rows * max_height
        grid = Image.new('RGB', (grid_width, grid_height), color='white')
        
        for idx, img in enumerate(pil_images):
            row = idx // cols
            col = idx % cols
            x = col * max_width
            y = row * max_height
            grid.paste(img, (x, y))
        
        # Save grid
        if not output_path:
            output_path = self.cache_dir / "comparison_grid.png"
        
        grid.save(output_path)
        logger.info(f"Created comparison grid: {output_path}")
        
        return str(output_path)
    
    # ============================================================
    # 6. SEARCH FUNCTIONS (for image-based queries)
    # ============================================================
    
    def get_all_cached_images(self) -> List[BioImage]:
        """Get list of all cached images."""
        images = []
        
        # Scan cache directory
        for subdir in ["structures_2d", "structures_3d", "pathways", "figures"]:
            path = self.cache_dir / subdir
            if path.exists():
                # Combine glob results properly
                img_files = list(path.glob("*.png")) + list(path.glob("*.jpeg")) + list(path.glob("*.jpg"))
                for img_file in img_files:
                    # Parse filename to get metadata
                    parts = img_file.stem.split("_")
                    if len(parts) >= 2:
                        image_type = "_".join(parts[:-1])
                        source_id = parts[-1]
                        
                        img = Image.open(img_file)
                        width, height = img.size
                        
                        images.append(BioImage(
                            image_id=img_file.stem,
                            image_type=image_type,
                            source=subdir,
                            source_id=source_id,
                            url="",
                            local_path=str(img_file),
                            width=width,
                            height=height,
                            format=img_file.suffix[1:].upper()
                        ))
        
        return images
    
    def save_metadata(self, images: List[BioImage], output_file: str = "image_metadata.json"):
        """Save image metadata to JSON."""
        import json
        
        metadata = [img.model_dump(exclude={'embedding'}) for img in images]
        
        output_path = self.cache_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata for {len(images)} images to {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("IMAGE MANAGER TEST")
    print("="*80)
    
    manager = ImageManager()
    
    # Test 1: Download compound 2D structures
    print("\n1. Downloading compound 2D structures...")
    compound_cids = [2244, 5311, 60823]  # Aspirin, Ibuprofen, Osimertinib
    
    compound_images = manager.batch_download_compound_structures(compound_cids)
    print(f"✓ Downloaded {len(compound_images)} compound structures")
    
    for img in compound_images:
        print(f"  - {img.caption} ({img.width}x{img.height})")
        print(f"    {img.local_path}")
    
    # Test 2: Download protein 3D structures
    print("\n2. Downloading protein 3D structures...")
    pdb_ids = ["6XM0", "1M17", "7BV2"]  # KRAS, p53, EGFR
    
    protein_images = manager.batch_download_protein_structures(pdb_ids)
    print(f"✓ Downloaded {len(protein_images)} protein structures")
    
    for img in protein_images:
        print(f"  - {img.caption}")
        print(f"    https://www.rcsb.org/structure/{img.source_id}")
    
    # Test 3: Download pathway diagram
    print("\n3. Downloading pathway diagram...")
    pathway_img = manager.get_pathway_diagram("04010", organism="hsa")  # MAPK pathway
    
    if pathway_img:
        print(f"✓ Downloaded pathway: {pathway_img.caption}")
        print(f"  {pathway_img.local_path}")
    
    # Test 4: Create comparison grid
    print("\n4. Creating comparison grid...")
    all_images = compound_images + protein_images
    if pathway_img:
        all_images.append(pathway_img)
    
    grid_path = manager.create_comparison_grid(all_images, cols=3)
    print(f"✓ Created grid: {grid_path}")
    
    # Test 5: Save metadata
    print("\n5. Saving metadata...")
    manager.save_metadata(all_images)
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print(f"📁 Images saved to: {manager.cache_dir}")