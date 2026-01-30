"""
Fixed 3D Model Ingestor with PyMOL Visualization
Downloads PDB structures and generates high-quality images using PyMOL.
"""

import os
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiologicalCategory:
    """Predefined biological categories with known structures."""
    
    # Proteins by function/family
    PROTEINS = {
        "kinases": ["6XM0", "4MXO", "5EZ3", "6Q0J", "7JHP"],
        "receptors": ["4ZUD", "5IU4", "6DDE", "4EIY", "7TN4"],
        "enzymes": ["1M17", "3KMD", "4HJO", "5D3Q", "6LU7"],
        "antibodies": ["1HZH", "5VKK", "6XC4", "7BWJ", "5IES"],
        "transporters": ["3J7Y", "5I20", "6QNO", "5WLC", "6W6V"],
    }
    
    VIRUSES = {
        "covid19": ["6VXX", "6W9C", "6LZG", "7BV2", "7JMP"],
        "influenza": ["4WE4", "4WE8", "3ZTJ", "1RUZ"],
        "hiv": ["1AIK", "1DLO", "1QBR", "2B4C"],
    }


class Fixed3DIngestor:
    """
    Downloads 3D biological structure files from RCSB PDB.
    Optionally generates images using PyMOL.
    """
    
    def __init__(self, cache_dir: str = "data/3d_structures"):
        """Initialize 3D structure ingestor."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories
        categories = [
            "proteins", "viruses"
        ]
        
        for cat in categories:
            (self.cache_dir / cat).mkdir(exist_ok=True)
            (self.cache_dir / cat / "images").mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BiologicalResearch/1.0 (Educational)'
        })
        
        # Check if PyMOL is available
        self.pymol_available = self._check_pymol()
        
        logger.info(f"✓ Fixed 3D ingestor initialized")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  PyMOL available: {self.pymol_available}")
    
    def _check_pymol(self) -> bool:
        """Check if PyMOL is installed and accessible."""
        try:
            result = subprocess.run(
                ['pymol', '-c', '-Q'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def download_pdb_structure(
        self,
        pdb_id: str,
        category: str = "general"
    ) -> Optional[Dict[str, Any]]:
        """
        Download PDB structure file.
        
        Args:
            pdb_id: PDB identifier
            category: Category for organization
            
        Returns:
            Dictionary with structure info or None
        """
        try:
            pdb_upper = pdb_id.upper()
            
            # Download CIF format (modern standard)
            url = f"https://files.rcsb.org/download/{pdb_upper}.cif"
            
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                logger.debug(f"Failed to download {pdb_id}: HTTP {response.status_code}")
                return None
            
            # Save structure file
            structure_path = self.cache_dir / category / f"{pdb_id.lower()}.cif"
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(structure_path, 'wb') as f:
                f.write(response.content)
            
            # Get metadata
            metadata = self._get_pdb_metadata(pdb_id)
            
            result = {
                "pdb_id": pdb_upper,
                "category": category,
                "structure_file": str(structure_path),
                "url": f"https://www.rcsb.org/structure/{pdb_upper}",
                **metadata
            }
            
            logger.info(f"✓ Downloaded {pdb_upper}: {metadata['title'][:60]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to download {pdb_id}: {e}")
            return None
    
    def _get_pdb_metadata(self, pdb_id: str) -> Dict[str, Any]:
        """Get metadata from PDB REST API."""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "title": data.get("struct", {}).get("title", f"Protein {pdb_id}"),
                    "keywords": data.get("struct_keywords", {}).get("pdbx_keywords", ""),
                    "method": data.get("exptl", [{}])[0].get("method", ""),
                    "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", []),
                }
        except:
            pass
        
        return {
            "title": f"Protein {pdb_id}",
            "keywords": "",
            "method": "",
            "resolution": [],
        }
    
    def generate_image_pymol(
        self,
        structure_file: str,
        output_image: str,
        width: int = 1200,
        height: int = 1200
    ) -> bool:
        """
        Generate image from structure using PyMOL.
        
        Args:
            structure_file: Path to structure file
            output_image: Path for output image
            width: Image width
            height: Image height
            
        Returns:
            True if successful
        """
        if not self.pymol_available:
            logger.warning("PyMOL not available, skipping image generation")
            return False
        
        try:
            # Create PyMOL script
            pymol_script = f"""
load {structure_file}
bg_color white
set ray_trace_mode, 1
set ray_shadows, 1
set antialias, 2
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
show cartoon
spectrum count, rainbow
zoom
ray {width}, {height}
png {output_image}, dpi=300
quit
"""
            
            script_path = Path(structure_file).parent / "temp_pymol.pml"
            with open(script_path, 'w') as f:
                f.write(pymol_script)
            
            # Run PyMOL
            result = subprocess.run(
                ['pymol', '-c', '-Q', str(script_path)],
                capture_output=True,
                timeout=30
            )
            
            # Clean up script
            script_path.unlink()
            
            if result.returncode == 0 and Path(output_image).exists():
                logger.info(f"✓ Generated image: {output_image}")
                return True
            else:
                logger.warning(f"Failed to generate image for {structure_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return False
    
    def download_category(
        self,
        category_name: str,
        pdb_ids: List[str],
        generate_images: bool = False,
        max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Download structures for a category.
        
        Args:
            category_name: Category name
            pdb_ids: List of PDB IDs
            generate_images: Whether to generate images using PyMOL
            max_workers: Parallel workers
            
        Returns:
            List of structure info dictionaries
        """
        logger.info(f"Downloading {len(pdb_ids)} structures for: {category_name}")
        
        structures = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_pdb_structure, pdb_id, category_name): pdb_id
                for pdb_id in pdb_ids
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    structures.append(result)
                    
                    # Generate image if requested
                    if generate_images and self.pymol_available:
                        image_path = Path(result['structure_file']).parent / "images" / f"{result['pdb_id'].lower()}.png"
                        self.generate_image_pymol(
                            result['structure_file'],
                            str(image_path)
                        )
                        result['image_file'] = str(image_path)
                
                time.sleep(0.5)  # Rate limiting
        
        logger.info(f"✓ Downloaded {len(structures)}/{len(pdb_ids)} for {category_name}")
        return structures
    
    def ingest_all_categories(
        self,
        generate_images: bool = False,
        include_proteins: bool = True,
        include_viruses: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest structures from categories.
        
        Args:
            generate_images: Use PyMOL to generate images
            include_*: Which categories to include
            
        Returns:
            Statistics dictionary
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE 3D STRUCTURE INGESTION")
        logger.info("="*80)
        
        if generate_images and not self.pymol_available:
            logger.warning("⚠️  PyMOL not found. Images will NOT be generated.")
            logger.warning("    Install PyMOL: pip install pymol-open-source")
            generate_images = False
        
        stats = {
            "by_category": {},
            "total": 0,
            "all_structures": []
        }
        
        # Proteins
        if include_proteins:
            logger.info("\n📊 1. Downloading PROTEIN structures...")
            for subcategory, pdb_ids in BiologicalCategory.PROTEINS.items():
                structures = self.download_category(
                    f"proteins/{subcategory}",
                    pdb_ids,
                    generate_images
                )
                stats["by_category"][f"proteins_{subcategory}"] = len(structures)
                stats["all_structures"].extend(structures)
        
        # Viruses
        if include_viruses:
            logger.info("\n🦠 2. Downloading VIRUS structures...")
            for virus, pdb_ids in BiologicalCategory.VIRUSES.items():
                structures = self.download_category(
                    f"viruses/{virus}",
                    pdb_ids,
                    generate_images
                )
                stats["by_category"][f"virus_{virus}"] = len(structures)
                stats["all_structures"].extend(structures)
        
        stats["total"] = len(stats["all_structures"])
        
        # Save metadata
        self._save_metadata(stats["all_structures"])
        
        logger.info("\n" + "="*80)
        logger.info("✅ STRUCTURE INGESTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total structures downloaded: {stats['total']}")
        logger.info("\nBreakdown by category:")
        for cat, count in stats["by_category"].items():
            if count > 0:
                logger.info(f"  {cat}: {count}")
        
        return stats
    
    def _save_metadata(self, structures: List[Dict[str, Any]]):
        """Save metadata to JSON."""
        output_path = self.cache_dir / "structures_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(structures, f, indent=2)
        
        logger.info(f"\n📄 Metadata saved: {output_path}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed 3D Structure Ingestion")
    parser.add_argument("--all", action="store_true", help="Download all categories")
    parser.add_argument("--proteins", action="store_true", help="Download proteins")
    parser.add_argument("--viruses", action="store_true", help="Download viruses")
    parser.add_argument("--images", action="store_true", help="Generate images with PyMOL")
    parser.add_argument("--test", action="store_true", help="Test with 1 structure")
    
    args = parser.parse_args()
    
    ingestor = Fixed3DIngestor()
    
    if args.test:
        # Test with a single structure
        result = ingestor.download_pdb_structure("4HHB", "test")
        if result and args.images:
            image_path = ingestor.cache_dir / "test" / "images" / "4hhb.png"
            ingestor.generate_image_pymol(result['structure_file'], str(image_path))
        print(f"\n✅ Test complete. Check: {ingestor.cache_dir}")
    else:
        stats = ingestor.ingest_all_categories(
            generate_images=args.images,
            include_proteins=args.all or args.proteins,
            include_viruses=args.all or args.viruses
        )
        
        print(f"\n✅ Total structures downloaded: {stats['total']}")
        print(f"📁 Saved to: {ingestor.cache_dir}")