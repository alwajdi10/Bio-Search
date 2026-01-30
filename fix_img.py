#!/usr/bin/env python3
"""
AUTO-FIX SCRIPT FOR IMAGE SEARCH
Automatically patches image_manager.py and image_embeddings.py

Usage:
    python auto_fix_images.py
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime


def backup_file(filepath):
    """Create backup of file."""
    backup_path = Path(str(filepath) + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(filepath, backup_path)
    print(f"✓ Backed up to: {backup_path}")
    return backup_path


def fix_image_manager():
    """Fix image_manager.py glob concatenation issue."""
    filepath = Path("src/image_manager.py")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"\n{'='*80}")
    print("FIXING: src/image_manager.py")
    print(f"{'='*80}")
    
    # Backup
    backup_file(filepath)
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: glob concatenation
    old_pattern = 'for img_file in path.glob("*.png") + path.glob("*.jpeg"):'
    new_pattern = '''# Combine glob results properly
                img_files = list(path.glob("*.png")) + list(path.glob("*.jpeg")) + list(path.glob("*.jpg"))
                for img_file in img_files:'''
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✓ Fixed glob concatenation issue")
    else:
        print("⚠ Glob issue not found (may already be fixed)")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed: {filepath}")
    return True


def fix_image_embeddings():
    """Fix image_embeddings.py import issue."""
    filepath = Path("src/image_embeddings.py")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"\n{'='*80}")
    print("FIXING: src/image_embeddings.py")
    print(f"{'='*80}")
    
    # Backup
    backup_file(filepath)
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix: Import path
    old_import = 'from image_manager import ImageManager'
    new_import = 'from src.image_manager import ImageManager'
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✓ Fixed import path")
    else:
        print("⚠ Import issue not found (may already be fixed)")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed: {filepath}")
    return True


def check_transformers():
    """Check if transformers is installed."""
    try:
        import transformers
        print(f"✅ transformers {transformers.__version__} is installed")
        return True
    except ImportError:
        print("❌ transformers is NOT installed")
        print("\n   Install with:")
        print("   pip install transformers")
        return False


def verify_fixes():
    """Verify that fixes work."""
    print(f"\n{'='*80}")
    print("VERIFYING FIXES")
    print(f"{'='*80}")
    
    # Test 1: Import image_manager
    try:
        from src.image_manager import ImageManager
        print("✅ ImageManager imports successfully")
    except Exception as e:
        print(f"❌ ImageManager import failed: {e}")
        return False
    
    # Test 2: Import image_embeddings
    try:
        from src.image_embeddings import ImageSearchEngine
        print("✅ ImageSearchEngine imports successfully")
    except Exception as e:
        print(f"❌ ImageSearchEngine import failed: {e}")
        return False
    
    # Test 3: Check for images
    image_dir = Path("data/images")
    if not image_dir.exists():
        print("⚠ No images cached yet - run: python quick_img.py")
    else:
        manager = ImageManager(cache_dir=str(image_dir))
        try:
            images = manager.get_all_cached_images()
            print(f"✅ get_all_cached_images() works - found {len(images)} images")
        except Exception as e:
            print(f"❌ get_all_cached_images() failed: {e}")
            return False
    
    # Test 4: Try to initialize search (if transformers available)
    try:
        import transformers
        try:
            search = ImageSearchEngine(image_dir="data/images", use_clip=True)
            print(f"✅ ImageSearchEngine initializes successfully")
            print(f"   Indexed {len(search.images)} images")
        except Exception as e:
            print(f"⚠ ImageSearchEngine init failed: {e}")
    except ImportError:
        print("⚠ Skipping ImageSearchEngine test (transformers not installed)")
    
    return True


def main():
    """Main fix script."""
    print("="*80)
    print("IMAGE SEARCH AUTO-FIX SCRIPT")
    print("="*80)
    
    print("\nThis script will:")
    print("1. Backup your files")
    print("2. Fix glob concatenation in image_manager.py")
    print("3. Fix import path in image_embeddings.py")
    print("4. Verify the fixes work")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Check transformers first
    print(f"\n{'='*80}")
    print("CHECKING DEPENDENCIES")
    print(f"{'='*80}")
    has_transformers = check_transformers()
    
    # Apply fixes
    success = True
    success &= fix_image_manager()
    success &= fix_image_embeddings()
    
    if not success:
        print("\n❌ Some fixes failed. Check error messages above.")
        return
    
    # Verify
    if verify_fixes():
        print(f"\n{'='*80}")
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        if has_transformers:
            print("\n🎉 Image search should now work!")
            print("\nNext steps:")
            print("1. Test with: python COMPLETE_FIX.py")
            print("2. Or run: streamlit run app/app_img.py")
        else:
            print("\n⚠️ Image search will work after installing transformers:")
            print("   pip install transformers")
    else:
        print("\n❌ Verification failed. Please check error messages.")
    
    print("\n📝 Note: Backup files were created (*.backup_*)")
    print("   You can restore from backup if needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)