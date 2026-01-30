"""
Qdrant Cloud Verification Script
Tests connection, collections, and data integrity.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


def test_qdrant():
    """Complete verification of Qdrant Cloud setup."""
    
    print("\n" + "="*70)
    print("  🔍 QDRANT CLOUD VERIFICATION")
    print("="*70 + "\n")
    
    # Step 1: Connection
    print("1️⃣  Testing connection...")
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        print("   ❌ Missing QDRANT_URL or QDRANT_API_KEY in .env")
        return False
    
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        print(f"   ✅ Connected to: {url[:50]}...")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Step 2: List Collections
    print("\n2️⃣  Checking collections...")
    
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if not collection_names:
            print("   ⚠️  No collections found")
            print("   Run: python -m src.qdrant_setup --create")
            return False
        
        print(f"   ✅ Found {len(collection_names)} collections:")
        for name in collection_names:
            print(f"      • {name}")
            
    except Exception as e:
        print(f"   ❌ Failed to list collections: {e}")
        return False
    
    # Step 3: Check Each Collection
    print("\n3️⃣  Verifying collection data...")
    
    expected_collections = {
        "research_papers": 768,
        "chemical_compounds": 4096,
        "proteins": 768,
        "clinical_trials": 768
    }
    
    total_points = 0
    
    for name, expected_dim in expected_collections.items():
        try:
            info = client.get_collection(name)
            count = info.points_count
            dim = info.config.params.vectors.size
            distance = info.config.params.vectors.distance.value
            
            print(f"\n   📊 {name}:")
            print(f"      Points: {count}")
            print(f"      Dimensions: {dim} (expected: {expected_dim})")
            print(f"      Distance: {distance}")
            
            if count == 0:
                print(f"      ⚠️  EMPTY - no data uploaded")
            else:
                print(f"      ✅ Has data")
                total_points += count
            
            if dim != expected_dim:
                print(f"      ⚠️  Dimension mismatch!")
                
        except Exception as e:
            print(f"\n   ❌ {name}: {e}")
    
    # Step 4: Sample Search
    print("\n4️⃣  Testing search functionality...")
    
    try:
        # Check if papers collection has data
        papers_info = client.get_collection("research_papers")
        
        if papers_info.points_count > 0:
            # Get a random paper
            results = client.scroll(
                collection_name="research_papers",
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if results[0]:
                sample = results[0][0]
                print(f"   ✅ Sample paper retrieved:")
                print(f"      PMID: {sample.payload.get('pmid')}")
                print(f"      Title: {sample.payload.get('title', '')[:60]}...")
                
                # Try a dummy search
                dummy_vector = [0.1] * 768
                search_results = client.search(
                    collection_name="research_papers",
                    query_vector=dummy_vector,
                    limit=1
                )
                
                if search_results:
                    print(f"   ✅ Search works! Got {len(search_results)} results")
                else:
                    print(f"   ⚠️  Search returned no results")
        else:
            print("   ⚠️  No data to search - upload data first")
            
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  📊 SUMMARY")
    print("="*70)
    
    if total_points > 0:
        print(f"\n  ✅ Qdrant Cloud is working!")
        print(f"  ✅ Total data points: {total_points}")
        print(f"  ✅ Collections active: {len(collection_names)}")
        print("\n  Your database is ready to use! 🎉")
        return True
    else:
        print(f"\n  ⚠️  Qdrant is connected but empty")
        print(f"  📥 Upload data with:")
        print(f"     python -m src.qdrant_setup --create --populate data/raw")
        return False


def quick_check():
    """Quick status check."""
    print("\n🔍 Quick Status Check\n")
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        
        for name in ["research_papers", "chemical_compounds"]:
            try:
                info = client.get_collection(name)
                print(f"✅ {name}: {info.points_count} points")
            except:
                print(f"❌ {name}: Not found")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_check()
    else:
        success = test_qdrant()
        sys.exit(0 if success else 1)