#!/usr/bin/env python3
"""
Enhanced Quick Start Script
Complete setup for multi-modal biological discovery platform.

This script:
1. Ingests data from multiple sources (papers, compounds, proteins, trials)
2. Uploads to Qdrant Cloud
3. Tests the AI agent
4. Launches the Streamlit app

Usage:
    python quickstart_enhanced.py [--skip-ingest] [--skip-upload] [--skip-test]
"""

import sys
import os
from pathlib import Path
import argparse
import json

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ingestion_manager import EnhancedIngestionManager
from src.qdrant_setup import EnhancedQdrantManager
from src.agent import BiologicalResearchAgent


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def check_environment():
    """Check required environment variables."""
    print_header("🔍 CHECKING ENVIRONMENT")
    
    required = {
        "GROQ_API_KEY": "Groq LLM API",
        "QDRANT_URL": "Qdrant Cloud URL",
        "QDRANT_API_KEY": "Qdrant API Key"
    }
    
    missing = []
    for var, desc in required.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {desc}: {value[:20]}...")
        else:
            print(f"❌ {desc}: NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        print("Please set these in your .env file")
        return False
    
    print("\n✅ All environment variables set")
    return True


def ingest_data():
    """Ingest multi-modal data."""
    print_header("📥 INGESTING DATA")
    
    manager = EnhancedIngestionManager(output_dir="data/raw")
    
    # Define research topics
    topics = [
        {
            "query": "KRAS inhibitor lung cancer",
            "max_papers": 20,
            "max_trials": 10,
            "min_date": "2020/01/01"
        },
        {
            "query": "CDK4/6 inhibitor breast cancer",
            "max_papers": 15,
            "max_trials": 8,
            "min_date": "2021/01/01"
        }
    ]
    
    all_stats = []
    
    for i, topic in enumerate(topics, 1):
        print(f"\n{'─'*80}")
        print(f"📚 Topic {i}/{len(topics)}: {topic['query']}")
        print(f"{'─'*80}\n")
        
        try:
            stats = manager.ingest_comprehensive(
                query=topic["query"],
                max_papers=topic["max_papers"],
                max_trials=topic["max_trials"],
                min_date=topic.get("min_date"),
                include_proteins=True,
                include_trials=True
            )
            
            print(f"\n✅ Success!")
            print(f"   Papers: {stats['papers']}")
            print(f"   Compounds: {stats['compounds']}")
            print(f"   Proteins: {stats['proteins']}")
            print(f"   Trials: {stats['trials']}")
            
            all_stats.append({
                "query": topic["query"],
                "success": True,
                **stats
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            all_stats.append({
                "query": topic["query"],
                "success": False,
                "error": str(e)
            })
    
    # Save summary
    summary = {
        "total_papers": sum(s.get('papers', 0) for s in all_stats),
        "total_compounds": sum(s.get('compounds', 0) for s in all_stats),
        "total_proteins": sum(s.get('proteins', 0) for s in all_stats),
        "total_trials": sum(s.get('trials', 0) for s in all_stats),
        "topics": all_stats
    }
    
    summary_path = Path("data/raw/ingestion_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print_header("📊 INGESTION SUMMARY")
    print(f"Total Papers: {summary['total_papers']}")
    print(f"Total Compounds: {summary['total_compounds']}")
    print(f"Total Proteins: {summary['total_proteins']}")
    print(f"Total Trials: {summary['total_trials']}")
    print(f"\n📁 Data saved to: data/raw/")
    
    return summary


def upload_to_qdrant():
    """Upload data to Qdrant Cloud."""
    print_header("☁️  UPLOADING TO QDRANT CLOUD")
    
    manager = EnhancedQdrantManager()
    
    # Create collections
    print("Creating collections...")
    manager.create_all_collections(recreate=False)
    
    # Populate all collections
    print("\nPopulating collections...")
    data_dir = Path("data/raw")
    manager.populate_all(data_dir)
    
    # Show stats
    print_header("📊 QDRANT STATISTICS")
    manager.print_stats()


def test_agent():
    """Test the AI agent."""
    print_header("🤖 TESTING AI AGENT")
    
    try:
        agent = BiologicalResearchAgent()
        
        test_queries = [
            "What are KRAS inhibitors for lung cancer?",
            "Tell me about CDK4/6 inhibitors",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'─'*80}")
            print(f"Test {i}/{len(test_queries)}: {query}")
            print(f"{'─'*80}\n")
            
            response = agent.query(query)
            
            print(f"📊 Data Found: {response.data_found}")
            print(f"\n💡 Answer:\n{response.answer[:300]}...")
            
            if response.sources:
                print(f"\n📚 Sources: {len(response.sources)}")
                for j, source in enumerate(response.sources[:2], 1):
                    if source["type"] == "paper":
                        print(f"  {j}. {source['title'][:60]}...")
                    elif source["type"] == "compound":
                        print(f"  {j}. {source['name']} ({source['formula']})")
        
        print("\n✅ Agent tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent test failed: {e}")
        return False


def launch_app():
    """Launch Streamlit app."""
    print_header("🚀 LAUNCHING APP")
    
    print("Starting Streamlit app...")
    print("Navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop\n")
    
    import subprocess
    subprocess.run(["streamlit", "run", "enhanced_app.py"])


def main():
    """Main quickstart flow."""
    parser = argparse.ArgumentParser(description="Enhanced Platform Quickstart")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Qdrant upload")
    parser.add_argument("--skip-test", action="store_true", help="Skip agent testing")
    parser.add_argument("--no-app", action="store_true", help="Don't launch app")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  🧬 ENHANCED BIOLOGICAL DISCOVERY PLATFORM")
    print("  Multi-Modal AI Research Assistant")
    print("="*80)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix issues and try again.")
        return
    
    # Step 2: Ingest data
    if not args.skip_ingest:
        try:
            summary = ingest_data()
        except Exception as e:
            print(f"\n❌ Ingestion failed: {e}")
            print("You can skip ingestion next time with --skip-ingest")
            return
    else:
        print_header("⏭️  SKIPPING INGESTION")
    
    # Step 3: Upload to Qdrant
    if not args.skip_upload:
        try:
            upload_to_qdrant()
        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            return
    else:
        print_header("⏭️  SKIPPING UPLOAD")
    
    # Step 4: Test agent
    if not args.skip_test:
        try:
            if not test_agent():
                print("\n⚠️  Agent tests had issues, but continuing...")
        except Exception as e:
            print(f"\n❌ Testing failed: {e}")
            print("You can skip testing with --skip-test")
    else:
        print_header("⏭️  SKIPPING TESTS")
    
    # Step 5: Launch app
    if not args.no_app:
        print_header("✅ SETUP COMPLETE")
        print("\n🎉 Everything is ready!")
        print("\nYou can now:")
        print("  1. Use the Streamlit app (launching now)")
        print("  2. Use the Python API:")
        print("     from src.agent import BiologicalResearchAgent")
        print("     agent = BiologicalResearchAgent()")
        print("     response = agent.query('your question')")
        
        input("\nPress Enter to launch the app, or Ctrl+C to exit...")
        launch_app()
    else:
        print_header("✅ SETUP COMPLETE")
        print("\nTo launch the app manually:")
        print("  streamlit run enhanced_app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)