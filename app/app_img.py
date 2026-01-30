"""
Enhanced App with Image Support
Multi-modal biological research platform with visual search.
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import BiologicalResearchAgent
from src.image_manager import ImageManager
from src.image_embeddings import ImageSearchEngine

st.set_page_config(
    page_title="🧬BIO-SEARCH",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .image-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background: #f9f9f9;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def init_agent():
    try:
        return BiologicalResearchAgent()
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

@st.cache_resource
def init_image_manager():
    try:
        return ImageManager(cache_dir="data/images")
    except Exception as e:
        st.warning(f"Image manager unavailable: {e}")
        return None

@st.cache_resource
def init_image_search():
    try:
        return ImageSearchEngine(image_dir="data/images", use_clip=True)
    except Exception as e:
        st.warning(f"Image search unavailable: {e}")
        return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent = init_agent()

if 'image_manager' not in st.session_state:
    st.session_state.image_manager = init_image_manager()

if 'image_search' not in st.session_state:
    st.session_state.image_search = init_image_search()

# Header
st.markdown('<p class="main-header">🧬 BIO-SEARCH</p>', unsafe_allow_html=True)
st.caption("Multi-modal search: text, compounds, proteins, trials, and images")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat", 
    "🖼️ Visual Search", 
    "🧪 Compound Viewer",
    "📊 Statistics"
])

# ==========================================
# TAB 1: CHAT WITH AGENT
# ==========================================
with tab1:
    st.header("Chat with AI Agent")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show images if available
                if "images" in message and message["images"]:
                    st.write("**Related Images:**")
                    
                    cols = st.columns(min(len(message["images"]), 3))
                    for idx, img_data in enumerate(message["images"][:3]):
                        with cols[idx]:
                            try:
                                img = Image.open(img_data["path"])
                                st.image(img, caption=img_data["caption"], use_container_width=True)
                                st.caption(f"Source: {img_data['source']}")
                            except:
                                st.info(img_data["url"])
                
                # Show sources
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for source in message["sources"]:
                            if source["type"] == "paper":
                                st.markdown(f"""
                                📄 **{source['title'][:100]}...**  
                                PMID: {source['pmid']} | [View on PubMed]({source['url']})
                                """)
                            elif source["type"] == "compound":
                                st.markdown(f"""
                                💊 **{source['name']}**  
                                CID: {source['cid']} | Formula: {source.get('formula', 'N/A')}  
                                [View on PubChem]({source['url']})
                                """)
    
    # Chat input
    user_input = st.chat_input("Ask about papers, compounds, proteins, trials, or request images...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get agent response
        if st.session_state.agent:
            with st.spinner("🔬 Researching..."):
                response = st.session_state.agent.query(user_input)
                
                # Check if query mentions images/structures
                images_data = []
                if any(word in user_input.lower() for word in ["structure", "image", "diagram", "show", "visualize"]):
                    # Fetch relevant images
                    img_manager = st.session_state.image_manager
                    if img_manager and response.sources:
                        for source in response.sources[:3]:
                            if source["type"] == "compound":
                                img = img_manager.get_compound_2d_structure(source["cid"])
                                if img:
                                    images_data.append({
                                        "path": img.local_path,
                                        "caption": f"2D structure of {source['name']}",
                                        "source": "PubChem",
                                        "url": source["url"]
                                    })
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "images": images_data,
                    "related_queries": response.related_queries
                })
        
        st.rerun()

# ==========================================
# TAB 2: VISUAL SEARCH
# ==========================================
with tab2:
    st.header("🖼️ Visual Search")
    st.write("Search images using text descriptions or find similar images")
    
    search_col1, search_col2 = st.columns([2, 1])
    
    with search_col1:
        search_type = st.radio(
            "Search Type",
            ["Text-to-Image", "Image-to-Image"],
            horizontal=True
        )
        
        if search_type == "Text-to-Image":
            text_query = st.text_input(
                "Describe what you're looking for",
                placeholder="e.g., '2D chemical structure', 'protein 3D model', 'pathway diagram'"
            )
            
            if st.button("🔍 Search Images") and text_query:
                img_search = st.session_state.image_search
                
                if img_search:
                    with st.spinner("Searching..."):
                        results = img_search.search_by_text(text_query, top_k=9)
                        
                        if results:
                            st.success(f"Found {len(results)} matching images")
                            
                            # Display in grid
                            cols = st.columns(3)
                            for idx, result in enumerate(results):
                                with cols[idx % 3]:
                                    try:
                                        img = Image.open(result['image'].local_path)
                                        st.image(img, use_container_width=True)
                                        st.caption(f"{result['image'].caption}")
                                        st.caption(f"Score: {result['score']:.3f}")
                                        
                                        if result['image'].url:
                                            st.markdown(f"[View Source]({result['image'].url})")
                                    except Exception as e:
                                        st.error(f"Error loading image: {e}")
                        else:
                            st.info("No images found. Try a different query.")
                else:
                    st.warning("Image search not available. Install transformers: `pip install transformers`")
        
        else:  # Image-to-Image
            uploaded_file = st.file_uploader(
                "Upload an image to find similar ones",
                type=["png", "jpg", "jpeg"]
            )
            
            if uploaded_file and st.button("Find Similar"):
                img_search = st.session_state.image_search
                
                if img_search:
                    # Save uploaded file
                    temp_path = Path("temp_query_image.png")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    with st.spinner("Finding similar images..."):
                        results = img_search.search_similar(temp_path, top_k=6)
                        
                        st.success(f"Found {len(results)} similar images")
                        
                        cols = st.columns(3)
                        for idx, result in enumerate(results):
                            with cols[idx % 3]:
                                try:
                                    img = Image.open(result['image'].local_path)
                                    st.image(img, use_container_width=True)
                                    st.caption(f"Similarity: {result['score']:.3f}")
                                except:
                                    pass
                    
                    temp_path.unlink()

# ==========================================
# TAB 3: COMPOUND STRUCTURE VIEWER
# ==========================================
with tab3:
    st.header("🧪 Compound Structure Viewer")
    
    viewer_col1, viewer_col2 = st.columns([1, 2])
    
    with viewer_col1:
        cid_input = st.text_input(
            "Enter PubChem CID",
            placeholder="e.g., 2244 (Aspirin)"
        )
        
        if st.button("Load Structure") and cid_input:
            try:
                cid = int(cid_input)
                img_manager = st.session_state.image_manager
                
                if img_manager:
                    with st.spinner("Downloading structure..."):
                        img_data = img_manager.get_compound_2d_structure(cid, size="large")
                        
                        if img_data:
                            st.session_state['current_structure'] = img_data
                            st.success(f"Loaded CID {cid}")
                        else:
                            st.error(f"Could not find structure for CID {cid}")
            except ValueError:
                st.error("Please enter a valid CID number")
    
    with viewer_col2:
        if 'current_structure' in st.session_state:
            img_data = st.session_state['current_structure']
            
            try:
                img = Image.open(img_data.local_path)
                st.image(img, caption=img_data.caption, use_container_width=True)
                
                # Show metadata
                st.write("**Image Details:**")
                st.write(f"- Size: {img_data.width}x{img_data.height}")
                st.write(f"- Format: {img_data.format}")
                st.write(f"- Source: PubChem")
                st.markdown(f"[View on PubChem]({img_data.url})")
                
                # Download button
                with open(img_data.local_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Image",
                        f,
                        file_name=f"compound_{img_data.source_id}.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Error displaying structure: {e}")

# ==========================================
# TAB 4: STATISTICS
# ==========================================
with tab4:
    st.header("📊 Platform Statistics")
    
    # Database stats
    if st.session_state.agent and st.session_state.agent.search_engine:
        try:
            from src.qdrant_setup import EnhancedQdrantManager
            manager = EnhancedQdrantManager()
            stats = manager.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Papers", stats.get('research_papers', {}).get('count', 0))
            with col2:
                st.metric("Compounds", stats.get('chemical_compounds', {}).get('count', 0))
            with col3:
                st.metric("Proteins", stats.get('proteins', {}).get('count', 0))
            with col4:
                st.metric("Trials", stats.get('clinical_trials', {}).get('count', 0))
        except:
            st.info("Database statistics unavailable")
    
    # Image stats
    st.divider()
    st.subheader("Image Library")
    
    img_manager = st.session_state.image_manager
    if img_manager:
        cached_images = img_manager.get_all_cached_images()
        
        if cached_images:
            st.metric("Total Images", len(cached_images))
            
            # Breakdown by type
            from collections import Counter
            type_counts = Counter(img.image_type for img in cached_images)
            
            st.write("**By Type:**")
            for img_type, count in type_counts.items():
                st.write(f"- {img_type}: {count}")
            
            # Sample gallery
            st.write("**Sample Gallery:**")
            sample_images = cached_images[:6]
            
            cols = st.columns(3)
            for idx, img in enumerate(sample_images):
                with cols[idx % 3]:
                    try:
                        pil_img = Image.open(img.local_path)
                        st.image(pil_img, caption=img.caption, use_container_width=True)
                    except:
                        pass
        else:
            st.info("No cached images yet. Use the compound viewer or run image ingestion.")

# Sidebar
with st.sidebar:
    st.header("🎨 Image Tools")
    
    if st.button("📥 Download Sample Structures"):
        img_manager = st.session_state.image_manager
        if img_manager:
            with st.spinner("Downloading..."):
                # Download popular drug structures
                sample_cids = [2244, 5311, 60823, 11526198]  # Aspirin, Ibuprofen, etc.
                images = img_manager.batch_download_compound_structures(sample_cids)
                st.success(f"Downloaded {len(images)} structures")
                st.rerun()
    
    st.divider()
    
    st.header("ℹ️ Features")
    st.write("""
    ✓ Text & visual search  
    ✓ 2D compound structures  
    ✓ 3D protein models  
    ✓ Pathway diagrams  
    ✓ Image similarity search  
    ✓ CLIP-powered search
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")

st.divider()
st.caption("🧬 Powered by Qdrant, Groq AI, CLIP, and open biological databases")