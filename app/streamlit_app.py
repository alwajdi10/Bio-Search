"""
Enhanced Streamlit App with AI Agent
Multi-modal biological research platform with conversational AI.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import BiologicalResearchAgent

st.set_page_config(
    page_title="🧬 AI Research Assistant",
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
    .source-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #400025;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #400025;
    }
    .assistant-message {
        background-color: #400025;
    }
</style>
""", unsafe_allow_html=True)

# Initialize agent
@st.cache_resource
def init_agent():
    try:
        return BiologicalResearchAgent()
    except Exception as e:
        st.error(f"❌ Error initializing agent: {e}")
        st.info("Make sure you have GROQ_API_KEY and Qdrant credentials in .env")
        return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent = init_agent()

# Header
st.markdown('<p class="main-header">🧬 AI Biological Research Assistant</p>', unsafe_allow_html=True)
st.caption("Ask me anything about biomedical research - papers, compounds, proteins, genes, clinical trials")

# Sidebar
with st.sidebar:
    st.header("💡 Example Queries")
    
    example_queries = [
        "What are KRAS inhibitors for lung cancer?",
        "Tell me about Sotorasib mechanism of action",
        "Find clinical trials for pancreatic cancer",
        "What proteins are involved in EGFR signaling?",
        "Recent PROTACs for CDK4/6 in breast cancer",
        "How do mRNA vaccines work?"
    ]
    
    for i, query in enumerate(example_queries):
        if st.button(query, key=f"example_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
    
    st.divider()
    
    st.header("ℹ️ Features")
    st.write("""
    ✓ Search papers (PubMed)  
    ✓ Find compounds (PubChem)  
    ✓ Explore proteins (UniProt)  
    ✓ Check clinical trials  
    ✓ AI-powered responses  
    ✓ Direct source links
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Status
    st.header("🔧 Status")
    if st.session_state.agent:
        st.success("✓ Agent online")
    else:
        st.error("✗ Agent offline")

# Main chat interface
st.header("💬 Chat Interface")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for source in message["sources"]:
                        if source["type"] == "paper":
                            st.markdown(f"""
                            <div class="source-card">
                                📄 <strong>Paper:</strong> {source['title'][:100]}...<br>
                                <small>PMID: {source['pmid']} | Relevance: {source.get('relevance', 0):.2f}</small><br>
                                <a href="{source['url']}" target="_blank">View on PubMed →</a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        elif source["type"] == "compound":
                            st.markdown(f"""
                            <div class="source-card">
                                💊 <strong>Compound:</strong> {source['name']}<br>
                                <small>CID: {source['cid']} | Formula: {source.get('formula', 'N/A')}</small><br>
                                <a href="{source['url']}" target="_blank">View on PubChem →</a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        elif source["type"] == "external":
                            st.markdown(f"""
                            <div class="source-card">
                                🔗 <strong>{source['name']}</strong><br>
                                <small>{source.get('description', '')}</small><br>
                                <a href="{source['url']}" target="_blank">Visit →</a>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Show related queries
            if "related_queries" in message and message["related_queries"]:
                with st.expander("🔍 Related Queries", expanded=False):
                    for rq in message["related_queries"]:
                        if st.button(rq, key=f"related_{hash(rq)}"):
                            st.session_state.messages.append({"role": "user", "content": rq})
                            st.rerun()

# Input area
st.divider()
user_input = st.chat_input("Ask me anything about biomedical research...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get agent response
    if st.session_state.agent:
        with st.spinner("🔬 Researching..."):
            response = st.session_state.agent.query(user_input)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "related_queries": response.related_queries,
                "data_found": response.data_found
            })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Sorry, the agent is not available. Please check your configuration."
        })
    
    st.rerun()

# Tabs for advanced features
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Database Stats", "🔬 Advanced Search", "⚙️ Settings"])

with tab1:
    st.subheader("Database Statistics")
    
    if st.session_state.agent and st.session_state.agent.search_engine:
        try:
            from src.qdrant_setup import QdrantCloudManager
            manager = QdrantCloudManager()
            stats = manager.get_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Papers", 
                    stats.get('papers', {}).get('count', 0),
                    help="Research papers from PubMed"
                )
            
            with col2:
                st.metric(
                    "Compounds", 
                    stats.get('compounds', {}).get('count', 0),
                    help="Chemical compounds from PubChem"
                )
            
            with col3:
                st.metric(
                    "Total Vectors", 
                    stats.get('papers', {}).get('count', 0) + 
                    stats.get('compounds', {}).get('count', 0)
                )
        except:
            st.info("Database stats unavailable")
    else:
        st.warning("Agent not connected to database")

with tab2:
    st.subheader("Advanced Search Options")
    
    search_type = st.selectbox(
        "Search Type",
        ["Semantic Search", "SMILES Structure Search", "Multi-Modal Search"]
    )
    
    if search_type == "SMILES Structure Search":
        smiles = st.text_input("Enter SMILES notation", placeholder="CC(=O)Oc1ccccc1C(=O)O")
        if st.button("Search Structure"):
            if st.session_state.agent and st.session_state.agent.search_engine:
                with st.spinner("Searching..."):
                    results = st.session_state.agent.search_engine.search_by_smiles(smiles, limit=10)
                    st.json(results)
            else:
                st.error("Search engine not available")

with tab3:
    st.subheader("Settings")
    
    st.number_input(
        "Max results per search",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of results to retrieve from database"
    )
    
    st.checkbox("Enable debug mode", value=False)
    st.checkbox("Show search intent extraction", value=False)

# Footer
st.divider()
st.caption("🧬 Powered by Qdrant, Groq AI, and open biological databases | Built with Streamlit")