# 🧬 Enhanced Biological Discovery Platform

**AI-Powered Multi-Modal Research Assistant**

A comprehensive platform that combines vector search (Qdrant) with LLM reasoning (Groq) to help researchers explore biological data across multiple modalities: papers, compounds, proteins, genes, and clinical trials.

---

## 🚀 What's New in This Version

### ✨ Major Enhancements

1. **AI Research Agent** 🤖
   - Responds intelligently even when data isn't in the database
   - Uses Groq LLMs for natural language understanding
   - Provides source links and citations

2. **Multi-Modal Data Sources** 📊
   - ✅ Papers (PubMed)
   - ✅ Compounds (PubChem)
   - ✅ Proteins (UniProt)
   - ✅ Clinical Trials (ClinicalTrials.gov)
   - 🔜 Genes (NCBI Gene) - coming soon
   - 🔜 Patents - coming soon

3. **Conversational Interface** 💬
   - Chat-based UI in Streamlit
   - Maintains conversation context
   - Suggests related queries

4. **Comprehensive Source Linking** 🔗
   - Direct URLs to PubMed, PubChem, UniProt, ClinicalTrials.gov
   - Citation tracking across modalities
   - Cross-modal entity linking

---
## ✨ Pipeline 
<img width="3677" height="4630" alt="Pipeline" src="https://github.com/user-attachments/assets/6a85b2d9-9611-4e69-a183-5c6ca81661c7" />

## 📋 Prerequisites

### Required

- Python 3.11+
- GROQ_API_KEY (get from [Groq Console](https://console.groq.com))
- QDRANT_URL and QDRANT_API_KEY (from [Qdrant Cloud](https://cloud.qdrant.io))

### Optional

- NCBI_EMAIL and NCBI_API_KEY for higher PubMed rate limits

---

## 🛠️ Installation

### 1. Clone and Setup

```bash
# Clone your project
cd your-project-directory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in your project root:

```env
# LLM API (REQUIRED)
GROQ_API_KEY=your_groq_api_key_here

# Qdrant Cloud (REQUIRED)
QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here

# NCBI (Optional - improves rate limits)
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=your_ncbi_api_key
```

### 3. Install New Dependencies

Add these to your `requirements.txt`:

```
instructor==1.0.0
```

Then run:
```bash
pip install instructor
```

---

## 📁 New File Structure

```
project/
├── src/
│   ├── agent.py                          # NEW: AI agent orchestrator
│   ├── uniprot_ingestor.py              # NEW: Protein data
│   ├── clinical_trials_ingestor.py      # NEW: Clinical trials
│   ├── enhanced_ingestion_manager.py    # NEW: Multi-modal ingestion
│   ├── pubmed_ingestor.py               # Existing
│   ├── pubchem_ingestor.py              # Existing
│   ├── llm_query.py                     # Existing - enhanced
│   ├── embeddings.py                    # Existing
│   ├── search.py                        # Existing
│   └── qdrant_setup.py                  # Existing
├── enhanced_app.py                       # NEW: AI chat interface
├── quickstart_ingestion.py              # Existing
├── .env                                  # Your credentials
└── requirements.txt
```

---

## 🎯 Quick Start

### Step 1: Ingest Data (Enhanced)

```bash
python -c "
from src.enhanced_ingestion_manager import EnhancedIngestionManager

manager = EnhancedIngestionManager()
manager.ingest_comprehensive(
    query='KRAS inhibitor lung cancer',
    max_papers=20,
    max_trials=10,
    min_date='2020/01/01',
    include_proteins=True,
    include_trials=True
)
"
```

This will fetch:
- 20 papers from PubMed
- Related compounds from PubChem
- Related proteins from UniProt
- 10 clinical trials from ClinicalTrials.gov

### Step 2: Upload to Qdrant

```bash
python -m src.qdrant_setup --create --populate data/raw
```

### Step 3: Launch AI Chat Interface

```bash
streamlit run enhanced_app.py
```

Open browser to `http://localhost:8501`

---

## 💡 Usage Examples

### 1. AI Agent (Python)

```python
from src.agent import BiologicalResearchAgent

agent = BiologicalResearchAgent()

# Ask a question
response = agent.query("What are KRAS inhibitors for lung cancer?")

print(response.answer)
print(f"Found data: {response.data_found}")
print(f"Sources: {len(response.sources)}")

for source in response.sources:
    if source['type'] == 'paper':
        print(f"📄 {source['title']}")
        print(f"   {source['url']}")
```

### 2. Multi-Modal Ingestion

```python
from src.enhanced_ingestion_manager import EnhancedIngestionManager

manager = EnhancedIngestionManager()

# Ingest all modalities for a topic
stats = manager.ingest_comprehensive(
    query="CDK4/6 inhibitor breast cancer",
    max_papers=30,
    max_trials=15,
    include_proteins=True,
    include_trials=True
)

print(f"Papers: {stats['papers']}")
print(f"Compounds: {stats['compounds']}")
print(f"Proteins: {stats['proteins']}")
print(f"Trials: {stats['trials']}")
```

### 3. Fetch Proteins

```python
from src.uniprot_ingestor import UniProtIngestor

ingestor = UniProtIngestor()

proteins = ingestor.search_and_fetch(["KRAS", "EGFR"], max_per_name=2)

for protein in proteins:
    print(f"{protein.protein_name}")
    print(f"  Gene: {', '.join(protein.gene_names)}")
    print(f"  Function: {protein.function[:100]}...")
    print(f"  Link: https://www.uniprot.org/uniprotkb/{protein.uniprot_id}")
```

### 4. Fetch Clinical Trials

```python
from src.clinical_trials_ingestor import ClinicalTrialsIngestor

ingestor = ClinicalTrialsIngestor()

trials = ingestor.search_and_fetch(
    "KRAS inhibitor", 
    max_results=5,
    status="RECRUITING"
)

for trial in trials:
    print(f"{trial.title}")
    print(f"  Status: {trial.status}")
    print(f"  Phase: {trial.phase}")
    print(f"  Link: https://clinicaltrials.gov/study/{trial.nct_id}")
```

---

## 🎨 Features

### AI Agent Capabilities

✅ **Hybrid Search**
- Searches Qdrant vector database
- Falls back to LLM general knowledge
- Combines both when appropriate

✅ **Source Attribution**
- Direct links to PubMed, PubChem, UniProt, ClinicalTrials.gov
- Relevance scoring
- Citation tracking

✅ **Natural Language**
- Understands research questions
- Extracts search intent
- Suggests related queries

✅ **Multi-Modal**
- Links papers ↔ compounds ↔ proteins ↔ trials
- Cross-modal search
- Entity relationship mapping

### Data Coverage

| Modality | Source | Count (Demo) | Features |
|----------|--------|--------------|----------|
| Papers | PubMed | 20-50 | Title, abstract, authors, journal, MeSH terms |
| Compounds | PubChem | 10-30 | SMILES, formula, properties, structure |
| Proteins | UniProt | 5-20 | Sequence, function, GO terms, interactions |
| Trials | ClinicalTrials.gov | 5-15 | Status, phase, outcomes, locations |

---

## 🔧 Advanced Configuration

### Scaling Up

To scale beyond demo data:

```python
# Ingest large dataset
manager = EnhancedIngestionManager()

topics = [
    "KRAS inhibitor",
    "EGFR tyrosine kinase inhibitor",
    "CDK4/6 inhibitor",
    "PARP inhibitor",
    "PD-1 checkpoint inhibitor"
]

for topic in topics:
    manager.ingest_comprehensive(
        query=topic,
        max_papers=100,  # Scale up
        max_trials=30,
        include_proteins=True,
        include_trials=True
    )
```

### Custom Qdrant Collections

Modify `src/qdrant_setup.py` to add more collections:

```python
# Add gene collection
client.create_collection(
    collection_name="genes",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
```

### Custom Embedding Models

Change the embedding model in `src/embeddings.py`:

```python
EmbeddingGenerator(
    text_model="sentence-transformers/all-mpnet-base-v2",  # Better quality
    device="cuda"  # Use GPU
)
```

---

## 🐛 Troubleshooting

### "Agent not available"

**Issue**: GROQ_API_KEY not set or invalid

**Fix**:
```bash
# Check .env file
cat .env | grep GROQ_API_KEY

# Test API key
python -c "import os; from groq import Groq; print(Groq(api_key=os.getenv('GROQ_API_KEY')).models.list())"
```

### "Qdrant connection failed"

**Issue**: QDRANT_URL or QDRANT_API_KEY incorrect

**Fix**:
```bash
# Test connection
python -c "
import os
from qdrant_client import QdrantClient
client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)
print(client.get_collections())
"
```

### Rate Limiting

**Issue**: Too many API requests

**Fix**: Adjust rate limits in ingestors:
```python
# In each ingestor
def __init__(self, rate_limit: float = 1.0):  # Increase from 0.5 to 1.0
```

---

## 📊 Performance

### Current Metrics

- **Query Response**: < 2 seconds
- **Vector Search**: < 100ms
- **LLM Generation**: ~1-2 seconds
- **Ingestion Speed**: ~10 papers/minute

### Scaling Estimates

| Dataset Size | Search Time | Storage |
|--------------|-------------|---------|
| 100 papers | 50ms | 5MB |
| 1,000 papers | 80ms | 50MB |
| 10,000 papers | 120ms | 500MB |
| 100,000 papers | 200ms | 5GB |

---

## 🗺️ Roadmap

### Phase 1 (Current) ✅
- [x] Multi-modal ingestion
- [x] AI agent with LLM
- [x] Chat interface
- [x] Source linking

### Phase 2 (Next) 🚧
- [ ] Gene data from NCBI
- [ ] Patent search
- [ ] Pathway visualization
- [ ] Export to PDF/CSV

### Phase 3 (Future) 🔮
- [ ] Real-time paper alerts
- [ ] Collaborative annotations
- [ ] API endpoint
- [ ] Docker deployment

---
# 🏗️ System Architecture Overview

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Streamlit App (enhanced_app.py)                  │  │
│  │  - Chat interface                                        │  │
│  │  - Multi-tab search                                      │  │
│  │  - Source visualization                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI AGENT LAYER                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   BiologicalResearchAgent (agent.py)                     │  │
│  │                                                           │  │
│  │   • Query understanding (LLM)                            │  │
│  │   • Vector search (Qdrant)                               │  │
│  │   • Response generation (Groq)                           │  │
│  │   • Source linking                                       │  │
│  └──────────────┬───────────────────────┬────────────────────┘  │
└─────────────────┼───────────────────────┼─────────────────────────┘
                  │                       │
        ┌─────────▼─────────┐   ┌────────▼──────────┐
        │  LLM Processing   │   │  Vector Search    │
        │  (llm_query.py)   │   │  (search.py)      │
        │                   │   │                   │
        │  • Intent extract │   │  • Semantic search│
        │  • Summarization  │   │  • SMILES search  │
        │  • Suggestions    │   │  • Filtering      │
        └───────────────────┘   └─────────┬─────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE LAYER                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Qdrant Cloud Collections                    │  │
│  │                                                           │  │
│  │  ┌────────────────┐  ┌────────────────┐                 │  │
│  │  │research_papers │  │   proteins     │                 │  │
│  │  │   (384-dim)    │  │   (384-dim)    │                 │  │
│  │  │                │  │                │                 │  │
│  │  │ • Title        │  │ • Name         │                 │  │
│  │  │ • Abstract     │  │ • Function     │                 │  │
│  │  │ • Authors      │  │ • Gene names   │                 │  │
│  │  │ • PMIDs        │  │ • GO terms     │                 │  │
│  │  └────────────────┘  └────────────────┘                 │  │
│  │                                                           │  │
│  │  ┌────────────────┐  ┌────────────────┐                 │  │
│  │  │chemical_       │  │clinical_trials │                 │  │
│  │  │ compounds      │  │   (384-dim)    │                 │  │
│  │  │ (2048-dim FP)  │  │                │                 │  │
│  │  │                │  │ • Title        │                 │  │
│  │  │ • SMILES       │  │ • Phase        │                 │  │
│  │  │ • Formula      │  │ • Status       │                 │  │
│  │  │ • CID          │  │ • NCT ID       │                 │  │
│  │  └────────────────┘  └────────────────┘                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │
┌────────────────────────────┴─────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   EnhancedIngestionManager                               │  │
│  │   (enhanced_ingestion_manager.py)                        │  │
│  └──────┬────────┬─────────┬──────────┬──────────────────────┘  │
│         │        │         │          │                         │
│    ┌────▼───┐ ┌─▼────┐ ┌──▼─────┐ ┌──▼──────┐                 │
│    │PubMed  │ │PubChem│ │UniProt │ │Clinical │                 │
│    │        │ │       │ │        │ │Trials   │                 │
│    │Papers  │ │Compds │ │Proteins│ │         │                 │
│    └────────┘ └───────┘ └────────┘ └─────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                             ▲
                             │
┌────────────────────────────┴─────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                         │
│                                                                  │
│  • PubMed (35M+ papers)                                         │
│  • PubChem (110M+ compounds)                                    │
│  • UniProt (200M+ proteins)                                     │
│  • ClinicalTrials.gov (400K+ trials)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### 1. **Ingestion Pipeline**

```
User Query
    │
    ▼
EnhancedIngestionManager
    │
    ├──► PubMed API ──────► Papers (JSON)
    │       │
    │       ├──► Extract compound names
    │       │       │
    │       │       ▼
    │       └──► PubChem API ──► Compounds (JSON)
    │
    ├──► Extract protein mentions
    │       │
    │       ▼
    │   UniProt API ──────► Proteins (JSON)
    │
    └──► ClinicalTrials.gov ──► Trials (JSON)
            │
            ▼
    Cross-Modal Linking
            │
            ▼
    Save to data/raw/
```

### 2. **Upload to Qdrant**

```
data/raw/*.json
    │
    ▼
EmbeddingGenerator
    │
    ├──► Papers → sentence-transformers → 384-dim vectors
    ├──► Proteins → sentence-transformers → 384-dim vectors
    ├──► Trials → sentence-transformers → 384-dim vectors
    └──► Compounds → RDKit fingerprints → 2048-dim vectors
            │
            ▼
    EnhancedQdrantManager
            │
            ▼
    Upload to Qdrant Cloud Collections
```

### 3. **Query Pipeline**

```
User Question
    │
    ▼
LLMQueryProcessor (Groq)
    │
    ├──► Extract search intent
    ├──► Identify entities (compounds, proteins, diseases)
    └──► Refine query
            │
            ▼
    CloudSearch (Qdrant)
            │
            ├──► Search papers collection
            ├──► Search compounds collection
            ├──► Search proteins collection
            └──► Search trials collection
                    │
                    ▼
            BiologicalResearchAgent
                    │
                    ├──► Build context from results
                    └──► Generate response with Groq
                            │
                            ▼
                    Return AgentResponse
                    (answer + sources + suggestions)
```

---

## 📦 Module Breakdown

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `agent.py` | AI orchestration | `BiologicalResearchAgent`, `AgentResponse` |
| `llm_query.py` | Query understanding | `LLMQueryProcessor`, `SearchIntent` |
| `search.py` | Vector search | `CloudSearch` |
| `embeddings.py` | Text/structure embeddings | `EmbeddingGenerator` |

### Ingestion Modules

| Module | Data Source | Model |
|--------|-------------|-------|
| `pubmed_ingestor.py` | NCBI PubMed | `PubMedPaper` |
| `pubchem_ingestor.py` | PubChem | `ChemicalCompound` |
| `uniprot_ingestor.py` | UniProt | `Protein` |
| `clinical_trials_ingestor.py` | ClinicalTrials.gov | `ClinicalTrial` |

### Infrastructure

| Module | Purpose |
|--------|---------|
| `enhanced_qdrant_setup.py` | Manage Qdrant collections |
| `enhanced_ingestion_manager.py` | Orchestrate multi-modal ingestion |
| `enhanced_app.py` | Streamlit UI |

---

## 🔗 Entity Linking

### Bidirectional Links

```
Paper ←──────→ Compound
  │              │
  │              │
  ├──────────────┼──────→ Clinical Trial
  │              │
  ▼              ▼
Protein ←──────→ (via PMIDs)
```

### Link Types

1. **Papers ↔ Compounds**
   - Compounds mentioned in paper abstracts
   - Papers stored in `compound.source_pmids`

2. **Papers ↔ Proteins**
   - Proteins have `source_pmids` from references
   - Papers reference proteins via MeSH terms

3. **Compounds ↔ Trials**
   - Trials list compounds in `interventions`
   - Compounds link to trials via `trial_ncts`

4. **Trials ↔ Papers**
   - Trials have `related_pmids` from references

---

## 🚀 Scaling Strategy

### Current (Demo)
- **Papers**: 20-50
- **Compounds**: 10-30
- **Proteins**: 5-20
- **Trials**: 5-15
- **Total vectors**: ~100
- **Storage**: ~10MB

### Medium Scale
- **Papers**: 1,000
- **Compounds**: 500
- **Proteins**: 200
- **Trials**: 100
- **Total vectors**: ~2,000
- **Storage**: ~100MB

### Production Scale
- **Papers**: 100,000+
- **Compounds**: 10,000+
- **Proteins**: 5,000+
- **Trials**: 1,000+
- **Total vectors**: ~120,000+
- **Storage**: ~5GB

### Scaling Considerations

1. **Qdrant Scaling**
   - Use sharding for >1M vectors
   - Enable quantization for memory
   - Consider HNSW parameter tuning

2. **Ingestion Parallelization**
   - Batch API calls
   - Async processing
   - Rate limit management

3. **Embedding Caching**
   - Cache generated embeddings
   - Incremental updates only

---

## 🔐 Security & Privacy

- API keys stored in `.env` (not in git)
- No user data stored
- Read-only access to public databases
- Qdrant Cloud uses TLS encryption

---

## 📊 Performance Metrics

### Query Latency

| Operation | Time |
|-----------|------|
| Intent extraction | ~200ms |
| Vector search (1 collection) | <50ms |
| Vector search (4 collections) | <200ms |
| LLM generation | ~1-2s |
| **Total** | **~2-3s** |

### Ingestion Throughput

| Source | Rate |
|--------|------|
| PubMed | ~10 papers/min |
| PubChem | ~30 compounds/min |
| UniProt | ~20 proteins/min |
| ClinicalTrials | ~15 trials/min |

---

## 🎯 Future Enhancements

1. **Additional Modalities**
   - Genes (NCBI Gene)
   - Patents (Google Patents)
   - Pathways (KEGG, Reactome)

2. **Advanced Features**
   - Real-time paper alerts
   - Collaborative annotations
   - Citation networks
   - Knowledge graphs

3. **Infrastructure**
   - REST API
   - Batch processing queue
   - Monitoring dashboard

---

This architecture provides a solid foundation for a comprehensive biological research platform that can scale from demo to production while maintaining performance and accuracy.

## 📚 Resources

- [Groq Documentation](https://console.groq.com/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [UniProt API](https://www.uniprot.org/help/api)
- [ClinicalTrials.gov API](https://clinicaltrials.gov/api/gui)
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)

---

## 🤝 Contributing

Want to add more modalities? Here's how:

1. Create new ingestor in `src/` (e.g., `gene_ingestor.py`)
2. Add to `EnhancedIngestionManager`
3. Create Qdrant collection in `qdrant_setup.py`
4. Update agent to search new collection

---

## 📝 License

[Your License Here]

---

## 🙏 Acknowledgments

- **Qdrant** for vector database
- **Groq** for fast LLM inference
- **NCBI** for PubMed and PubChem
- **UniProt** for protein data
- **ClinicalTrials.gov** for trial data

---

## 📧 Contact

[wajdi.kalthoum@ept.ucar.tn]

---

**Happy Researching!** 🧬🔬✨
