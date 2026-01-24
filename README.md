# 🧬 Bio-Search

AI-powered research assistant that searches across biological databases: papers, compounds, proteins, and clinical trials.

---

## ✨ Features

- 🤖 **AI Chat Interface** - Ask questions in natural language
- 📊 **Multi-Source Search** - PubMed, PubChem, UniProt, ClinicalTrials.gov
- 🔗 **Smart Linking** - Connects related papers, compounds, proteins, and trials
- 💬 **Conversational** - Maintains context and suggests follow-up questions

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Bio-Search.git
cd Bio-Search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys

Create a `.env` file:

```env
# Get from https://console.groq.com
GROQ_API_KEY=your_groq_api_key

# Get from https://cloud.qdrant.io
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Optional: for better PubMed access
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=your_ncbi_key
```

### 3. Load Data

```bash
# Example: Load data about KRAS inhibitors
python -c "
from src.enhanced_ingestion_manager import EnhancedIngestionManager

manager = EnhancedIngestionManager()
manager.ingest_comprehensive(
    query='KRAS inhibitor lung cancer',
    max_papers=20,
    max_trials=10,
    include_proteins=True,
    include_trials=True
)
"

# Upload to database
python -m src.qdrant_setup --create --populate data/raw
```

### 4. Launch App

```bash
streamlit run enhanced_app.py
```

Open `http://localhost:8501` in your browser.

---

## 💡 Usage

### Chat Interface

Simply ask questions like:
- "What are KRAS inhibitors for lung cancer?"
- "Find compounds similar to aspirin"
- "Show me clinical trials for CDK4/6 inhibitors"

### Python API

```python
from src.agent import BiologicalResearchAgent

agent = BiologicalResearchAgent()
response = agent.query("What are KRAS inhibitors?")

print(response.answer)
for source in response.sources:
    print(f"- {source['title']}: {source['url']}")
```

---

## 📁 Project Structure

```
Bio-Search/
├── src/
│   ├── agent.py                          # AI agent
│   ├── enhanced_ingestion_manager.py     # Data loader
│   ├── pubmed_ingestor.py               # Papers
│   ├── pubchem_ingestor.py              # Compounds
│   ├── uniprot_ingestor.py              # Proteins
│   ├── clinical_trials_ingestor.py      # Trials
│   ├── llm_query.py                     # Query processing
│   ├── search.py                        # Vector search
│   └── embeddings.py                    # Text embeddings
├── enhanced_app.py                       # Streamlit UI
├── .env                                  # API keys (create this)
└── requirements.txt
```

---

## 🏗️ Architecture

```
User Question
    ↓
AI Agent (Groq LLM)
    ↓
Vector Search (Qdrant)
    ↓
Multiple Databases
  ├─ PubMed (papers)
  ├─ PubChem (compounds)
  ├─ UniProt (proteins)
  └─ ClinicalTrials.gov
    ↓
AI Response + Sources
```

---

## 🔧 Troubleshooting

**"Agent not available"**
- Check `GROQ_API_KEY` in `.env` file

**"Qdrant connection failed"**
- Verify `QDRANT_URL` and `QDRANT_API_KEY` are correct

**"No results found"**
- Make sure you've run the ingestion step
- Try a different search query

---

## 📊 What You Can Search

| Type | Source | Example Query |
|------|--------|---------------|
| Papers | PubMed | "CRISPR gene editing" |
| Compounds | PubChem | "aspirin structure" |
| Proteins | UniProt | "KRAS protein function" |
| Trials | ClinicalTrials.gov | "breast cancer trials" |

---

## 🛠️ Requirements

- Python 3.11+
- Groq API key (free at console.groq.com)
- Qdrant Cloud account (free tier available)

---

## 📝 License

MIT License

---

## 📧 Contact

wajdi.kalthoum@ept.ucar.tn

---


