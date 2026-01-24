# Project Brief: Multimodal Biological Discovery Intelligence Platform

## 📋 Project Overview

**Objective**: Build a production-grade multimodal search platform that connects scientific literature, chemical compounds, and biological sequences into a unified knowledge graph.

**Core Technology**: Qdrant vector database + sentence transformers + RDKit + LLM integration

**Target Users**: Biomedical researchers, drug discovery teams, computational biologists

---

## 🎯 Session Log

### Session 1: Foundation Setup (Current)
**Date**: January 21, 2026
**Status**: ✅ Complete

**Completed**:
- [x] Created project directory structure
- [x] Generated requirements.txt with all dependencies
- [x] Created docker-compose.yml for Qdrant orchestration
- [x] Initialized Git repository structure
- [x] Created comprehensive README.md
- [x] Created .gitignore with proper exclusions
- [x] Documented project brief for session continuity

**Key Decisions**:
1. **LLM Provider**: Using Groq API (due to API key availability) instead of Claude
2. **Embeddings**: 
   - Text: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, good quality)
   - Compounds: RDKit Morgan Fingerprints (2048-bit, standard in cheminformatics)
3. **Vector DB**: Qdrant (open-source, fast, excellent filtering capabilities)
4. **Web Framework**: Streamlit (rapid prototyping) + FastAPI (future scalability)

**Directory Structure Created**:
```
demo/
├── src/
│   ├── data/          # Data ingestion modules (Phase 2)
│   ├── embeddings/    # Embedding generators (Phase 3)
│   ├── vector_db/     # Qdrant management (Phase 3)
│   └── search/        # Search functions (Phase 3)
├── app/               # Web interface (Phase 4)
├── data/
│   ├── raw/           # Original API responses
│   └── processed/     # Cleaned, structured data
├── notebooks/         # Exploratory analysis
└── tests/             # Unit tests
```

### Session 2: Data Ingestion System (Current)
**Date**: January 21, 2026
**Status**: ✅ Complete

**Completed**:
- [x] Created PubMed ingestor with NCBI E-utilities integration
- [x] Created PubChem ingestor with REST API + pubchempy
- [x] Built ingestion manager for orchestration
- [x] Implemented bidirectional linking (papers ↔ compounds)
- [x] Added compound name extraction (regex-based)
- [x] Created comprehensive test suite
- [x] Added quickstart script for testing

**Key Features Implemented**:
1. **PubMed Ingestor** (pubmed_ingestor.py):
   - Rate limiting (3 req/sec)
   - Search with date filters and MeSH terms
   - XML parsing for metadata extraction
   - Compound mention detection
   - Pydantic models for data validation

2. **PubChem Ingestor** (pubchem_ingestor.py):
   - Multiple search methods (name, SMILES, CID)
   - RDKit integration for property calculation
   - SMILES validation
   - Synonym handling
   - 2D structure URL generation

3. **Ingestion Manager** (ingestion_manager.py):
   - Complete pipeline orchestration
   - Bidirectional paper-compound linking
   - Batch processing for multiple topics
   - JSON output with structured data
   - Comprehensive statistics tracking

**Data Models**:
- `PubMedPaper`: PMID, title, abstract, authors, journal, date, MeSH terms, mentioned_compounds
- `ChemicalCompound`: CID, name, SMILES, IUPAC, formula, weight, properties, source_pmids

**Files Created**:
- src/data/pubmed_ingestor.py (270 lines)
- src/data/pubchem_ingestor.py (320 lines)
- src/data/ingestion_manager.py (280 lines)
- src/data/__init__.py
- tests/test_ingestion.py (180 lines)
- quickstart_ingestion.py (110 lines)

**Next Session**: Phase 3 - Embedding Generation & Qdrant Setup
- Text embedding generation (sentence-transformers)
- Chemical fingerprint generation (RDKit)
- Qdrant collection creation and schema
- Vector database population
- Multimodal search functions

---

## 🔧 Technical Architecture

### Data Flow
```
1. Data Sources (PubMed, PubChem)
   ↓
2. Ingestion Layer (API clients, parsers)
   ↓
3. Embedding Generation (text → vectors, SMILES → fingerprints)
   ↓
4. Qdrant Vector DB (collections: papers, compounds)
   ↓
5. Search Layer (multimodal queries, filtering)
   ↓
6. LLM Enhancement (query understanding, summarization)
   ↓
7. Web Interface (Streamlit app)
```

### Collections Schema (Planned)

**Collection: research_papers**
- **Vector**: 384-dim text embedding
- **Payload**:
  - `pmid`: str
  - `title`: str
  - `abstract`: str
  - `authors`: List[str]
  - `journal`: str
  - `publication_date`: str
  - `mentioned_compounds`: List[str]
  - `mesh_terms`: List[str]

**Collection: chemical_compounds**
- **Vector**: 2048-bit Morgan fingerprint (as float array)
- **Payload**:
  - `cid`: int
  - `name`: str
  - `smiles`: str
  - `iupac_name`: str
  - `molecular_weight`: float
  - `molecular_formula`: str
  - `source_pmids`: List[str]

---

## 🚀 Running Instructions

### Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "GROQ_API_KEY=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" > .env

# 4. Start Qdrant
docker-compose up -d

# 5. Verify Qdrant is running
curl http://localhost:6333/
```

### Development Workflow
```bash
# Check Qdrant status
docker-compose ps

# View Qdrant logs
docker-compose logs -f qdrant

# Stop services
docker-compose down

# Remove all data (fresh start)
docker-compose down -v
rm -rf qdrant_storage/
```

---

## 📝 Notes for Next Session

### Phase 2 Prerequisites
- Ensure Qdrant is running (`docker-compose up -d`)
- Verify internet connection (for API calls)
- NCBI requires email for Entrez API - add to .env

### Phase 2 Implementation Plan
1. **PubMed Ingestor** (~100 lines)
   - Use `Bio.Entrez` for API access
   - Implement search with filters (date range, MeSH terms)
   - Parse XML responses
   - Extract: PMID, title, abstract, authors, journal, date
   - Rate limiting: max 3 requests/second
   - Save as Pydantic models

2. **PubChem Ingestor** (~80 lines)
   - Use `pubchempy` or direct REST API
   - Search by: name, SMILES, CID
   - Extract: CID, canonical SMILES, MW, formula, IUPAC name
   - Handle synonyms and multiple matches
   - Error handling for invalid structures

3. **Ingestion Manager** (~60 lines)
   - Orchestrate: search PubMed → extract compounds → fetch from PubChem
   - Example query: "KRAS inhibitor"
   - Named Entity Recognition for compounds (simple regex initially)
   - Save outputs to `data/raw/papers.json` and `data/raw/compounds.json`

### Testing Strategy
- Unit tests for each ingestor
- Mock API responses to avoid rate limits during testing
- Validate Pydantic models
- Check data quality (no null values in required fields)

---

## 🐛 Known Issues & Limitations

### Current
- None (Phase 1 complete)

### Anticipated
- PubMed rate limits: 3 req/sec without API key, 10 req/sec with key
- PubChem throttling: Use delays between requests
- Compound name extraction: Will be naive initially (improve with NER in Phase 5)
- SMILES validation: Some strings may be invalid (handle with RDKit)

---

## 📚 References & Resources

### APIs
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/)
- [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- [Groq API Docs](https://console.groq.com/docs)

### Libraries
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [Sentence Transformers](https://www.sbert.net/)
- [RDKit Documentation](https://www.rdkit.org/docs/GettingStartedInPython.html)

### Models
- Text Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: Groq (Mixtral or Llama models)

---

## 💡 Future Enhancements (Post-MVP)

- Protein sequence embeddings (ESM-2 model)
- Image similarity search (microscopy, protein structures)
- Knowledge graph visualization (NetworkX + Plotly)
- Batch processing for large-scale ingestion
- API endpoint for programmatic access
- Real-time alerts for new papers matching criteria
- Integration with lab notebooks (ELN systems)
- Export to standard formats (BioPAX, SBML)