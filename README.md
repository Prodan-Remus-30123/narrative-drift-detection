# 📰 Narrative Drift Detection System

**Master's Dissertation Project**  
**Remus Mihai Prodan**  
**Technical University of Cluj-Napoca**

---

## 🎯 Quick Start

```bash
# 1. Setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start Ollama (for LLM features)
ollama serve

# 3. Collect data
python src/collector.py       # Download metadata
python src/enricher.py        # Extract full text

# 4. Run full analysis
python src/main.py

# 5. Get LLM explanations (optional)
python src/test_agentic_system.py
```

For detailed setup, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📊 Project Overview

This system investigates **narrative drift** — how news media evolve their coverage and framing of events over time.

### Example Questions Answered:
- *"How did CNN's portrayal of China change during COVID?"*
- *"What are key differences between BBC and Washington Post narratives?"*
- *"When did significant shifts in coverage occur?"*

### 8 Types of Analysis:
1. **Semantic Drift** - Document embeddings changing over time
2. **Entity Framing** - Which actors get associated with which actions
3. **Latent Frames** - Automatically discovered semantic patterns
4. **Narrative Signatures** - Distinctive per-source profiles
5. **Temporal Analysis** - Statistical breakpoints in narratives
6. **Affective Dynamics** - Sentiment evolution and relationships
7. **Editorial Patterns** - Source-level bias and behavior classification
8. **AI Interpretation** - LLM-generated natural language explanations

---

## 🏗️ System Architecture

The system is organized into **7 analytical layers** plus supporting infrastructure:

```
Layer 0: Infrastructure
├─ database.py, embeddings.py, preprocessing.py

Layer 1: Data Collection & Enrichment
├─ collector.py, enricher.py, providers/

Layer 2: Entity Recognition & Temporal Organization
├─ entities.py, entity_normalization.py, temporal_entity_analysis.py

Layer 3: Semantic Drift Detection
├─ drift.py, embeddings.py

Layer 4: Frame Discovery & LLM Labeling
├─ latent_frames.py, semantic_frame_labeling.py, llm_frame_labeler.py

Layer 5: Narrative Signatures & Cross-Source Comparison
├─ narrative_signatures.py, signature_comparison.py, entity_frame_alignment.py

Layer 6: Temporal Dynamics & Editorial Analysis
├─ changepoints.py, affective_dynamics.py, sentiment_analysis.py, editorial_behavior.py

Layer 7: Evidence Packaging & AI Interpretation
├─ evidence_packet_builder.py, agentic_llm/agentic_runner.py, agentic_explainers/
```

**Complete documentation**: See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed layer breakdown.

---

## 🚀 Installation

### Requirements
- Python 3.13+
- 8GB RAM minimum
- Ollama (for LLM features)

### Setup Steps

```bash
# Clone/navigate to project
cd e:\Facultate\Master\ProiectDisertatie

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download NLP model
python -m spacy download en_core_web_sm

# Setup Ollama
# 1. Download from https://ollama.ai
# 2. Run: ollama pull qwen2.5:7b
# 3. Start server: ollama serve
```

### Configuration

Create `.env` file in project root:
```env
GUARDIAN_API_KEY=your_api_key_here    # Optional: For Guardian news source
DATABASE_PATH=database/articles.db
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 📍 Entry Points

### **Main Analysis Pipeline**
```bash
python src/main.py
```
Runs all 7 layers. Outputs evidence packets and visualizations.

### **Data Collection**
```bash
python src/collector.py       # Get metadata from APIs
python src/enricher.py        # Extract full article text
```

### **LLM Agent Testing**
```bash
python src/test_agentic_system.py
```
Tests AI explanations using pre-computed evidence.

---

## 📂 Project Structure

```
src/
├── main.py                          # Main analysis pipeline
├── collector.py                     # Data collection from GDELT/Guardian
├── enricher.py                      # Full-text article extraction
├── database.py                      # SQLite interface
├── embeddings.py                    # Sentence-BERT wrapper
├── preprocessing.py                 # Text processing
│
├── entities.py                      # NER + entity extraction
├── entity_normalization.py          # Entity deduplication
├── temporal_entity_analysis.py      # Time period grouping
│
├── drift.py                         # Semantic drift detection
├── changepoints.py                  # Statistical breakpoint detection
│
├── latent_frames.py                 # Frame discovery (clustering)
├── frame_normalization.py           # Frame deduplication
├── semantic_frame_labeling.py       # LLM-based frame labeling
├── llm_frame_labeler.py            # Ollama integration
│
├── narrative_signatures.py          # Per-source narrative profiles
├── signature_comparison.py          # Cross-source comparison
├── entity_frame_alignment.py        # Entity-frame transitions
├── semantic_signature_embedding.py  # Signature vectorization
│
├── affective_dynamics.py            # Sentiment evolution
├── sentiment_analysis.py            # Sentiment scoring
├── emotional_volatility.py          # Sentiment consistency
├── editorial_behavior.py            # Editorial bias classification
├── correlation_analysis.py          # Sentiment-framing correlation
│
├── evidence_packet_builder.py       # Result packaging
├── interpreter.py                   # Narrative interpretation
│
├── agentic_llm/                    # LLM agent orchestration
│   ├── llm_client.py
│   ├── agentic_runner.py
│   ├── semantic_agent.py
│   ├── framing_agent.py
│   └── synthesis_agent.py
│
├── agentic_explainers/             # Explanation generators
│   └── explanation_runner.py
│
├── agentic_tools/                  # Tool registry
│   ├── framing_tools.py
│   ├── salience_tools.py
│   └── ...
│
├── providers/                       # Data source integrations
│   ├── raw_gdelt_provider.py
│   ├── guardian_provider.py
│   └── gdelt_query_builder.py
│
├── utils/                          # Utility functions
│   ├── date_utils.py
│   ├── deduplication.py
│   ├── filtering.py
│   └── period_sorting.py
│
├── filters/                        # Filtering blacklists
│   ├── entity_filters.py
│   └── verb_filters.py
│
└── plots/                          # Visualization outputs
    ├── plot_semantic_drift.py
    ├── plot_entity_evolution.py
    └── ...

database/
└── articles.db                      # SQLite database

ARCHITECTURE.md                      # Detailed technical documentation
```

---

## 🔄 Data Pipeline

1. **Collection** → `collector.py` queries APIs, inserts metadata
2. **Enrichment** → `enricher.py` extracts full article text
3. **Processing** → `main.py` runs all 7 analytical layers
4. **Interpretation** → `agentic_runner.py` generates explanations
5. **Output** → JSON packets + visualizations

---

## 💾 Database Schema

```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    provider TEXT,              -- "RawGDELT", "Guardian"
    source TEXT,                -- "bbc.co.uk", "cnn.com"
    date TEXT,                  -- Publication date
    title TEXT UNIQUE,
    url TEXT UNIQUE,
    text LONGBLOB,              -- Full article body
    ingestion_status TEXT,      -- "pending", "ingested"
    extraction_status TEXT,     -- "pending", "success", "failed"
    extraction_attempts INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 🤖 LLM Integration

The system uses **Ollama + Qwen2.5:7b** for local LLM inference.

### Setup:
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:7b
ollama serve
```

### Why Local LLM?
- No API costs
- Complete data privacy
- Full control over model
- Runs locally on 8GB RAM

### How It Works:
1. Extract evidence from analysis layers
2. Package as structured JSON
3. Route to specialized agents (semantic, framing, affective)
4. Each agent generates explanation
5. Synthesis agent combines into narrative

---

## 📊 Output Examples

### Evidence Packet (JSON)
```json
{
  "source": "cnn.com",
  "semantic_drift": [
    {"period": "2020-01", "drift": 0.23},
    {"period": "2020-02", "drift": 0.31}
  ],
  "top_entities": [
    {"entity": "China", "importance": 0.89},
    {"entity": "WHO", "importance": 0.76}
  ],
  "frames": [
    {
      "frame": "China as scapegoat",
      "evidence": ["blamed", "criticized", "accused"],
      "trend": "increasing"
    }
  ]
}
```

### Visualizations
- `plot_semantic_drift.png` - Drift over time
- `plot_entity_evolution.png` - Entity importance trends
- `plot_signature_comparison.png` - Source comparison

---

## 🧪 Testing

```bash
# Test entire pipeline
python src/main.py

# Test LLM agents only
python src/test_agentic_system.py

# Test specific component
python -c "from src.drift import compute_cosine_drift; print('Import OK')"
```

---

## ⚙️ Configuration

### LLM Settings (agentic_llm/llm_client.py)
- `OLLAMA_BASE_URL = "http://localhost:11434"`
- `OLLAMA_MODEL = "qwen2.5:7b"`
- `OLLAMA_TEMPERATURE = 0.7`

### Database (database.py)
- `DB_PATH = "database/articles.db"`

### Data Collection (collector.py)
- Topics: COVID-19 related
- Sources: BBC, CNN, NYT, Washington Post
- Providers: GDELT (free), Guardian (requires API key)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not responding | `ollama serve` in separate terminal |
| Spacy model missing | `python -m spacy download en_core_web_sm` |
| Database locked | Ensure only one process runs at a time |
| Memory issues | Reduce batch sizes in `main.py` |
| Rate limited | Wait 60s before retrying collector |

See [ARCHITECTURE.md](ARCHITECTURE.md#-troubleshooting) for more solutions.

---

## 📚 Research Methodology

### Layers & Techniques:

| Layer | Technique | Library |
|-------|-----------|---------|
| Embeddings | Sentence-BERT | sentence-transformers |
| NER | Spacy pipeline | spacy |
| Drift | Cosine similarity | scikit-learn |
| Clustering | Hierarchical | scikit-learn |
| Changepoints | PELT algorithm | ruptures |
| Sentiment | VADER | vaderSentiment |
| LLM | Inference | ollama |

---

## 🗑️ Dead Code Cleanup

Files safe to delete:
- `run_agent.py` - Broken imports
- `data_collection.py` - Replaced by collector.py + enricher.py
- `corpus_statistics.py` - Utility only
- `actor_graph.py` - Alternative not used
- `interpreter.py` - Superseded by agents

---

## 📖 Further Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete technical breakdown
- Module docstrings - API documentation
- `src/agentic_llm/` - Agent implementation details
- `src/orchestration/` - Orchestration system (future)

---

## 📞 Support

For issues or questions:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md#-troubleshooting)
2. Review module docstrings
3. Check terminal output for error messages
4. Ensure Ollama is running: `ollama serve`

---

**Status**: ✅ Production-Ready (Core Components)  
**Last Updated**: June 5, 2026  
**Python**: 3.13  
**Main Dependencies**: sentence-transformers, spacy, scikit-learn, pandas, ollama
- Structured explanation of narrative shifts

---

## 4. Expected Outputs

- Temporal drift visualization
- Detected change points
- Entity-level shift analysis
- AI-generated narrative explanations

---

## 5. Planned Repository Structure
```
narrative-drift-detection/
│
├── data/ # raw or processed datasets
│
├── notebooks/ # exploratory analysis & experiments
│
├── src/
│ ├── preprocessing.py # text cleaning & preparation
│ ├── embeddings.py # SBERT encoding logic
│ ├── drift.py # document-level drift computation
│ ├── changepoints.py # change-point detection methods
│ ├── entities.py # entity extraction & contextual drift
│ └── agents.py # CrewAI multi-agent orchestration
│
├── README.md
└── requirements.txt
```

## 6. Current Status

Project structure and core pipeline skeleton implemented.
Document-level embedding and drift computation modules initialized.
Planned next steps: real dataset integration and temporal drift signal generation.

## 7. Installation
```
pip install -r requirements.txt
```