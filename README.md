# Narrative Drift Detection

A research framework for detecting, analysing and explaining **narrative drift** in online news.

Developed as part of a Master's dissertation at the Technical University of Cluj-Napoca, the project combines semantic analysis, entity evolution, framing analysis, temporal segmentation and local large language models into a unified pipeline for longitudinal news analysis.

---

## Motivation

News narratives rarely change abruptly. Instead, they gradually evolve as new actors emerge, topics shift, sentiment changes and different outlets adopt new framing strategies.

This project aims to automatically identify these transitions and provide interpretable explanations supported by evidence extracted from the underlying articles.

---

## Main Features

- Multi-provider news collection
- Full-text enrichment
- Sentence embedding generation
- Semantic drift computation
- Named entity extraction and normalization
- Temporal entity evolution
- Latent frame discovery
- LLM-assisted frame labeling
- Narrative signatures
- Cross-source comparison
- Change-point detection
- Narrative regime analysis
- Sentiment and affective dynamics
- Editorial behaviour analysis
- Confidence scoring
- Evidence retrieval
- Multi-agent narrative explanation using Ollama
- Actor ↔ latent-frame graph and narrative-ecosystem community detection

---

## High-Level Pipeline

```mermaid
flowchart LR
A[News Providers] --> B[Collection]
B --> C[Enrichment]
C --> D[Preprocessing]
D --> E[Semantic Analysis]
D --> F[Entity Analysis]
D --> G[Frame Analysis]
D --> H[Sentiment Analysis]
E --> I[Evidence Builder]
F --> I
G --> I
H --> I
I --> J[LLM Agents]
J --> K[Narrative Explanation]
```

---

## Project Structure

```text
src/
├── main.py                    # Pipeline orchestrator (entry point)
├── collector.py                # Multi-provider article metadata collection
├── enricher.py                  # Full-text extraction
├── providers/                    # GDELT / Guardian source adapters
├── preprocessing.py, database.py, embeddings.py
├── drift.py                     # Semantic drift (cosine distance over embeddings)
├── entities.py, entity_normalization.py, temporal_entity_analysis.py
├── entity_framing_drift.py, actor_salience.py, dynamic_entity_ecosystem.py
├── actor_graph.py, actor_frame_graph.py, entity_latent_frames.py,
│   frame_normalization.py, llm_frame_labeler.py, frame_cache_db.py
├── latent_frames.py, semantic_frame_labeling.py, temporal_frame_evolution.py
├── narrative_signatures.py, signature_comparison.py, narrative_archetypes.py
├── change_point_detection.py, temporal_narrative_regimes.py
├── cross_source_divergence.py, confidence_scoring.py
├── sentiment_analysis.py, affective_dynamics.py, editorial_behavior.py
├── evidence_packet_builder.py, evidence_retrieval.py
├── agentic_explainers/           # LLM-based evidence-grounded explanations
├── filters/, migrations/
└── utils/
```

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

For local LLM support:

```bash
ollama pull qwen2.5:7b
ollama serve
```

---

## Usage

Collect metadata:

```bash
python src/collector.py
```

Extract article text:

```bash
python src/enricher.py
```

Run the complete analysis:

```bash
python src/main.py
```

`main.py` is configured through module-level constants near the top of the
file (`TOPIC_FILTER`, `DEBUG_SOURCES`, and a set of `SKIP_*` flags). Several
stages — LLM frame labeling, the actor↔frame graph, evidence packets and
agentic explanations — require a local Ollama server (see below) and are
disabled by default; flip the corresponding `SKIP_*` flag to `False` to
enable them.

---

## Visualisations

## Narrative Dashboard

![Narrative Dashboard](docs/assets/narrative_dashboard.png)

## Semantic Drift Across Sources

![Semantic Drift](docs/assets/semantic_drift_comparison.png)

## Semantic vs Framing Drift

![Semantic vs Framing](docs/assets/semantic_vs_framing_drift.png)

## Entity Framing Heatmap

![Heatmap](docs/assets/entity_framing_heatmap.png)

---

## Live Demo

A Gradio demo (`app.py`) runs a bounded, live subset of this pipeline over
a small bundled sample of articles — see [DEPLOY.md](DEPLOY.md) for how to
publish it as a Hugging Face Space. The full pipeline above always runs
locally against the complete research database.


---

## Technologies

- Python
- SQLite
- Sentence Transformers
- spaCy
- scikit-learn
- pandas
- NumPy
- ruptures
- VADER Sentiment
- Ollama

---

## Research Context

Unlike traditional concept drift detection, this framework studies how narratives evolve through multiple complementary perspectives. Instead of relying solely on semantic similarity, it integrates entity evolution, latent framing, sentiment dynamics, temporal change detection and evidence-driven LLM interpretation into a single research pipeline capable of identifying and explaining narrative shifts across time and news sources.
