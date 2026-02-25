# Agent-Assisted Narrative Drift Detection

Master Dissertation Project  
Remus Mihai Prodan  
Technical University of Cluj-Napoca  

---

## 1. Project Overview

This project investigates narrative drift in online news media — the measurable semantic evolution of how the same event or topic is framed over time.

The system combines:
- Document-level semantic drift detection
- Entity-level framing analysis
- Statistical change-point detection
- Agent-assisted interpretation (CrewAI)

---

## 2. Research Objectives

- Model narrative evolution as a temporal semantic signal
- Detect statistically significant narrative shifts
- Analyze entity-level framing changes
- Provide interpretable explanations using AI agents

---

## 3. Methodological Framework

### 3.1 Data Collection
- Selection of topic(s) and news sources
- Temporal segmentation of articles

### 3.2 Document-Level Drift
- Sentence-BERT embeddings
- Aggregation per time window
- Cosine-based drift signal

### 3.3 Change-Point Detection
- Drift time-series analysis
- Identification of significant structural breaks

### 3.4 Entity-Level Framing Drift
- Named Entity Recognition
- Contextual embedding comparison across time

### 3.5 Agent-Assisted Interpretation
- Multi-agent analysis using CrewAI
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