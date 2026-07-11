---
title: Narrative Drift Detection
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
python_version: "3.11"
---

# Narrative Drift Detection

A research framework for detecting, analysing and explaining **narrative drift**
in online news, developed as part of a Master's dissertation at the Technical
University of Cluj-Napoca.

This Space runs a real, live subset of the full pipeline over a small bundled
sample of Ukraine-war coverage (BBC, CNN, The Guardian; Jan-Aug 2022; ~480
articles, excerpts truncated to 800 characters each) rather than the full
~40,000-article research database, and uses the Hugging Face Inference API
for LLM-based frame labeling instead of a local Ollama server.

Pick a source, optionally enable the LLM-based stages (latent frame labeling
+ actor-frame graph), and click **Run analysis** to see:
- Semantic drift between consecutive two-month periods
- Sentiment evolution
- Top entities by framing-drift importance
- Editorial-behavior classification and a confidence score
- (optional) LLM-labeled narrative frames and an actor↔frame community graph

The full pipeline (entity framing drift, narrative signatures/archetypes,
cross-source divergence, change-point detection, evidence retrieval, and
more) runs locally against the complete research database — see the
[project repository](https://github.com/Prodan-Remus-30123/narrative-drift-detection)
for the code and `README.md` for the full write-up.
