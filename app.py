"""
app.py

Gradio demo for the Narrative Drift Detection pipeline (see README.md).

Runs the real analysis pipeline live -- not a canned/precomputed demo
-- but over a small bundled sample of articles
(data/sample_articles.db, see scripts/build_sample_dataset.py) rather
than the full ~40k-article local research database, and using the
Hugging Face Inference API instead of a local Ollama server for the
LLM-based stages (see src/llm_backend.py).
"""

# Must be imported before torch (transitively pulled in by
# sentence-transformers below) -- otherwise Gradio's background
# "spaces" reload-watcher thread crashes on Spaces with
# "CUDA has been initialized before importing the `spaces` package."
# Not installed outside a Space, hence the guard.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    # This Space is provisioned with ZeroGPU (ZeroGPU-A10G) hardware,
    # which refuses to start ("No @spaces.GPU function detected during
    # startup") unless at least one function is decorated -- even
    # though nothing in this demo actually needs a GPU (spaCy/sklearn/
    # sentence-transformers all run fine on CPU). This dummy function
    # satisfies that startup check without being wired into any real
    # analysis path.
    @spaces.GPU
    def _zerogpu_startup_check():
        return True

import os

os.environ.setdefault("ARTICLES_DB_PATH", "data/sample_articles.db")
os.environ.setdefault("LLM_BACKEND", "hf")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from database import load_full_articles
from temporal_entity_analysis import group_articles_by_period
from preprocessing import preprocess_corpus
from embedding_model_registry import get_embedding_model
from drift import compute_cosine_drift, compute_dynamic_threshold, classify_drift
from sentiment_analysis import analyze_sentiment
from entity_framing_drift import compute_entity_drift, compute_entity_importance
from actor_salience import compute_actor_salience, compute_total_actor_salience
from dynamic_entity_ecosystem import build_dynamic_entity_ecosystem
from editorial_behavior import classify_editorial_behavior
from confidence_scoring import compute_confidence_score
from utils.period_sorting import sort_period_key


TOPIC = "ukraine_war"
SOURCES = ["bbc.co.uk", "cnn.com", "theguardian.com"]
# Lower than main.py's 50: the bundled sample is deliberately small.
MIN_TOTAL_DOCS = 30


def _load_source_df(source):
    df = load_full_articles()
    return df[(df["topic"] == TOPIC) & (df["source"] == source)].copy()


def _plot_line(labels, series, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 4))

    for name, values in series.items():
        ax.plot(labels, values, marker="o", label=name)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

    if len(series) > 1:
        ax.legend()

    fig.tight_layout()
    return fig


def _plot_barh(labels, values, xlabel, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, values)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _run_llm_stages(source, grouped, framing_drift, entity_importance, salience_totals, summary_lines):
    from latent_frames import compute_frame_evolution_from_grouped
    from semantic_frame_labeling import label_all_latent_frames
    from actor_graph import build_actor_verb_graph, compute_actor_centrality
    from entity_latent_frames import compute_entity_latent_frame_transitions
    from actor_frame_graph import (
        build_actor_frame_graph_from_results,
        detect_actor_frame_communities,
        summarize_actor_frame_graph
    )

    latent_result = compute_frame_evolution_from_grouped(source, grouped)
    top_frame_ids = list(latent_result["clusters"].keys())[:5]
    filtered_latent_result = {
        **latent_result,
        "clusters": {fid: latent_result["clusters"][fid] for fid in top_frame_ids}
    }
    labeled_frames = label_all_latent_frames(filtered_latent_result)

    frame_labels = [f["frame_label"] for f in labeled_frames.values()]
    summary_lines.append("")
    summary_lines.append("**LLM-labeled latent frames:** " + ", ".join(frame_labels))

    actor_graph = build_actor_verb_graph(grouped)
    actor_centrality = compute_actor_centrality(actor_graph)
    ranked_actors = sorted(actor_centrality.items(), key=lambda x: x[1], reverse=True)
    top_actors = [actor for actor, _ in ranked_actors[:5]]

    actor_frame_results = {}
    for actor in top_actors:
        try:
            actor_frame_results[actor] = compute_entity_latent_frame_transitions(
                source=source,
                entity=actor,
                framing_drift=framing_drift,
                entity_importance=entity_importance,
                salience_totals=salience_totals
            )
        except Exception:
            continue

    result = build_actor_frame_graph_from_results(source, top_actors, actor_frame_results)
    graph_summary = summarize_actor_frame_graph(result["graph"])
    communities = detect_actor_frame_communities(result["graph"])

    graph_text = (
        f"Actor-frame graph: {graph_summary['nodes']} nodes, "
        f"{graph_summary['edges']} edges, {len(communities)} communities.\n\n"
    )

    for community in communities[:5]:
        graph_text += (
            f"- Community {community['community_id']}: "
            f"actors={community['actors']}, frames={community['frames'][:5]}\n"
        )

    return graph_text


def run_analysis(source, run_llm_stages, progress=gr.Progress()):
    progress(0, desc="Loading sample articles...")
    df = _load_source_df(source)

    if df.empty:
        raise gr.Error(f"No sample articles found for {source}.")

    grouped = group_articles_by_period(df)
    total_docs = sum(len(texts) for texts in grouped.values())

    if total_docs < MIN_TOTAL_DOCS:
        raise gr.Error(f"Only {total_docs} sample articles for {source} -- too few to analyze.")

    progress(0.1, desc="Preprocessing...")
    for period in grouped:
        grouped[period] = preprocess_corpus(grouped[period])

    periods = sorted(grouped.keys(), key=sort_period_key)

    progress(0.2, desc="Computing embeddings and semantic drift...")
    model = get_embedding_model()
    vectors = []

    for period in periods:
        embeddings = model.encode_documents(grouped[period])
        vectors.append((period, model.aggregate_embeddings(embeddings)))

    drift_labels, drift_values = [], []

    for i in range(len(vectors) - 1):
        m1, v1 = vectors[i]
        m2, v2 = vectors[i + 1]
        drift_labels.append(f"{m1}->{m2}")
        drift_values.append(compute_cosine_drift(v1, v2))

    threshold = compute_dynamic_threshold(drift_values, method="median_mad")
    drift_classes = [classify_drift(v, threshold) for v in drift_values]
    semantic_fig = _plot_line(
        drift_labels, {"Semantic drift": drift_values},
        "Cosine distance", f"{source}: Semantic Drift"
    )

    progress(0.4, desc="Analyzing sentiment...")
    sentiment_results = {period: analyze_sentiment(grouped[period]) for period in periods}
    sentiment_fig = _plot_line(
        periods,
        {
            "Compound": [sentiment_results[p]["compound"] for p in periods],
            "Positive": [sentiment_results[p]["positive"] for p in periods],
            "Negative": [sentiment_results[p]["negative"] for p in periods],
        },
        "Sentiment", f"{source}: Sentiment Evolution"
    )

    progress(0.6, desc="Computing entity framing drift...")
    framing_drift = compute_entity_drift(grouped)
    salience_totals = compute_total_actor_salience(compute_actor_salience(grouped))
    entity_importance = compute_entity_importance(framing_drift, salience_totals)

    top_entities = sorted(entity_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    entity_fig = _plot_barh(
        [entity for entity, _ in top_entities][::-1],
        [value for _, value in top_entities][::-1],
        "Importance (drift x salience)",
        f"{source}: Top Entities by Importance"
    )

    entity_ecosystem = build_dynamic_entity_ecosystem(framing_drift, entity_importance)

    avg_semantic = sum(drift_values) / len(drift_values) if drift_values else 0.0

    turnover_values = [
        stats.get("vocabulary_turnover")
        for entities in framing_drift.values()
        for stats in entities.values()
        if stats.get("vocabulary_turnover") is not None
    ]
    avg_framing = sum(turnover_values) / len(turnover_values) if turnover_values else 0.0
    avg_sentiment = sum(sentiment_results[p]["compound"] for p in periods) / len(periods)

    behavior = classify_editorial_behavior(avg_semantic, avg_framing, avg_sentiment)

    confidence = compute_confidence_score({
        "semantic_drift": {"labels": drift_labels, "values": drift_values, "threshold": threshold},
        "entity_ecosystem": entity_ecosystem,
        "framing_drift": framing_drift,
        "temporal_narrative_regimes": {},
    })

    summary_lines = [
        f"**Source:** {source}  |  **Sample articles analyzed:** {total_docs}  |  **Periods:** {len(periods)}",
        f"**Editorial behavior:** {behavior}",
        f"**Confidence:** {confidence['confidence_label']} ({confidence['confidence_score']:.2f}) "
        f"-- semantic={confidence['semantic_strength']:.2f}, framing={confidence['framing_strength']:.2f}, "
        f"actor={confidence['actor_strength']:.2f}",
        "",
        "**Drift per transition:**",
    ]

    for label, value, cls in zip(drift_labels, drift_values, drift_classes):
        summary_lines.append(f"- {label}: {value:.4f} ({cls})")

    graph_text = "LLM-based stages weren't run for this analysis."

    if run_llm_stages:
        progress(0.75, desc="Running LLM-based stages (latent frames + actor-frame graph)...")

        try:
            graph_text = _run_llm_stages(
                source, grouped, framing_drift, entity_importance, salience_totals, summary_lines
            )
        except Exception as e:
            graph_text = f"LLM-based stages failed: {e}"

    progress(1.0, desc="Done.")

    return semantic_fig, sentiment_fig, entity_fig, "\n".join(summary_lines), graph_text


with gr.Blocks(title="Narrative Drift Detection") as demo:
    gr.Markdown(
        "# Narrative Drift Detection\n"
        "Master's dissertation demo (Technical University of Cluj-Napoca). "
        "Runs the real analysis pipeline live, over a small bundled sample "
        "of Ukraine-war coverage (a few hundred truncated article excerpts "
        "per source) rather than the full ~40k-article research database.\n\n"
        "See the [project README](https://github.com/Prodan-Remus-30123/narrative-drift-detection) "
        "for the full pipeline, which also covers latent frame discovery, "
        "narrative signatures/archetypes, cross-source divergence and more."
    )

    with gr.Row():
        source_dropdown = gr.Dropdown(SOURCES, value=SOURCES[0], label="News source")
        llm_checkbox = gr.Checkbox(
            value=False,
            label="Run LLM-based stages (latent frame labeling + actor-frame graph)",
            info="Slower -- calls the Hugging Face Inference API once per frame/actor."
        )
        run_button = gr.Button("Run analysis", variant="primary")

    summary_output = gr.Markdown()

    with gr.Row():
        semantic_plot = gr.Plot(label="Semantic drift")
        sentiment_plot = gr.Plot(label="Sentiment evolution")

    entity_plot = gr.Plot(label="Top entities by importance")
    graph_output = gr.Markdown(label="Actor-frame graph")

    run_button.click(
        fn=run_analysis,
        inputs=[source_dropdown, llm_checkbox],
        outputs=[semantic_plot, sentiment_plot, entity_plot, summary_output, graph_output],
    )


if __name__ == "__main__":
    # Gradio 6 defaults to server-side rendering via a Node.js proxy
    # on Spaces, which was crashing this Space right after startup
    # (RUNTIME_ERROR, log just shows "Stopping Node.js server...").
    # Disabling it falls back to plain client-side rendering.
    demo.launch(ssr_mode=False)
