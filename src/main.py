import pandas as pd
import numpy as np
from datetime import datetime

from preprocessing import preprocess_corpus
from embeddings import EmbeddingModel
from drift import (compute_cosine_drift, compute_dynamic_threshold, classify_drift)
from plots.plot_semantic_comparison_across_sources import plot_multiple_sources
from entities import analyze_entities
from interpreter import interpret_shift
from database import load_full_articles

from entity_framing_drift import compute_entity_drift, compute_entity_importance
from plots.plot_entity_drift import plot_top_entity_drift
from plots.plot_entity_heatmap import plot_entity_heatmap

from temporal_entity_analysis import group_articles_by_period
from plots.plot_semantic_vs_framing import plot_semantic_vs_framing
from plots.plot_source_summary import build_source_summary
from plots.plot_actor_evolution import plot_actor_evolution

from sentiment_analysis import analyze_sentiment
from plots.plot_sentiment_evolution import plot_sentiment_evolution

from correlation_analysis import (compute_sentiment_deltas, compute_average_framing, compute_correlation)
from editorial_behavior import classify_editorial_behavior
from emotional_volatility import compute_emotional_volatility
from plots.plot_source_dashboard import plot_source_dashboard
from utils.period_sorting import sort_period_key

from actor_salience import compute_actor_salience, compute_total_actor_salience
from plots.plot_actor_salience import plot_actor_salience
from embedding_model_registry import get_embedding_model
from latent_frames import compute_frame_evolution_from_grouped
from temporal_frame_evolution import build_frame_trajectories
from dynamic_entity_ecosystem import (
    build_dynamic_entity_ecosystem,
    summarize_dynamic_entity_ecosystem,
    print_top_dynamic_entities
)

from entity_frame_alignment import (
    build_entity_frame_alignment,
    summarize_entity_frame_migrations,
    print_top_entity_frame_migrations
)
from semantic_frame_labeling import (
    label_all_latent_frames,
    print_labeled_frames
)

from narrative_signatures import (
    build_all_narrative_signatures,
    print_narrative_signature
)

from signature_comparison import (
    compute_signature_similarity,
    cluster_signatures,
    compute_signature_pca,
    print_signature_similarity,
    print_signature_clusters,
    print_signature_pca
)

from affective_dynamics import (
    compute_affective_dynamics,
    print_affective_dynamics
)

from narrative_change_profile import (
    build_narrative_change_profile,
    print_narrative_change_profile
)

from narrative_archetypes import (
    build_transition_archetype_table,
    discover_narrative_archetypes,
    print_archetype_summaries
)

from evidence_packet_builder import (
    build_all_evidence_packets,
    print_evidence_packet_summary
)

from agentic_explainers.explanation_runner import (
    explain_packets,
    print_explanations
)

from change_point_detection import (
    detect_source_change_points,
    detect_embedding_change_points,
    detect_monthly_semantic_change_points,
    print_change_point_summary
)

# ==========================================
# DEBUG / EXECUTION MODES
# ==========================================

FAST_MODE = False

# DEBUG_SOURCES = None
DEBUG_SOURCES = {"bbc.co.uk"}
# DEBUG_SOURCES = {"cnn.com"}
# DEBUG_SOURCES = {"bbc.co.uk", "cnn.com"}

SKIP_PLOTS = True
SKIP_FRAME_LABELING = True
SKIP_SIGNATURE_COMPARISON = True
SAVE_ANALYSIS_RESULTS = False
VERBOSE_ENTITY_DEBUG = False
SKIP_ENTITY_FRAME_ALIGNMENT = True
PRINT_SENTIMENT_PERIODS = False
SKIP_CHANGE_POINT_DETECTION = False
SKIP_EVIDENCE_PACKETS = True
SKIP_AGENTIC_EXPLANATIONS = True
SKIP_NARRATIVE_SIGNATURES = False
SKIP_ARCHETYPES = False
SKIP_CROSS_LAYER_CORRELATION = False
SKIP_CHANGE_PROFILE_PRINT = True

# def group_by_source_and_month(df):
#     df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
#     df["date"] = df["date"].dt.tz_localize(None)
#     df["month"] = df["date"].dt.to_period("M")

#     grouped = {}

#     for (source, month), rows in df.groupby(["source", "month"]):
#         if source not in grouped:
#             grouped[source] = {}

#         grouped[source][month] = rows["text"].tolist()

#     return grouped

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(item)
            for item in obj
        ]

    if isinstance(obj, tuple):
        return [
            make_json_serializable(item)
            for item in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj

def main():
    #  Load CSV
    df = load_full_articles()

    #  Group by month
    grouped = {}

    for source in df["source"].unique():

        source_df = df[df["source"] == source]

        grouped[source] = (group_articles_by_period(source_df))

    monthly_grouped = {}

    df_monthly = df.copy()
    df_monthly["date"] = pd.to_datetime(
        df_monthly["date"],
        format="mixed",
        utc=True
    )
    df_monthly["date"] = df_monthly["date"].dt.tz_localize(None)
    df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)

    for source in df_monthly["source"].unique():
        source_df = df_monthly[df_monthly["source"] == source]

        monthly_grouped[source] = {}

        for month, rows in source_df.groupby("month"):
            texts = rows["text"].dropna().tolist()

            if len(texts) >= 20:
                monthly_grouped[source][month] = texts

    print("\n=== GROUPED DATA ===")

    source_summaries = []

    for source in grouped:
        if (DEBUG_SOURCES is not None and source not in DEBUG_SOURCES):
            continue

        total_docs = sum( len(grouped[source][period]) for period in grouped[source])

        if total_docs < 50:
            print(f"\nSkipping {source}: insufficient documents ({total_docs})")
            continue

        print(f"\nSource: {source}")

        for month in sorted(grouped[source].keys(),key=sort_period_key):
            print(month, len(grouped[source][month]))

    #  Preprocess
    for source in grouped:
        if (DEBUG_SOURCES is not None and source not in DEBUG_SOURCES):
            continue

        for month in sorted(grouped[source].keys(), key=sort_period_key):
            grouped[source][month] = preprocess_corpus(grouped[source][month])


            # for point in trajectory["values"]:
            #     print(
            #         f"  {point['period']}: "
            #         f"share={point['share']:.3f}, "
            #         f"count={point['count']:.2f}"
            #     )
    
    for source in monthly_grouped:
        if DEBUG_SOURCES is not None and source not in DEBUG_SOURCES:
            continue

        for month in sorted(monthly_grouped[source].keys()):
            monthly_grouped[source][month] = preprocess_corpus(
                monthly_grouped[source][month]
            )
        
    #  Embeddings
    model = get_embedding_model()

    source_results = {}
    analysis_results = {}
    latent_results_by_source = {}
    print("\n=== Source-Aware Narrative Drift ===")

    for source in grouped:
        if (DEBUG_SOURCES is not None and source not in DEBUG_SOURCES):
            continue
        total_docs = sum(len(grouped[source][period]) for period in grouped[source])
        if total_docs < 50:
            print(f"\nSkipping {source}: insufficient documents ({total_docs})")
            continue

        aggregated_vectors = []

        for month in sorted(grouped[source].keys(), key=sort_period_key):
            embeddings = model.encode_documents(grouped[source][month])

            vec = model.aggregate_embeddings(embeddings)

            aggregated_vectors.append((month, vec))


        drift_values = []
        drift_labels = []

        print(f"\nSource: {source}")
        analysis_results[source] = {}

        monthly_embedding_labels = []
        monthly_embedding_vectors = []

        if source in monthly_grouped:
            for month in sorted(monthly_grouped[source].keys()):
                monthly_embeddings = model.encode_documents(
                    monthly_grouped[source][month]
                )

                monthly_vec = model.aggregate_embeddings(
                    monthly_embeddings
                )

                monthly_embedding_labels.append(month)
                monthly_embedding_vectors.append(monthly_vec)

        analysis_results[source]["monthly_semantic_embedding_trajectory"] = {
            "labels": monthly_embedding_labels,
            "vectors": monthly_embedding_vectors
        }

        embedding_labels = [
            str(month)
            for month, vec in aggregated_vectors
        ]

        embedding_vectors = [
            vec
            for month, vec in aggregated_vectors
        ]

        analysis_results[source]["semantic_embedding_trajectory"] = {
            "labels": embedding_labels,
            "vectors": embedding_vectors
        }

        print("\n=== Temporal Frame Evolution ===")

        latent_result = compute_frame_evolution_from_grouped(source, grouped[source])
        latent_results_by_source[source] = latent_result
        trajectories = build_frame_trajectories(latent_result)

        analysis_results[source]["latent_frames"] = trajectories

        ranked_trajectories = sorted(trajectories.items(), key=lambda x: x[1]["volatility"], reverse=True)

        for frame_id, trajectory in ranked_trajectories[:5]:
            print(f"\nFrame {frame_id}: {trajectory['label']}")
            print(f"Peak period: " f"{trajectory['peak_period']}")
            print(f"Peak share: " f"{trajectory['peak_share']:.3f}")
            print(f"Volatility: " f"{trajectory['volatility']:.3f}")
            print(f"Net change: " f"{trajectory['net_change']:.3f}")

        print("\n=== Semantic Frame Labeling ===")

        top_frames = 3 if FAST_MODE else 10
        important_frame_ids = [
            frame_id
            for frame_id, trajectory
            in ranked_trajectories[:top_frames]
        ]

        filtered_clusters = {
            frame_id: latent_result["clusters"][frame_id]
            for frame_id in important_frame_ids
        }

        filtered_latent_result = {
            **latent_result,
            "clusters": filtered_clusters
        }

        if SKIP_FRAME_LABELING:
            labeled_frames = {}
        else:
            labeled_frames = label_all_latent_frames(filtered_latent_result)

        if not SKIP_FRAME_LABELING:
            print_labeled_frames(
                labeled_frames,
                top_n=3
            )

        analysis_results[source][
            "semantic_frames"
        ] = labeled_frames

        print("\n=== Semantic Drift Evolution ===")

        for i in range(len(aggregated_vectors) - 1):

            m1, v1 = aggregated_vectors[i]
            m2, v2 = aggregated_vectors[i + 1]

            drift = compute_cosine_drift(v1, v2)
            
            drift_values.append(drift)
            drift_labels.append(f"{m1}->{m2}")

            print(f"{m1} → {m2} | Drift: {drift:.4f}")
            # drift_interpretation = analyze_drift(
            # source=source,
            # period=f"{m1}->{m2}",
            # drift_value=drift
            # )

            # print("\n[Drift Agent]")
            # print(drift_interpretation)
        
        dynamic_threshold = compute_dynamic_threshold(drift_values,method="median_mad")

        analysis_results[source]["semantic_drift"] = {
            "labels": drift_labels,
            "values": drift_values,
            "threshold": dynamic_threshold
        }

        print(f"\nDynamic semantic threshold: " f"{dynamic_threshold:.4f}" if dynamic_threshold is not None else "\nDynamic semantic threshold: insufficient data")

        for label, drift in zip(drift_labels, drift_values):
            classification = classify_drift(drift, dynamic_threshold)

            print(f"{label} | Drift: {drift:.4f} | " f"Classification: {classification}")

        source_results[source] = {
            "labels": drift_labels,
            "values": drift_values
        }

        analysis_results[source][
            "semantic_drift"
        ] = {
            "labels": drift_labels,
            "values": drift_values,
            "threshold": dynamic_threshold
        }

        print("\n=== Sentiment Evolution ===")

        sentiment_results = {}

        for period in sorted(grouped[source].keys(),key=sort_period_key):

            sentiment = analyze_sentiment(grouped[source][period])
            sentiment_results[str(period)] = sentiment
            if PRINT_SENTIMENT_PERIODS:
                print(f"\n{period}")
                print(f"Compound: " f"{sentiment['compound']:.4f}")
                print(f"Positive: " f"{sentiment['positive']:.4f}")
                print(f"Negative: " f"{sentiment['negative']:.4f}")

                print(f"Neutral: " f"{sentiment['neutral']:.4f}")
        
        if not SKIP_PLOTS:
            plot_sentiment_evolution(sentiment_results, source)
        analysis_results[source]["sentiment"] = sentiment_results

        affective_result = compute_affective_dynamics(
            sentiment_results
        )

        analysis_results[source]["affective_dynamics"] = affective_result

        print_affective_dynamics(
            affective_result
        )

        print("\n=== Entity Framing Drift ===")

        framing_drift = compute_entity_drift(grouped[source])
        analysis_results[source]["framing_drift"] = framing_drift
        salience_results = compute_actor_salience(grouped[source])

        if not SKIP_PLOTS:
            plot_actor_salience(salience_results, top_n=5)

        salience_totals = compute_total_actor_salience(salience_results)
        entity_importance = compute_entity_importance(framing_drift, salience_totals)

        print("\n=== Dynamic Entity Ecosystem ===")

        entity_ecosystem = build_dynamic_entity_ecosystem(
            framing_drift=framing_drift,
            entity_importance=entity_importance
        )

        analysis_results[source]["entity_ecosystem"] = entity_ecosystem

        # ==========================================
        # CHANGE POINT DETECTION
        # ==========================================

        if not SKIP_CHANGE_POINT_DETECTION:
            print("\n=== RUNNING CHANGE POINT DETECTION ===")

            change_points = detect_source_change_points(
                analysis_results[source]
            )

            embedding_change_points = detect_embedding_change_points(
                vectors=embedding_vectors,
                labels=embedding_labels,
                model="rbf",
                penalty=3.0
            )

            change_points["semantic_embedding_trajectory"] = embedding_change_points
            monthly_embedding_change_points = detect_monthly_semantic_change_points(
                monthly_vectors=monthly_embedding_vectors,
                labels=monthly_embedding_labels,
                n_components=3,
                model="rbf",
                penalty=0.75
            )

            change_points["monthly_semantic_embedding_trajectory"] = (
                monthly_embedding_change_points
            )
            analysis_results[source]["change_points"] = change_points

            print_change_point_summary(
                source,
                change_points
            )

        else:
            analysis_results[source]["change_points"] = {}


        if VERBOSE_ENTITY_DEBUG:
            print("\nENTITY DRIFT DEBUG")

            all_drifts = [
                stats["mean_drift"]
                for stats in entity_ecosystem.values()
            ]

            print(
                "mean ecosystem drift:",
                np.mean(all_drifts)
            )

        ecosystem_summary = summarize_dynamic_entity_ecosystem(
            entity_ecosystem
        )

        print(ecosystem_summary)

        print_top_dynamic_entities(
            entity_ecosystem,
            top_n=3,
            sort_by="importance"
        )
        if SKIP_ENTITY_FRAME_ALIGNMENT:
            print("\n=== Entity → Latent Frame Alignment ===")
            print("Skipped entity-frame alignment.")
            analysis_results[source]["frame_migrations"] = {}

        else:
            print("\n=== Entity → Latent Frame Alignment ===")

            latent_result = latent_results_by_source.get(source)

            if latent_result is not None:
                entity_frame_alignment = build_entity_frame_alignment(
                    framing_drift=framing_drift,
                    latent_result=latent_result,
                    semantic_frames=analysis_results[source]["semantic_frames"],
                    top_k=3
                )

                migration_summary = summarize_entity_frame_migrations(
                    entity_frame_alignment
                )

                print_top_entity_frame_migrations(
                    migration_summary,
                    top_n=3
                )

                analysis_results[source]["frame_migrations"] = migration_summary

            else:
                print(
                    f"No latent frame result available for {source}; "
                    "skipping entity-frame alignment."
                )

                analysis_results[source]["frame_migrations"] = {}

        change_profile = build_narrative_change_profile(
            analysis_results[source],
            top_n=5
        )

        analysis_results[source]["change_profile"] = change_profile

        if not SKIP_CHANGE_PROFILE_PRINT:
            print_narrative_change_profile(
                change_profile,
                top_n=3
            )

        summary = build_source_summary(
            source= source,
            semantic_values= drift_values,
            framing_drift= framing_drift,
            entity_importance= entity_importance,
            num_periods= len(grouped[source]),
            num_articles= sum(len(grouped[source][period]) for period in grouped[source]),
            avg_sentiment= sum(sentiment_results[p]["compound"] for p in sentiment_results) / len(sentiment_results)
        )

        source_summaries.append(summary)

        behavior = classify_editorial_behavior(
            semantic=summary["avg_semantic_drift"],
            framing=summary["avg_framing_turnover"],
            sentiment=summary["avg_sentiment"]
        )

        print(f"\nEditorial behavior: "f"{behavior}")

        volatility = compute_emotional_volatility(sentiment_results)
        print(f"\nEmotional volatility: "f"{volatility:.4f}")


        average_framing_values = []

        for transition in sorted(framing_drift.keys(),key=sort_period_key):

            entities = framing_drift[transition]
            if len(entities) == 0:
                average_framing_values.append(np.nan)
                continue

            turnover_values = [stats.get("vocabulary_turnover") for stats in entities.values() if stats.get("vocabulary_turnover") is not None]
            avg = sum(turnover_values) / len(turnover_values) if turnover_values else np.nan

            average_framing_values.append(avg)
        if not SKIP_PLOTS:
            plot_semantic_vs_framing(
                semantic_labels=drift_labels,
                semantic_values=drift_values,
                framing_values=average_framing_values,
                source=source
            )

        if not SKIP_PLOTS:
            plot_top_entity_drift(framing_drift)
        # plot_entity_heatmap(framing_drift)
        # plot_actor_evolution(framing_drift)

        print("\n=== Cross-Layer Correlation ===")

        sentiment_deltas = compute_sentiment_deltas(sentiment_results)
        avg_framing = compute_average_framing(framing_drift)
        correlation = compute_correlation(sentiment_deltas, avg_framing)

        print(f"Sentiment ↔ Framing correlation: " f"{correlation}")

        # for transition, entities in framing_drift.items():
        #     print(f"\n{transition}")
        #     ranked_entities = sorted(entities.items(), key=lambda x: x[1]["drift"], reverse=True)

        #     for entity, stats in ranked_entities[:5]:
        #         print(f"\nEntity: {entity}")
        #         print(f"Drift: " f"{stats['drift']:.3f}")
        #         print(f"Before: " f"{list(stats['before'].keys())[:5]}")
        #         print(f"After: " f"{list(stats['after'].keys())[:5]}")
        
        # plot_source_dashboard(
        #     source=source,
        #     semantic_labels=drift_labels,
        #     semantic_values=drift_values,
        #     sentiment_results=sentiment_results,
        #     framing_drift=framing_drift
        # )

        # Entity analysis per month
        months_sorted = sorted(grouped[source].keys(), key=sort_period_key)

        # for month in months_sorted:

        #     print(f"\n{source} | {month}")

        #     entity_stats = analyze_entities(
        #         grouped[source][month]
        #     )

        #     for entity, stats in entity_stats.items():

        #         print(f"\nEntity: {entity}")

        #         print(
        #             f"Subject count: {stats['subject_count']}"
        #         )

        #         print(
        #             f"Object count: {stats['object_count']}"
        #         )

        #         print(
        #             f"Top verbs: {stats['verbs'].most_common(5)}"
        #         )

        #  Narrative interpretation per transition

        # for i in range(len(months_sorted) - 1):

        #     current_month = months_sorted[i + 1]

        #     entity_stats = analyze_entities(
        #         grouped[source][current_month]
        #     )

        #     interpretation = interpret_shift(
        #         source=source,
        #         period=f"{months_sorted[i]}->{current_month}",
        #         drift=drift_values[i],
        #         entity_stats=entity_stats
        #     )

        #     print("\nNarrative Interpretation:")
        #     print(
        #         f"{months_sorted[i]}->{current_month}"
        #     )

        #     print(interpretation)

        # # Change point
        # change_points = detect_changepoints(drift_values)
        # print(f"Detected change points: {change_points}")


    #  Plot drift signal
    if not SKIP_PLOTS:
        plot_multiple_sources(source_results)

    if not SKIP_NARRATIVE_SIGNATURES:
        print("\n=== NARRATIVE SIGNATURES ===")

        narrative_signatures = build_all_narrative_signatures(
            analysis_results
        )

        for signature in narrative_signatures:
            print_narrative_signature(signature)

        signature_df = pd.DataFrame(narrative_signatures)

        print("\n=== NARRATIVE SIGNATURE TABLE ===")
        print(signature_df)

    else:
        narrative_signatures = []
        signature_df = pd.DataFrame()

    if not SKIP_EVIDENCE_PACKETS:
        print("\n=== EVIDENCE PACKETS ===")

        evidence_packets = build_all_evidence_packets(
            analysis_results,
            top_n=3
        )

        print_evidence_packet_summary(
            evidence_packets,
            max_packets=5
        )

    else:
        evidence_packets = []

    if not SKIP_AGENTIC_EXPLANATIONS:
        print("\n=== AGENTIC NARRATIVE EXPLANATIONS ===")

        agentic_explanations = explain_packets(
            evidence_packets,
            max_packets=5
        )

        print_explanations(
            agentic_explanations
        )

    else:
        agentic_explanations = []


    if not SKIP_ARCHETYPES:
        print("\n=== NARRATIVE ARCHETYPES ===")

        archetype_table = build_transition_archetype_table(
            analysis_results
        )

        archetype_table, archetype_summaries, silhouette_scores = (
            discover_narrative_archetypes(
                archetype_table,
                n_clusters=None
            )
        )

        print("\nSilhouette scores by k:")
        print(silhouette_scores)

        print_archetype_summaries(
            archetype_summaries
        )

        archetype_results = {
            "table": archetype_table.to_dict("records"),
            "summaries": archetype_summaries,
            "silhouette_scores": silhouette_scores
        }

    else:
        archetype_results = {}

    if not SKIP_SIGNATURE_COMPARISON:
        similarity_matrix, sources = compute_signature_similarity(
            narrative_signatures
        )

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=sources,
            columns=sources
        )

        print_signature_similarity(similarity_df)
        cluster_df = cluster_signatures(narrative_signatures,n_clusters=2)
        print_signature_clusters(cluster_df)
        pca_df = compute_signature_pca(narrative_signatures)
        print_signature_pca(pca_df)

    summary_df = pd.DataFrame(source_summaries)

    print("\n=== SOURCE SUMMARY ===")

    print(summary_df)
    analysis_results["__evidence_packets__"] = (evidence_packets)
    analysis_results["__narrative_signatures__"] = (narrative_signatures)
    analysis_results["__source_summary__"] = (summary_df.to_dict("records"))
    analysis_results["__archetypes__"] = archetype_results

    if SAVE_ANALYSIS_RESULTS:
        import json

        with open(
            "analysis_results.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                make_json_serializable(analysis_results),
                f,
                indent=2,
                ensure_ascii=False
            )

        print("\nSaved analysis_results.json")



def debug_signature_comparison():
    import json

    with open(
        "analysis_results.json",
        "r",
        encoding="utf-8"
    ) as f:
        analysis_results = json.load(f)

    narrative_signatures = build_all_narrative_signatures(
        analysis_results
    )


    signatures = build_all_narrative_signatures(
        analysis_results
    )

    for s in signatures:
        print("\n")
        print(s["source"])

        for k, v in s.items():

            if isinstance(v, (int, float)):
                print(f"{k}: {v}")

    for weight in [0.0, 1.0, 4.0, 10.0, 20.0]:
        print(
            f"\n=== SIGNATURE COMPARISON | semantic_weight={weight} ==="
        )

        similarity_matrix, sources = compute_signature_similarity(
            narrative_signatures,
            semantic_weight=weight
        )

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=sources,
            columns=sources
        )

        print_signature_similarity(similarity_df)

        cluster_df = cluster_signatures(
            narrative_signatures,
            n_clusters=2,
            include_semantic=(weight > 0),
            semantic_weight=weight
        )

        print_signature_clusters(cluster_df)


if __name__ == "__main__":
    main()
    #debug_signature_comparison()