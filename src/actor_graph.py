"""
actor_graph.py

Narrative actor graph centrality.
"""

from collections import defaultdict
import networkx as nx

from database import load_full_articles
from preprocessing import preprocess_corpus
from temporal_entity_analysis import (
    group_articles_by_period
)
from entities import analyze_entities


MIN_VERB_COUNT = 3


def build_actor_verb_graph(grouped_texts):

    graph = nx.Graph()

    for period, texts in grouped_texts.items():

        entity_data = analyze_entities(
            texts
        )

        for entity, stats in entity_data.items():

            entity_node = f"ENTITY::{entity}"

            graph.add_node(
                entity_node,
                node_type="entity"
            )

            for verb, count in stats["verbs"].items():

                if count < MIN_VERB_COUNT:
                    continue

                verb_node = f"VERB::{verb}"

                graph.add_node(
                    verb_node,
                    node_type="verb"
                )

                if graph.has_edge(
                    entity_node,
                    verb_node
                ):

                    graph[entity_node][verb_node][
                        "weight"
                    ] += count

                else:

                    graph.add_edge(
                        entity_node,
                        verb_node,
                        weight=count
                    )

    return graph


def compute_actor_centrality(graph):

    centrality = nx.pagerank(
        graph,
        weight="weight"
    )

    actor_scores = {}

    for node, score in centrality.items():

        if node.startswith("ENTITY::"):

            actor = node.replace(
                "ENTITY::",
                ""
            )

            actor_scores[actor] = score

    return actor_scores


def compute_actor_graph_centrality(
    source
):

    df = load_full_articles()

    source_df = df[
        df["source"] == source
    ]

    grouped = group_articles_by_period(
        source_df
    )

    for period in grouped:

        grouped[period] = preprocess_corpus(
            grouped[period]
        )

    graph = build_actor_verb_graph(
        grouped
    )

    centrality = compute_actor_centrality(
        graph
    )

    ranked = sorted(

        centrality.items(),

        key=lambda x: x[1],

        reverse=True
    )

    return {

        "source":
            source,

        "graph_nodes":
            graph.number_of_nodes(),

        "graph_edges":
            graph.number_of_edges(),

        "actor_centrality":
            ranked
    }


def main():

    result = compute_actor_graph_centrality(
        "cnn.com"
    )

    print(
        "\n=== ACTOR GRAPH CENTRALITY ==="
    )

    print(
        "Nodes:",
        result["graph_nodes"]
    )

    print(
        "Edges:",
        result["graph_edges"]
    )

    print(
        "\n=== TOP ACTORS ==="
    )

    for actor, score in result[
        "actor_centrality"
    ][:30]:

        print(
            f"{actor}: {score:.6f}"
        )


if __name__ == "__main__":
    main()