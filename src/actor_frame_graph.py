"""
actor_frame_graph.py

Builds Actor ↔ Latent Frame graph using
entity-conditioned latent frame transitions.
"""

import networkx as nx

from actor_graph import compute_actor_graph_centrality
from entity_latent_frames import compute_entity_latent_frame_transitions
from frame_normalization import normalize_frame_labels

TOP_ACTORS = 10


def build_actor_frame_graph(source, top_n=TOP_ACTORS):
    actor_data = compute_actor_graph_centrality(source)

    top_actors = [actor for actor, score in actor_data["actor_centrality"][:top_n]]
    all_frame_labels = []
    graph = nx.Graph()

    actor_frame_results = {}
    pending_edges = []

    for actor in top_actors:
        try:
            frame_data = compute_entity_latent_frame_transitions(source=source, entity=actor)
        except Exception as error:
            print(f"Skipping {actor}: {error}")
            continue

        actor_frame_results[actor] = frame_data

        actor_node = f"ACTOR::{actor}"

        graph.add_node(actor_node, node_type="actor", label=actor)
        

        for transition in frame_data["latent_frame_transitions"]:

            frame_labels = []

            frame_labels.extend(transition["before_frame"].split(" / "))
            frame_labels.extend(transition["after_frame"].split(" / "))
            all_frame_labels.extend(frame_labels)

            for frame_label in frame_labels:
                frame_label = frame_label.strip()

                if not frame_label:
                    continue

                pending_edges.append((actor, frame_label))
                
    normalization_map = normalize_frame_labels(all_frame_labels)

    for actor, frame_label in pending_edges:

        normalized_label = normalization_map.get(frame_label, frame_label)
        actor_node = f"ACTOR::{actor}"
        frame_node = f"FRAME::{normalized_label}"

        graph.add_node(
            actor_node,
            node_type="actor",
            label=actor
        )

        graph.add_node(
            frame_node,
            node_type="frame",
            label=normalized_label
        )

        if graph.has_edge(actor_node, frame_node):
            graph[actor_node][frame_node]["weight"] += 1
        else:
            graph.add_edge(actor_node, frame_node, weight=1)
    
    return {
        "source": source,
        "top_actors": top_actors,
        "graph": graph,
        "actor_frame_results": actor_frame_results
    }


def detect_actor_frame_communities(graph):
    communities = nx.algorithms.community.greedy_modularity_communities(
        graph,
        weight="weight"
    )

    output = []

    for index, community in enumerate(communities):
        actors = []
        frames = []

        for node in community:
            node_type = graph.nodes[node].get("node_type")
            label = graph.nodes[node].get("label")

            if node_type == "actor":
                actors.append(label)

            elif node_type == "frame":
                frames.append(label)

        output.append({
            "community_id": index,
            "actors": sorted(actors),
            "frames": sorted(frames),
            "size": len(community)
        })

    return output


def summarize_actor_frame_graph(graph):
    actor_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("node_type") == "actor"
    ]

    frame_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("node_type") == "frame"
    ]

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "actors": len(actor_nodes),
        "frames": len(frame_nodes)
    }


def main():
    result = build_actor_frame_graph(
        source="cnn.com",
        top_n=5
    )

    graph = result["graph"]

    summary = summarize_actor_frame_graph(graph)

    print("\n=== ACTOR-FRAME GRAPH ===")
    print("Source:", result["source"])
    print("Top actors:", result["top_actors"])
    print("Nodes:", summary["nodes"])
    print("Edges:", summary["edges"])
    print("Actors:", summary["actors"])
    print("Frames:", summary["frames"])

    communities = detect_actor_frame_communities(graph)

    print("\n=== NARRATIVE ECOSYSTEM COMMUNITIES ===")

    for community in communities:
        print(f"\nCommunity {community['community_id']}")
        print("Actors:", community["actors"])
        print("Frames:")

        for frame in community["frames"][:15]:
            print("-", frame)


if __name__ == "__main__":
    main()