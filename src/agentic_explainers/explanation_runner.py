"""
Runs explanation agents over evidence packets.
"""

from agentic_explainers.semantic_agent import SemanticAgent


def explain_packet(packet):
    semantic_agent = SemanticAgent()

    return {
        "source": packet.get("source"),
        "transition": packet.get("transition"),
        "semantic_explanation": semantic_agent.explain(packet)
    }


def explain_packets(packets, max_packets=None):
    explanations = []

    selected_packets = packets

    if max_packets is not None:
        selected_packets = packets[:max_packets]

    for packet in selected_packets:
        explanations.append(
            explain_packet(packet)
        )

    return explanations


def print_explanations(explanations):
    print("\n=== AGENTIC NARRATIVE EXPLANATIONS ===")

    for item in explanations:
        print("\n" + "=" * 60)
        print(
            f"{item['source']} | {item['transition']}"
        )
        print("=" * 60)
        print(item["semantic_explanation"])