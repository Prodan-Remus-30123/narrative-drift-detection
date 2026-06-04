from agentic_explainers.semantic_agent import explain_semantics
from agentic_explainers.framing_agent import explain_framing


def explain_packet(packet):
    explanations = []

    explanations.append(
        explain_semantics(packet)
    )

    explanations.append(
        explain_framing(packet)
    )

    return {
        "source": packet["source"],
        "transition": packet["transition"],
        "explanations": explanations
    }


def explain_packets(packets, max_packets=None):
    selected_packets = packets

    if max_packets is not None:
        selected_packets = packets[:max_packets]

    return [
        explain_packet(packet)
        for packet in selected_packets
    ]


def print_explanations(results):
    print("\n=== AGENTIC NARRATIVE EXPLANATIONS ===")

    for result in results:
        print("\n" + "=" * 60)
        print(
            f"{result['source']} | "
            f"{result['transition']}"
        )
        print("=" * 60)

        for explanation in result["explanations"]:
            print()
            print(explanation)