from agentic_explainers.semantic_explainer import explain_semantics
from agentic_explainers.framing_explainer import explain_framing
from agentic_explainers.synthesis_explainer import explain_synthesis
from agentic_explainers.affective_explainer import explain_affective

def explain_packet(packet):
    explanations = []

    explanations.append(
        explain_semantics(packet)
    )

    explanations.append(
        explain_framing(packet)
    )

    explanations.append(
        explain_affective(packet)
    )

    synthesis_text = explain_synthesis(packet)



    return {
        "source": packet["source"],
        "transition": packet["transition"],
        "explanations": explanations,
        "synthesis": synthesis_text
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

        print()
        print(result["synthesis"])