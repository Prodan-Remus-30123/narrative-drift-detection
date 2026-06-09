import json

from agentic_llm.agentic_runner import (
    AgenticNarrativeRunner
)


def load_packets(
    path="analysis_results.json"
):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def find_packet(
    analysis_results,
    source,
    transition
):
    packets = analysis_results["__evidence_packets__"]

    for packet in packets:

        if (
            packet["source"] == source
            and
            packet["transition"] == transition
        ):
            return packet

    return None


def main():

    analysis_results = load_packets()

    runner = AgenticNarrativeRunner(
        model="qwen2.5:7b"
    )

    source = "bbc.co.uk"

    transition = "2020-07_08->2020-09_10"

    packet = find_packet(
        analysis_results,
        source,
        transition
    )

    if packet is None:
        print("Packet not found")
        return

    while True:

        question = input(
            "\nQuestion (exit=quit): "
        )

        if question.lower() == "exit":
            break

        result = runner.answer(
            question=question,
            packet=packet
        )

        print("\n====================")
        print("SELECTED AGENTS")
        print("====================")

        print(
            result["selected_agents"]
        )

        print("\n====================")
        print("VALIDATOR WARNINGS")
        print("====================")

        for warning in result["warnings"]:
            print("-", warning)

        print("\n====================")
        print("SPECIALIST OUTPUTS")
        print("====================")

        for name, output in (
            result["specialist_outputs"].items()
        ):
            print(f"\n[{name.upper()}]\n")
            print(output)

        print("\n====================")
        print("FINAL ANSWER")
        print("====================\n")

        print(
            result["final_answer"]
        )


if __name__ == "__main__":
    main()