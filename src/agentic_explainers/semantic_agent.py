"""
Semantic explanation agent.

Explains WHAT changed semantically in one narrative transition.
Consumes evidence packets only. Does not recompute analytics.
"""


def _format_frame(frame):
    if isinstance(frame, dict):
        label = frame.get("frame_label", "unknown frame")
        delta = frame.get("delta_share")

        if delta is None:
            return f"- {label}"

        return f"- {label} (delta={delta:.4f})"

    return f"- {frame}"


class SemanticAgent:
    """
    Evidence-grounded semantic explanation agent.
    """

    name = "SemanticAgent"

    def explain(self, packet):
        what_how = packet.get("what_how", {})

        source = packet.get("source", "unknown source")
        transition = packet.get("transition", "unknown transition")

        semantic_drift = what_how.get("semantic_drift", 0.0)
        threshold = what_how.get("semantic_threshold", 0.0)
        classification = what_how.get(
            "semantic_classification",
            "unknown"
        )

        rising_frames = what_how.get(
            "top_rising_frames",
            []
        )

        falling_frames = what_how.get(
            "top_falling_frames",
            []
        )

        lines = []

        lines.append(
            f"SemanticAgent analysis for {source} | {transition}"
        )

        lines.append("")
        lines.append(
            "Semantic movement:"
        )

        lines.append(
            f"- Drift value: {semantic_drift:.4f}"
        )

        lines.append(
            f"- Dynamic threshold: {threshold:.4f}"
        )

        lines.append(
            f"- Classification: {classification}"
        )

        lines.append("")

        if classification == "significant":
            lines.append(
                "Interpretation: the transition shows a substantial semantic shift relative to this source's own temporal baseline."
            )
        else:
            lines.append(
                "Interpretation: the transition shows semantic movement, but it does not exceed this source's dynamic significance threshold."
            )

        lines.append("")

        lines.append("Emerging semantic frames:")
        if rising_frames:
            for frame in rising_frames:
                lines.append(_format_frame(frame))
        else:
            lines.append("- No rising frame evidence available.")

        lines.append("")

        lines.append("Declining semantic frames:")
        if falling_frames:
            for frame in falling_frames:
                lines.append(_format_frame(frame))
        else:
            lines.append("- No falling frame evidence available.")

        lines.append("")

        lines.append(
            "Evidence boundary: this explanation describes semantic movement only. It does not explain actor-level reframing, sentiment, or causal reasons."
        )

        return "\n".join(lines)
    
def explain_semantics(packet):
    agent = SemanticAgent()
    return agent.explain(packet)