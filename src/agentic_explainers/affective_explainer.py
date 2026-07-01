class AffectiveExplainer:
    def explain(self, packet):
        affective = packet.get("affective", {})

        compound_delta = affective.get("compound_delta", 0.0)
        intensity_delta = affective.get("intensity_delta", 0.0)
        polarization_delta = affective.get("polarization_delta", 0.0)

        lines = []
        lines.append(
            f"AffectiveExplainer analysis for {packet['source']} | {packet['transition']}"
        )
        lines.append("")
        lines.append("Affective narrative dynamics:")

        lines.append(f"- Compound sentiment delta: {compound_delta:.4f}")
        lines.append(f"- Intensity delta: {intensity_delta:.4f}")
        lines.append(f"- Polarization delta: {polarization_delta:.4f}")

        lines.append("")

        if compound_delta > 0.05:
            lines.append("- The narrative became more positive in tone.")
        elif compound_delta < -0.05:
            lines.append("- The narrative became more negative in tone.")
        else:
            lines.append("- The overall sentiment tone remained relatively stable.")

        if intensity_delta > 0.01:
            lines.append("- Emotional intensity increased.")
        elif intensity_delta < -0.01:
            lines.append("- Emotional intensity decreased.")
        else:
            lines.append("- Emotional intensity remained relatively stable.")

        if polarization_delta > 0.01:
            lines.append("- Emotional polarization increased.")
        elif polarization_delta < -0.01:
            lines.append("- Emotional polarization decreased.")
        else:
            lines.append("- Emotional polarization remained relatively stable.")

        lines.append("")
        lines.append(
            "Evidence boundary: this agent explains affective change from sentiment-derived signals only."
        )

        return "\n".join(lines)


def explain_affective(packet):
    agent = AffectiveExplainer()
    return agent.explain(packet)