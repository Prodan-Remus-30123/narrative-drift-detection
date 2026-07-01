class NarrativeSynthesisExplainer:
    """
    Synthesizes semantic and framing evidence into a human-readable
    narrative explanation.

    It combines evidence from semantic movement, frame change,
    actor reframing, and affective dynamics.
    """

    def explain(self, packet):
        source = packet["source"]
        transition = packet["transition"]

        what_how = packet.get("what_how", {})
        affective = packet.get("affective", {})
        who = packet.get("who", {})

        semantic_drift = what_how.get("semantic_drift", 0.0)
        semantic_classification = what_how.get(
            "semantic_classification",
            "unknown"
        )

        rising_frames = what_how.get("top_rising_frames", [])
        falling_frames = what_how.get("top_falling_frames", [])
        top_entities = who.get("top_entities", [])

        compound_delta = affective.get("compound_delta", 0.0)
        intensity_delta = affective.get("intensity_delta", 0.0)
        polarization_delta = affective.get("polarization_delta", 0.0)

        lines = []

        lines.append(
            f"NarrativeSynthesisExplainer interpretation for "
            f"{source} | {transition}"
        )

        lines.append("")
        lines.append("Overall narrative change:")

        lines.append(
            self._summarize_semantic_change(
                semantic_drift,
                semantic_classification,
                rising_frames,
                falling_frames
            )
        )

        entity_summary = self._summarize_actor_reframing(
            top_entities
        )

        if entity_summary:
            lines.append("")
            lines.append(entity_summary)

        affective_summary = self._summarize_affective_change(
            compound_delta,
            intensity_delta,
            polarization_delta
        )

        if affective_summary:
            lines.append("")
            lines.append(affective_summary)

        lines.append("")
        lines.append(
            "Evidence boundary: this synthesis combines observed "
            "semantic, framing, entity, and affective signals. "
            "It should be read as an evidence-backed interpretation, "
            "not as causal proof."
        )

        return "\n".join(lines)

    def _frame_label(self, frame):
        if isinstance(frame, dict):
            return (
                frame.get("label")
                or frame.get("frame_label")
                or "unknown frame"
            )

        return str(frame)

    def _frame_delta(self, frame):
        if not isinstance(frame, dict):
            return None

        return (
            frame.get("delta")
            or frame.get("delta_share")
            or frame.get("change")
        )

    def _summarize_semantic_change(
        self,
        semantic_drift,
        semantic_classification,
        rising_frames,
        falling_frames
    ):
        lines = []

        if semantic_classification == "significant":
            lines.append(
                f"The transition shows a significant semantic shift "
                f"(drift={semantic_drift:.4f})."
            )
        else:
            lines.append(
                f"The transition shows measurable but non-significant "
                f"semantic movement (drift={semantic_drift:.4f})."
            )

        if rising_frames:
            strongest_rising = self._frame_label(rising_frames[0])
            lines.append(
                f"The strongest emerging frame is: "
                f"{strongest_rising}."
            )

        if falling_frames:
            strongest_falling = self._frame_label(falling_frames[0])
            lines.append(
                f"The strongest declining frame is: "
                f"{strongest_falling}."
            )

        if rising_frames and falling_frames:
            lines.append(
                "This suggests that the narrative emphasis moved "
                f"from {self._frame_label(falling_frames[0])} "
                f"toward {self._frame_label(rising_frames[0])}."
            )

        return "\n".join(lines)

    def _summarize_actor_reframing(self, entities):
        if not entities:
            return None

        lines = []
        lines.append("Actor-level interpretation:")

        for entity in entities[:3]:
            name = entity.get("entity", "unknown entity")
            turnover = entity.get("vocabulary_turnover")
            similarity = entity.get("shared_similarity")
            js = entity.get("framing_drift_js")
            role = entity.get("ecosystem_role", "unknown role")

            added = entity.get("added_verbs", [])
            removed = entity.get("removed_verbs", [])

            turnover_text = (
                f"{turnover:.4f}"
                if turnover is not None
                else "N/A"
            )

            similarity_text = (
                f"{similarity:.4f}"
                if similarity is not None
                else "N/A"
            )

            js_text = (
                f"{js:.4f}"
                if js is not None
                else "N/A"
            )

            lines.append(
                f"- {name} appears as a {role} "
                f"js={js_text}, "
                f"(turnover={turnover_text}, "
                f"shared_similarity={similarity_text})."
            )

            if removed and added:
                lines.append(
                    f"  Its framing moves away from verbs such as "
                    f"{', '.join(removed[:4])} and toward verbs such as "
                    f"{', '.join(added[:4])}."
                )
            elif added:
                lines.append(
                    f"  New associated verbs include "
                    f"{', '.join(added[:4])}."
                )
            elif removed:
                lines.append(
                    f"  Previously associated verbs such as "
                    f"{', '.join(removed[:4])} become less visible."
                )

        return "\n".join(lines)

    def _summarize_affective_change(
        self,
        compound_delta,
        intensity_delta,
        polarization_delta
    ):
        lines = []
        lines.append("Affective interpretation:")

        if compound_delta > 0.05:
            lines.append(
                f"- Sentiment becomes more positive "
                f"(compound_delta={compound_delta:.4f})."
            )
        elif compound_delta < -0.05:
            lines.append(
                f"- Sentiment becomes more negative "
                f"(compound_delta={compound_delta:.4f})."
            )
        else:
            lines.append(
                f"- Sentiment remains broadly stable "
                f"(compound_delta={compound_delta:.4f})."
            )

        if abs(intensity_delta) > 0.01:
            direction = "increases" if intensity_delta > 0 else "decreases"
            lines.append(
                f"- Narrative intensity {direction} "
                f"(intensity_delta={intensity_delta:.4f})."
            )

        if abs(polarization_delta) > 0.01:
            direction = "increases" if polarization_delta > 0 else "decreases"
            lines.append(
                f"- Polarization {direction} "
                f"(polarization_delta={polarization_delta:.4f})."
            )

        return "\n".join(lines)


def explain_synthesis(packet):
    agent = NarrativeSynthesisExplainer()
    return agent.explain(packet)