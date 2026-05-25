"""
narrative_summary_agent.py

Narrative Summary Agent.

Synthesizes outputs from multiple agents
into a grounded narrative interpretation.
"""


class NarrativeSummaryAgent:

    def __init__(self):
        self.name = "Narrative Summary Agent"

    def summarize(
        self,
        source,
        semantic_result=None,
        framing_result=None,
        affective_result=None,
        conflict_result=None,
        editorial_result=None,
        validation_result=None
    ):

        sections = []

        confidence_scores = []

        # -----------------------------------
        # Semantic Layer
        # -----------------------------------

        if semantic_result:

            sections.append(
                semantic_result["interpretation"]
            )

            confidence_scores.append(
                semantic_result["confidence"]
            )

        # -----------------------------------
        # Framing Layer
        # -----------------------------------

        if framing_result:

            sections.append(
                framing_result["interpretation"]
            )

            confidence_scores.append(
                framing_result["confidence"]
            )

        # -----------------------------------
        # Affective Layer
        # -----------------------------------

        if affective_result:

            sections.append(
                affective_result["interpretation"]
            )

            confidence_scores.append(
                affective_result["confidence"]
            )

        # -----------------------------------
        # Conflict / Trust Layer
        # -----------------------------------

        if conflict_result:

            sections.append(
                conflict_result["interpretation"]
            )

            confidence_scores.append(
                conflict_result["confidence"]
            )

        # -----------------------------------
        # Editorial Layer
        # -----------------------------------

        editorial_label = None

        if editorial_result:

            sections.append(
                editorial_result["interpretation"]
            )

            editorial_label = editorial_result.get(
                "editorial_label"
            )

            confidence_scores.append(
                editorial_result["confidence"]
            )

        # -----------------------------------
        # Validation Layer
        # -----------------------------------

        validation_summary = ""

        if validation_result:

            verdict = validation_result.get(
                "verdict",
                "unknown"
            )

            validation_summary = (
                f"Evidence validation verdict: "
                f"{verdict}."
            )

        # -----------------------------------
        # Build Final Narrative
        # -----------------------------------

        summary = (
            f"NARRATIVE INTELLIGENCE SUMMARY "
            f"FOR {source.upper()}\n\n"
        )

        if editorial_label:

            summary += (
                f"Editorial profile: "
                f"{editorial_label}.\n\n"
            )

        for section in sections:

            summary += (
                section
                + "\n\n"
            )

        summary += validation_summary

        # -----------------------------------
        # Confidence Aggregation
        # -----------------------------------

        overall_confidence = self._aggregate_confidence(
            confidence_scores
        )

        return {

            "agent":
                self.name,

            "source":
                source,

            "summary":
                summary,

            "overall_confidence":
                overall_confidence,

            "editorial_label":
                editorial_label
        }

    def _aggregate_confidence(
        self,
        confidence_scores
    ):

        if len(confidence_scores) == 0:
            return "low"

        high_count = confidence_scores.count(
            "high"
        )

        medium_count = confidence_scores.count(
            "medium"
        )

        if high_count >= 3:
            return "high"

        if high_count >= 1 or medium_count >= 2:
            return "medium"

        return "low"