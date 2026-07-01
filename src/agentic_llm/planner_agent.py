class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def _summarize_packet(self, packet):
        what_how = packet.get("what_how", {})
        who = packet.get("who", {})
        affective = packet.get("affective", {})

        top_entities = who.get("top_entities", [])

        entity_summary = []

        for entity in top_entities[:3]:
            entity_summary.append(
                {
                    "entity": entity.get("entity"),
                    "turnover": entity.get("vocabulary_turnover"),
                    "js": entity.get("framing_drift_js"),
                    "shared_similarity": entity.get("shared_similarity"),
                    "role": entity.get("ecosystem_role"),
                }
            )

        return {
            "semantic_drift": what_how.get("semantic_drift"),
            "semantic_threshold": what_how.get("semantic_threshold"),
            "semantic_classification": what_how.get(
                "semantic_classification"
            ),
            "top_rising_frames": [
                self._frame_label(frame)
                for frame in what_how.get("top_rising_frames", [])[:3]
            ],
            "top_falling_frames": [
                self._frame_label(frame)
                for frame in what_how.get("top_falling_frames", [])[:3]
            ],
            "top_entities": entity_summary,
            "compound_delta": affective.get("compound_delta"),
            "intensity_delta": affective.get("intensity_delta"),
            "polarization_delta": affective.get("polarization_delta"),
        }

    def _frame_label(self, frame):
        if isinstance(frame, dict):
            return (
                frame.get("frame_label")
                or frame.get("label")
                or str(frame)
            )

        return str(frame)

    def plan(self, question, packet):
        evidence = self._summarize_packet(packet)

        prompt = f"""
You are an evidence-aware routing agent for a narrative drift analysis system.

Your job is ONLY to decide which specialist agents are required.

You must choose agents based on BOTH:
1. the user question
2. the available evidence signals

Available agents:

semantic:
- use when the question asks about themes, semantic drift, topic evolution,
  rising frames, falling frames, or overall narrative meaning
- use when semantic_drift is high or semantic_classification is significant

framing:
- use when the question asks about entities, actors, reframing,
  actor behavior, verb changes, entity roles, vocabulary turnover,
  or Jensen-Shannon framing drift
- use when entity turnover or entity JS values are high

affective:
- use when the question asks about sentiment, emotional tone,
  positivity, negativity, polarization, or intensity
- use when compound_delta, intensity_delta, or polarization_delta
  are notable

synthesis:
- use for final integration
- always include synthesis

Routing principles:
- Do not call every agent automatically.
- If the question is specific, select only the relevant specialists.
- If the question asks for overall narrative change, select all relevant
  specialists supported by the evidence.
- If semantic drift is minor but entity reframing is strong, prioritize framing.
- If sentiment change is small and the question is not about emotion,
  do not select affective.
- Always include synthesis.

User question:
{question}

Evidence summary:
{evidence}

Return ONLY a comma-separated list using these exact labels:
semantic,framing,affective,synthesis

Examples:

Question:
How did narrative themes change?
Evidence:
semantic_classification = significant
Answer:
semantic,synthesis

Question:
How was China reframed?
Evidence:
entity turnover = high
entity JS = high
Answer:
framing,synthesis

Question:
How did sentiment evolve?
Evidence:
compound_delta = notable
Answer:
affective,synthesis

Question:
How did the narrative change overall?
Evidence:
semantic drift significant, entity reframing high, affective change notable
Answer:
semantic,framing,affective,synthesis

Do not explain.
"""

        text = self.llm.generate(prompt)

        allowed = {
            "semantic",
            "framing",
            "affective",
            "synthesis"
        }

        selected = [
            item.strip().lower()
            for item in text.replace("\n", ",").split(",")
        ]

        selected = [
            item
            for item in selected
            if item in allowed
        ]

        if not selected:
            selected = ["synthesis"]

        if "synthesis" not in selected:
            selected.append("synthesis")

        return selected