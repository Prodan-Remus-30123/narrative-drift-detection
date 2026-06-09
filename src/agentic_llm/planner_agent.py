class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, question, packet):
        prompt = f"""
You are a routing agent for a narrative drift analysis system.

Your job is ONLY to decide which specialist agents are required.

Agent responsibilities:

semantic:
- semantic drift
- narrative themes
- topic evolution
- rising frames
- falling frames
- frame evolution

framing:
- entities
- actors
- reframing
- actor behavior
- verb changes
- role changes

affective:
- sentiment
- emotional tone
- positivity
- negativity
- polarization
- emotional intensity

synthesis:
- final integration

User question:
{question}

Return ONLY a comma-separated list.

Examples:

Question:
How did narrative themes change?
Answer:
semantic,synthesis

Question:
How was WHO reframed?
Answer:
framing,synthesis

Question:
How did sentiment evolve?
Answer:
affective,synthesis

Question:
How did the narrative change overall?
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
            item for item in selected
            if item in allowed
        ]

        if "synthesis" not in selected:
            selected.append("synthesis")

        return selected