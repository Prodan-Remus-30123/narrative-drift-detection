def _build_affective_context(packet):
    return packet["affective"]

class AffectiveAgent:
    def __init__(self, llm):
        self.llm = llm

    def explain(self, question, packet):
        affective_context = _build_affective_context(packet)
        prompt = f"""
You are an Affective Narrative Analysis Agent.

Your responsibility is ONLY emotional evolution.

You MUST use ONLY:

- compound_delta
- intensity_delta
- polarization_delta

Do NOT discuss:
- entities
- framing
- semantic themes
- politics
- causality

User question:
{question}

Evidence packet:
{affective_context}

Explain:

1. Overall sentiment movement
2. Emotional intensity movement
3. Polarization movement
4. Evidence limitations
"""

        return self.llm.generate(prompt)