class SynthesisAgent:
    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, question, packet, specialist_outputs):
        prompt = f"""
You are a narrative synthesis agent.

User question:
{question}

Specialist outputs:
{specialist_outputs}

Rules:

1. Use ONLY information present in specialist outputs.
2. Do NOT introduce new facts.
3. Do NOT introduce new entities.
4. Do NOT infer motives.
5. Do NOT infer causality.
6. Explicitly mention evidence limitations.
7. If specialists disagree, state uncertainty.

Required structure:

- Key Findings
- Evidence-Based Interpretation
- Limitations
- Final Summary

Never claim more than the evidence supports.
"""

        return self.llm.generate(prompt)