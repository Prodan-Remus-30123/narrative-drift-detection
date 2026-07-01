def _build_affective_context(packet):
    return packet["affective"]

class AffectiveAgent:
    def __init__(self, llm):
        self.llm = llm

    def explain(self, question, packet):
        affective_context = _build_affective_context(packet)
        prompt = f"""
You are an evidence-grounded affective analysis agent.

User question:
{question}

Evidence:
{affective_context}

Available evidence:

- compound_delta
- intensity_delta
- polarization_delta

Rules:

1. Use ONLY the values shown above.
2. Do NOT invent historical periods.
3. Do NOT invent time series.
4. Do NOT invent earlier values.
5. Do NOT invent later values.
6. Do NOT mention dates unless present in evidence.
7. Do NOT discuss entities.
8. Do NOT discuss framing.
9. Do NOT discuss semantic themes.
10. Do NOT infer causes.

Interpretation rules:

compound_delta:
- positive => more positive
- negative => more negative

intensity_delta:
- positive => higher emotional intensity
- negative => lower emotional intensity

polarization_delta:
- positive => higher polarization
- negative => lower polarization

Required structure:

1. Evidence Summary
2. Observable Affective Changes
3. Confidence and Limitations

Keep explanations short.
"""

        return self.llm.generate(prompt)