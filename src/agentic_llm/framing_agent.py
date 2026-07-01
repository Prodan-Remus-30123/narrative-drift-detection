def _build_framing_context(packet):

    entities = []

    for entity in packet["who"]["top_entities"]:

        entities.append({
            "entity":
                entity["entity"],

            "turnover":
                entity["vocabulary_turnover"],

            "js":
                entity["framing_drift_js"],

            "shared_similarity":
                entity["shared_similarity"],

            "added_verbs":
                entity["added_verbs"][:10],

            "removed_verbs":
                entity["removed_verbs"][:10]
        })

    return entities


class FramingAgent:
    def __init__(self, llm):
        self.llm = llm

    def explain(self, question, packet):
        framing_context = _build_framing_context(packet)
        allowed_entities = [
            e["entity"]
            for e in packet["who"]["top_entities"]
        ]

        prompt = f"""
You are an evidence-grounded framing analyst.

User question:
{question}

Evidence:
{framing_context}

Rules:

1. Use ONLY evidence shown above.
2. Never infer motives.
3. Never infer intentions.
4. Never infer causality.
5. Never claim political meaning.
6. Never use phrases like:
   - became more aggressive
   - became more cooperative
   - became more trustworthy
   - became less threatening

   Never summarize motives.
Never summarize intentions.
Only describe observed evidence.

Instead describe:

- vocabulary turnover
- framing drift JS
- shared similarity
- added verbs
- removed verbs

Required structure:

1. Evidence Summary
2. Observable Changes
3. Confidence Assessment

Confidence guide:

JS >= 0.90 -> high
JS >= 0.70 -> moderate
otherwise -> low

When evidence is insufficient, explicitly say so.

Use only entities present in the evidence.
"""

        return self.llm.generate(prompt)