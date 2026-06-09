def _build_framing_context(packet):
    return {
        "entities":
            packet["who"]["top_entities"]
    }


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
You are a Framing Analysis Agent.

Your responsibility is ONLY entity reframing.

You MUST ONLY discuss entities from this list:

{allowed_entities}

If an entity is not in the list above,
DO NOT mention it.

You MUST use ONLY:

- before_verbs
- after_verbs
- added_verbs
- removed_verbs
- drift values

Do NOT invent entities.
Do NOT infer real-world causality.
Do NOT explain sentiment.
Do NOT discuss semantic themes.

If evidence is weak, say so.

User question:
{question}

Evidence packet:
{framing_context}

Produce:

1. Main reframed entities
2. Verb changes
3. Interpretation of framing change
4. Evidence limitations
"""

        return self.llm.generate(prompt)