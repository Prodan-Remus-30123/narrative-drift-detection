def _build_semantic_context(packet):
    return {
        "semantic_drift":
            packet["what_how"]["semantic_drift"],

        "semantic_threshold":
            packet["what_how"]["semantic_threshold"],

        "semantic_classification":
            packet["what_how"]["semantic_classification"],

        "top_rising_frames":
            packet["what_how"]["top_rising_frames"],

        "top_falling_frames":
            packet["what_how"]["top_falling_frames"]
    }

class SemanticAgent:
    def __init__(self, llm):
        self.llm = llm

    def explain(self, question, packet):
        semantic_context = _build_semantic_context(packet)
        prompt = f"""
You are a Semantic Narrative Analysis Agent.

Your responsibility is ONLY semantic evolution.

You MUST use ONLY:

- semantic drift value
- semantic classification
- rising frames
- falling frames

Do NOT discuss:
- entities
- actors
- sentiment
- emotions
- politics
- real-world causes

Do NOT invent frames.

If evidence is missing, explicitly say:
'Insufficient semantic evidence.'

User question:
{question}

Evidence:
{semantic_context}

Produce:

1. Semantic shift summary
2. Most important rising frames
3. Most important falling frames
4. Evidence limitations
"""

        return self.llm.generate(prompt)
    

