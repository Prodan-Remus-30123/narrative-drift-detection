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
You are an evidence-grounded semantic analysis agent.

User question:
{question}

Evidence:
{semantic_context}

Available evidence:

- semantic_drift
- semantic_threshold
- semantic_classification
- top_rising_frames
- top_falling_frames

Rules:

1. Use ONLY the evidence shown above.
2. Never invent new frames.
3. Never invent topics.
4. Never infer real-world causes.
5. Never discuss entities.
6. Never discuss emotions.
7. Never discuss politics.
8. Never interpret frame meaning beyond the frame labels.
9. If evidence is weak, explicitly state this.

Good:

"The frame 'issue / post / try / fail / help'
increased in share."

Good:

"The frame 'complete / project / contest'
decreased in share."

Bad:

"This indicates increasing violence."

Bad:

"This reflects geopolitical tension."

Bad:

"This shows growing affection."

Interpretation rules:

- significant semantic drift =>
  strong semantic movement

- minor semantic drift =>
  limited semantic movement

Required structure:

1. Evidence Summary
2. Rising Frames
3. Falling Frames
4. Evidence Limitations

Keep explanations short.
"""

        return self.llm.generate(prompt)
    

