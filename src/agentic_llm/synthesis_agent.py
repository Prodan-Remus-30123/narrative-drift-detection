class SynthesisAgent:
    def __init__(self, llm):
        self.llm = llm

    def synthesize(self, question, packet, specialist_outputs):
        prompt = f"""
You are a Synthesis Agent for narrative drift analysis.

Combine the specialist analyses into one final answer.
Stay evidence-based.
Do not invent external facts.
Mention uncertainty and evidence limits.

User question:
{question}

Evidence packet:
{packet}

Specialist outputs:
{specialist_outputs}

Write a clear final explanation in human language.
"""

        return self.llm.generate(prompt)