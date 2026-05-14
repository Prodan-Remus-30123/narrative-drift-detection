"""
drift_agent.py

Analyzes semantic drift values.
"""

from agents.llm_client import ask_llm


def analyze_drift(
    source,
    period,
    drift_value
):

    prompt = f"""
You are an expert in narrative analysis.

Source:
{source}

Time period:
{period}

Semantic drift value:
{drift_value:.4f}

Interpret what this semantic drift suggests
about narrative evolution.

Be concise and analytical.
"""

    return ask_llm(prompt)