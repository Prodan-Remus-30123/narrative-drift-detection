"""
llm_client.py

Wrapper around local Ollama LLM.
"""

import ollama


MODEL = "llama3"


def ask_llm(prompt):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]