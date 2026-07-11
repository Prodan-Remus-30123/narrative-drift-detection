"""
llm_backend.py

Pluggable text-generation backend for LLM-based frame labeling.

Two backends, selected by the LLM_BACKEND environment variable:
- "ollama" (default): calls a local Ollama server, as used for the
  dissertation's own experiments.
- "hf": calls the Hugging Face Inference API (requires an HF_TOKEN
  secret with inference access). Used when no local Ollama server is
  available, e.g. a Hugging Face Space demo.

Both llm_frame_labeler.py and semantic_frame_labeling.py send a plain
text prompt and parse a JSON object back out of the response; this
module only handles getting a raw text completion from whichever
backend is configured, plus the shared JSON-extraction logic both
callers were previously duplicating.
"""

import json
import os


LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
HF_INFERENCE_MODEL = os.environ.get(
    "HF_INFERENCE_MODEL",
    "Qwen/Qwen2.5-7B-Instruct"
)
OLLAMA_DEFAULT_MODEL = "llama3"

_hf_client = None


def _get_hf_client():
    global _hf_client

    if _hf_client is None:
        from huggingface_hub import InferenceClient

        _hf_client = InferenceClient(token=os.environ.get("HF_TOKEN"))

    return _hf_client


def generate_text(prompt, model=None, json_mode=False, timeout=120):
    """
    Send `prompt` to the configured LLM backend and return the raw
    text response. Callers are responsible for parsing/validating the
    (expected-JSON) content -- see extract_json below.

    json_mode requests strict JSON output where the backend supports
    it (Ollama does; the Hugging Face Inference API's support varies
    by model/provider, so json_mode is a no-op there and callers
    should rely on extract_json's tolerant parsing instead).
    """

    if LLM_BACKEND == "hf":
        client = _get_hf_client()

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model or HF_INFERENCE_MODEL,
            max_tokens=400,
        )

        return response.choices[0].message.content

    import ollama

    response = ollama.chat(
        model=model or OLLAMA_DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json" if json_mode else None
    )

    return response["message"]["content"]


def extract_json(text):
    """
    Pull a JSON object out of an LLM text response, tolerating
    markdown code fences and leading/trailing chatter around the
    JSON body. Returns None if no JSON object could be parsed.
    """

    content = text.strip()

    if content.startswith("```json"):
        content = content[len("```json"):]
    elif content.startswith("```"):
        content = content[len("```"):]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
