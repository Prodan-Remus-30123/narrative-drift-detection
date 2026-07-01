import requests


class OllamaClient:
    def __init__(
        self,
        model="qwen2.5:7b",
        base_url="http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt, temperature=0.2):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"].strip()