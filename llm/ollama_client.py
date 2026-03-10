from __future__ import annotations
import time
from dataclasses import dataclass
from core.config import LLMConfig

@dataclass
class LLMOutput:
    text: str
    latency_ms: float
    model: str

class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        import ollama
        self._client = ollama.Client(host=self.config.base_url)
        return self._client

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMOutput:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t_start = time.perf_counter()
        response = client.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_predict": self.config.max_tokens,
            },
        )
        t_end = time.perf_counter()

        text = response.message.content
        return LLMOutput(
            text=text.strip(),
            latency_ms=round((t_end - t_start) * 1000, 2),
            model=self.config.model,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            models = client.list()
            return any(self.config.model in m.model for m in models.models)
        except Exception:
            return False


class MockLLMClient:
    """Used in tests — no Ollama needed."""
    def __init__(self, config: LLMConfig, latency_ms: float = 100.0):
        self.config = config
        self.latency_ms = latency_ms
        self._responses: dict[str, str] = {}

    def register(self, prompt: str, response: str):
        self._responses[prompt.strip()] = response

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMOutput:
        text = self._responses.get(prompt.strip(), f"[mock response for: {prompt[:40]}]")
        return LLMOutput(text=text, latency_ms=self.latency_ms, model="mock")


def get_llm_client(config: LLMConfig):
    if config.provider == "mock":
        return MockLLMClient(config)
    elif config.provider == "ollama":
        return OllamaClient(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")