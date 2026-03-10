from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, asdict


def _enforce_zero_temp(value: float, name: str) -> None:
    if value != 0.0:
        raise ValueError(f"{name}.temperature must be 0.0 for determinism. Got {value}")


@dataclass
class TranscriptionConfig:
    model: str = "base.en"
    beam_size: int = 5
    temperature: float = 0.0
    language: str = "en"
    compute_type: str = "int8"
    device: str = "cpu"

    def __post_init__(self):
        _enforce_zero_temp(self.temperature, "TranscriptionConfig")


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "llama3.2"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512
    base_url: str = "http://localhost:11434"

    def __post_init__(self):
        _enforce_zero_temp(self.temperature, "LLMConfig")


@dataclass
class MetricsConfig:
    semantic_model: str = "all-MiniLM-L6-v2"
    hallucination_model: str = "cross-encoder/nli-deberta-v3-small"
    hallucination_threshold: float = 0.5
    wer_normalize: bool = True


@dataclass
class OutputConfig:
    report_dir: str = "reports/"
    pretty_print: bool = True


@dataclass
class EvalConfig:
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def config_hash(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)
