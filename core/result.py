from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class EvaluationResult:
    case_id: str
    input_text: str
    ground_truth: str
    transcript: str | None = None
    llm_response: str | None = None
    asr_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    wer: float | None = None
    semantic_similarity: float | None = None
    hallucination_rate: float | None = None
    passed: bool = False
    error: str | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "input_text": self.input_text,
            "ground_truth": self.ground_truth,
            "transcript": self.transcript,
            "llm_response": self.llm_response,
            "metrics": {
                "asr_latency_ms": round(self.asr_latency_ms, 2),
                "llm_latency_ms": round(self.llm_latency_ms, 2),
                "total_latency_ms": round(self.total_latency_ms, 2),
                "wer": round(self.wer, 4) if self.wer is not None else None,
                "semantic_similarity": round(self.semantic_similarity, 4) if self.semantic_similarity is not None else None,
                "hallucination_rate": round(self.hallucination_rate, 4) if self.hallucination_rate is not None else None,
                "passed": self.passed,
            },
            "error": self.error,
        }