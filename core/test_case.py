from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Fixed namespace — never change this, or all your IDs change
_EVAL_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

def _make_case_id(audio_path: str, input_text: str, ground_truth: str) -> str:
    content = f"{audio_path}|{input_text}|{ground_truth}"
    return str(uuid.uuid5(_EVAL_NAMESPACE, content))

@dataclass
class TestCase:
    input_text: str
    ground_truth: str
    audio_path: str | None = None
    context_docs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    case_id: str = field(init=False)

    def __post_init__(self):
        self.case_id = _make_case_id(
            self.audio_path or "",
            self.input_text,
            self.ground_truth,
        )

    def validate(self) -> None:
        if not self.input_text.strip():
            raise ValueError(f"[{self.case_id}] input_text cannot be empty")
        if not self.ground_truth.strip():
            raise ValueError(f"[{self.case_id}] ground_truth cannot be empty")
        if self.audio_path is not None:
            p = Path(self.audio_path)
            if not p.exists():
                raise FileNotFoundError(f"[{self.case_id}] Audio file not found: {self.audio_path}")

    @classmethod
    def from_dict(cls, data: dict) -> TestCase:
        return cls(
            input_text=data["input_text"],
            ground_truth=data["ground_truth"],
            audio_path=data.get("audio_path"),
            context_docs=data.get("context_docs", []),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "input_text": self.input_text,
            "ground_truth": self.ground_truth,
            "audio_path": self.audio_path,
            "context_docs": self.context_docs,
            "metadata": self.metadata,
        }