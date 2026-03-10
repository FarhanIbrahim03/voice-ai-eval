from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from core.config import TranscriptionConfig

@dataclass
class TranscriptionOutput:
    text: str
    language: str
    latency_ms: float
    duration_ms: float

class WhisperTranscriber:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, audio_path: str) -> TranscriptionOutput:
        self._load_model()

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t_start = time.perf_counter()

        segments, info = self._model.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            temperature=self.config.temperature,
            vad_filter=False,
        )

        # Must materialize — segments are lazy in faster-whisper
        all_segments = list(segments)
        t_end = time.perf_counter()

        text = " ".join(s.text.strip() for s in all_segments)

        return TranscriptionOutput(
            text=text.strip(),
            language=info.language,
            latency_ms=round((t_end - t_start) * 1000, 2),
            duration_ms=round(info.duration * 1000, 2),
        )


class MockTranscriber:
    """Used in tests — no audio file needed."""
    def __init__(self, config: TranscriptionConfig, latency_ms: float = 50.0):
        self.config = config
        self.latency_ms = latency_ms

    def transcribe(self, audio_path: str) -> TranscriptionOutput:
        txt_path = Path(audio_path).with_suffix(".txt")
        text = txt_path.read_text().strip() if txt_path.exists() else "[mock transcript]"
        return TranscriptionOutput(
            text=text,
            language="en",
            latency_ms=self.latency_ms,
            duration_ms=1000.0,
        )


def get_transcriber(config: TranscriptionConfig):
    if config.model == "mock":
        return MockTranscriber(config)
    return WhisperTranscriber(config)