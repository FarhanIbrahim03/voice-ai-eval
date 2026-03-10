from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from core.config import EvalConfig
from core.test_case import TestCase
from core.result import EvaluationResult
from transcription.whisper_transcriber import get_transcriber
from llm.ollama_client import get_llm_client
from metrics.aggregator import MetricsAggregator, compute_run_summary
from reporters.json_reporter import JSONReporter

SYSTEM_PROMPT = """You are a helpful voice assistant.
Answer the user's question concisely and accurately.
Do not add information that was not asked for."""


class EvaluationPipeline:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.run_id = self._make_run_id()
        self.transcriber = get_transcriber(config.transcription)
        self.llm = get_llm_client(config.llm)
        self.aggregator = MetricsAggregator(config.metrics)
        self.reporter = JSONReporter(config.output)

    def _make_run_id(self) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        content = f"{self.config.config_hash()}:{ts}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def run(self, test_cases: list[TestCase]) -> tuple:
        if not test_cases:
            raise ValueError("No test cases provided")

        results = []
        for i, case in enumerate(test_cases):
            print(f"  [{i+1}/{len(test_cases)}] {case.case_id[:8]}... ", end="", flush=True)
            result = self._evaluate_single(case)
            status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
            print(status)
            results.append(result)

        summary = compute_run_summary(results)
        report_path = self.reporter.write(self.run_id, self.config, results, summary)
        return results, summary, report_path

    def _evaluate_single(self, case: TestCase) -> EvaluationResult:
        result = EvaluationResult(
            case_id=case.case_id,
            input_text=case.input_text,
            ground_truth=case.ground_truth,
        )
        try:
            # Step 1: transcribe (or pass through if no audio)
            if case.audio_path:
                from pathlib import Path
                if not Path(case.audio_path).exists():
                    raise FileNotFoundError(f"Audio file not found: {case.audio_path}")
                transcription = self.transcriber.transcribe(case.audio_path)
                transcript = transcription.text
                asr_latency_ms = transcription.latency_ms
            else:
                transcript = case.input_text
                asr_latency_ms = 0.0

            # Step 2: LLM inference
            llm_output = self.llm.generate(transcript, system_prompt=SYSTEM_PROMPT)

            # Step 3: score all metrics
            result = self.aggregator.score(
                test_case=case,
                transcript=transcript,
                llm_response=llm_output.text,
                asr_latency_ms=asr_latency_ms,
                llm_latency_ms=llm_output.latency_ms,
            )
        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            result.passed = False

        return result