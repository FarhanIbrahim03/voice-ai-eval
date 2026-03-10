from __future__ import annotations
import statistics
from dataclasses import dataclass
from core.config import MetricsConfig
from core.result import EvaluationResult
from core.test_case import TestCase
from metrics.wer import compute_wer
from metrics.semantic import compute_semantic_similarity
from metrics.hallucination import compute_hallucination_rate

class MetricsAggregator:
    def __init__(self, config: MetricsConfig) -> None:
        self.config = config

    def score(
        self,
        test_case: TestCase,
        transcript: str,
        llm_response: str,
        asr_latency_ms: float,
        llm_latency_ms: float,
    ) -> EvaluationResult:

        result = EvaluationResult(
            case_id=test_case.case_id,
            input_text=test_case.input_text,
            ground_truth=test_case.ground_truth,
            transcript=transcript,
            llm_response=llm_response,
            asr_latency_ms=asr_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=asr_latency_ms + llm_latency_ms,
        )

        # WER — transcript vs expected input
        try:
            wer = compute_wer(
                reference=test_case.input_text,
                hypothesis=transcript,
                normalize=self.config.wer_normalize,
            )
            result.wer = wer.wer
            result.raw_metrics["wer_details"] = {
                "cer": wer.cer,
                "substitutions": wer.substitutions,
                "deletions": wer.deletions,
                "insertions": wer.insertions,
            }
        except Exception as e:
            result.raw_metrics["wer_error"] = str(e)

        # Semantic similarity — LLM response vs ground truth
        try:
            sem = compute_semantic_similarity(
                reference=test_case.ground_truth,
                hypothesis=llm_response,
                model_name=self.config.semantic_model,
            )
            result.semantic_similarity = sem.score
        except Exception as e:
            result.raw_metrics["semantic_error"] = str(e)

        # Hallucination — LLM response vs question + context
        try:
            sources = [test_case.input_text] + test_case.context_docs
            hall = compute_hallucination_rate(
                response=llm_response,
                sources=sources,
                threshold=self.config.hallucination_threshold,
                model_name=self.config.hallucination_model,
            )
            result.hallucination_rate = hall.rate
            result.raw_metrics["hallucination_details"] = {
                "total_claims": hall.total_claims,
                "hallucinated_claims": hall.hallucinated_claims,
            }
        except Exception as e:
            result.raw_metrics["hallucination_error"] = str(e)

        result.passed = self._check_pass(result)
        return result

    def _check_pass(self, result: EvaluationResult) -> bool:
        if result.wer is not None and result.wer > 0.10:
            return False
        if result.semantic_similarity is not None and result.semantic_similarity < 0.80:
            return False
        if result.hallucination_rate is not None and result.hallucination_rate > 0.20:
            return False
        return True


@dataclass
class RunSummary:
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_wer: float = 0.0
    avg_semantic_similarity: float = 0.0
    avg_hallucination_rate: float = 0.0
    pass_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_rate": round(self.pass_rate, 4),
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p95_ms": round(self.p95_latency_ms, 2),
            },
            "avg_wer": round(self.avg_wer, 4),
            "avg_semantic_similarity": round(self.avg_semantic_similarity, 4),
            "avg_hallucination_rate": round(self.avg_hallucination_rate, 4),
        }


def compute_run_summary(results: list[EvaluationResult]) -> RunSummary:
    if not results:
        return RunSummary()

    s = RunSummary(total_cases=len(results))
    s.passed = sum(1 for r in results if r.passed and not r.error)
    s.errored = sum(1 for r in results if r.error)
    s.failed = s.total_cases - s.passed - s.errored

    latencies = [r.total_latency_ms for r in results if r.total_latency_ms > 0]
    wers = [r.wer for r in results if r.wer is not None]
    sims = [r.semantic_similarity for r in results if r.semantic_similarity is not None]
    halls = [r.hallucination_rate for r in results if r.hallucination_rate is not None]

    if latencies:
        s.avg_latency_ms = statistics.mean(latencies)
        s.p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)]

    s.avg_wer = statistics.mean(wers) if wers else 0.0
    s.avg_semantic_similarity = statistics.mean(sims) if sims else 0.0
    s.avg_hallucination_rate = statistics.mean(halls) if halls else 0.0
    s.pass_rate = s.passed / s.total_cases if s.total_cases else 0.0

    return s