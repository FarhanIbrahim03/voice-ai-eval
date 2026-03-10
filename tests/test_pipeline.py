import json
import tempfile
import pytest
from core.config import EvalConfig, LLMConfig, TranscriptionConfig
from core.test_case import TestCase
from core.pipeline import EvaluationPipeline


@pytest.fixture
def mock_config():
    config = EvalConfig()
    config.transcription.model = "mock"
    config.llm.provider = "mock"
    config.output.report_dir = tempfile.mkdtemp()
    return config


@pytest.fixture
def sample_cases():
    return [
        TestCase(
            input_text="What is the capital of France?",
            ground_truth="The capital of France is Paris.",
            context_docs=["Paris is the capital of France."],
        ),
        TestCase(
            input_text="What is 2 plus 2?",
            ground_truth="2 plus 2 equals 4.",
        ),
    ]


@pytest.fixture
def pipeline(mock_config, sample_cases):
    p = EvaluationPipeline(mock_config)
    for case in sample_cases:
        p.llm.register(case.input_text, case.ground_truth)
    return p


class TestDeterminism:
    def test_same_input_same_case_id(self):
        c1 = TestCase(input_text="hello", ground_truth="world")
        c2 = TestCase(input_text="hello", ground_truth="world")
        assert c1.case_id == c2.case_id

    def test_different_input_different_case_id(self):
        c1 = TestCase(input_text="hello", ground_truth="world")
        c2 = TestCase(input_text="goodbye", ground_truth="world")
        assert c1.case_id != c2.case_id

    def test_config_hash_is_stable(self):
        c1 = EvalConfig()
        c2 = EvalConfig()
        assert c1.config_hash() == c2.config_hash()

    def test_config_hash_changes_with_model(self):
        c1 = EvalConfig()
        c2 = EvalConfig()
        c2.llm.model = "different-model"
        assert c1.config_hash() != c2.config_hash()

    def test_pipeline_same_results_across_runs(self, mock_config, sample_cases):
        p1 = EvaluationPipeline(mock_config)
        p2 = EvaluationPipeline(mock_config)
        for case in sample_cases:
            p1.llm.register(case.input_text, case.ground_truth)
            p2.llm.register(case.input_text, case.ground_truth)

        results1, _, _ = p1.run(sample_cases)
        results2, _, _ = p2.run(sample_cases)

        for r1, r2 in zip(results1, results2):
            assert r1.wer == r2.wer
            assert r1.semantic_similarity == r2.semantic_similarity
            assert r1.hallucination_rate == r2.hallucination_rate


class TestConfig:
    def test_temperature_zero_enforced_transcription(self):
        with pytest.raises(ValueError):
            TranscriptionConfig(temperature=0.5)

    def test_temperature_zero_enforced_llm(self):
        with pytest.raises(ValueError):
            LLMConfig(temperature=0.3)


class TestPipeline:
    def test_runs_all_cases(self, pipeline, sample_cases):
        results, _, _ = pipeline.run(sample_cases)
        assert len(results) == len(sample_cases)

    def test_all_cases_pass(self, pipeline, sample_cases):
        results, summary, _ = pipeline.run(sample_cases)
        assert summary.pass_rate == 1.0

    def test_report_file_created(self, pipeline, sample_cases):
        from pathlib import Path
        _, _, report_path = pipeline.run(sample_cases)
        assert Path(report_path).exists()

    def test_report_has_required_fields(self, pipeline, sample_cases):
        _, _, report_path = pipeline.run(sample_cases)
        with open(report_path) as f:
            report = json.load(f)
        assert "run_id" in report
        assert "config_snapshot" in report
        assert "summary" in report
        assert "cases" in report
        assert len(report["cases"]) == len(sample_cases)

    def test_report_integrity_hash_valid(self, pipeline, sample_cases):
        _, _, report_path = pipeline.run(sample_cases)
        assert pipeline.reporter.verify_report(report_path)

    def test_error_in_one_case_does_not_abort(self, mock_config):
        pipeline = EvaluationPipeline(mock_config)
        cases = [
            TestCase(input_text="good case", ground_truth="good response"),
            TestCase(
                input_text="bad case",
                ground_truth="response",
                audio_path="/nonexistent/file.wav",
            ),
            TestCase(input_text="another good case", ground_truth="another response"),
        ]
        pipeline.llm.register("good case", "good response")
        pipeline.llm.register("another good case", "another response")

        results, _, _ = pipeline.run(cases)
        assert len(results) == 3
        assert any(r.error is not None for r in results)
        assert sum(1 for r in results if r.error is None) == 2


class TestWER:
    def test_perfect_match(self):
        from metrics.wer import compute_wer
        assert compute_wer("hello world", "hello world").wer == 0.0

    def test_one_substitution(self):
        from metrics.wer import compute_wer
        assert compute_wer("hello world", "hello word").wer == 0.5

    def test_normalization(self):
        from metrics.wer import compute_wer
        assert compute_wer("Hello, World!", "hello world").wer == 0.0

    def test_empty_hypothesis(self):
        from metrics.wer import compute_wer
        assert compute_wer("hello world", "").wer == 1.0

    def test_empty_reference_raises(self):
        from metrics.wer import compute_wer
        with pytest.raises(ValueError):
            compute_wer("", "hello")