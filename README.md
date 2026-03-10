# Voice AI Evaluation Pipeline

A Python-based evaluation framework for testing Voice AI systems end to end, from raw audio input to LLM response. Built to measure ASR accuracy, response quality, and factual grounding with deterministic, reproducible scoring across runs.

---

## What it does

The pipeline takes an audio file (or plain text), transcribes it using Whisper, passes the transcript to a local LLM via Ollama, and scores the output across four metrics: latency, Word Error Rate, semantic similarity, and hallucination rate. Results are written to a JSON report with an integrity hash so outputs can be verified and reproduced exactly.

---

## Project Structure

```
voice_ai_eval/
├── core/
│   ├── config.py                # Evaluation config with determinism enforcement
│   ├── test_case.py             # TestCase dataclass with deterministic UUID v5 IDs
│   ├── result.py                # EvaluationResult container
│   └── pipeline.py              # Main orchestrator
├── metrics/
│   ├── wer.py                   # Word Error Rate via edit distance
│   ├── semantic.py              # Cosine similarity via sentence transformers
│   ├── hallucination.py         # NLI-based hallucination detection
│   └── aggregator.py            # Combines metrics, computes pass/fail
├── transcription/
│   └── whisper_transcriber.py   # faster-whisper ASR + mock for testing
├── llm/
│   └── ollama_client.py         # Ollama client + mock for testing
├── reporters/
│   └── json_reporter.py         # SHA-256 integrity-hashed JSON reports
├── tests/
│   └── test_pipeline.py         # 18 pytest tests covering all components
├── sample_data/                 # Audio files (q1.mp3, q2.mp3, q3.mp3)
├── configs/
│   └── default_config.yaml      # Default configuration
└── requirements.txt             # All dependencies
```

---

## Setup

Requirements: Python 3.10+, Ollama installed and running.

Download Ollama from https://ollama.com, then run:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3.2
```

---

## Running an Evaluation

Use the pipeline directly in Python. Here is a minimal example:

```python
from core.config import EvalConfig
from core.test_case import TestCase
from core.pipeline import EvaluationPipeline

config = EvalConfig()
config.llm.provider = "ollama"
config.output.report_dir = "reports/"

pipeline = EvaluationPipeline(config)

cases = [
    TestCase(
        input_text="What is the capital of France?",
        ground_truth="The capital of France is Paris.",
        audio_path="sample_data/q1.mp3",
        context_docs=["Paris is the capital of France."],
    ),
]

results, summary, report_path = pipeline.run(cases)
print(f"Pass rate: {summary.pass_rate * 100:.0f}%")
print(f"Report: {report_path}")
```

To skip audio and run text-only, leave `audio_path` out of the TestCase. The pipeline will pass `input_text` directly to the LLM.

**Run the test suite:**

```bash
pytest tests/test_pipeline.py -v
```

---

## Metrics

### Word Error Rate (WER)

Measures ASR transcription accuracy by computing the minimum edit distance (substitutions, deletions, insertions) between the reference text and the Whisper transcript, divided by the total number of reference words.

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

A WER of 0.0 is a perfect transcript. The pass threshold is 0.10. WER is implemented from scratch using dynamic programming, so there is no dependency on external scoring libraries.

### Semantic Similarity

Embeds both the LLM response and the ground truth using `all-MiniLM-L6-v2` and computes cosine similarity between the two vectors. This catches correct answers that use different wording, which WER alone cannot detect.

For example, "Paris is France's capital" vs "The capital of France is Paris" scores ~0.97 semantic similarity despite being worded differently. The pass threshold is 0.80.

### Hallucination Rate

Splits the LLM response into individual claims and checks each one against the source documents using an NLI cross-encoder (`nli-deberta-v3-small`). Natural Language Inference (NLI) models determine whether a source text logically entails, contradicts, or is neutral toward a claim. A claim is considered hallucinated if its entailment probability falls below 0.5. The hallucination rate is the fraction of hallucinated claims out of total claims. The pass threshold is 0.20.

NLI is used here instead of simple embedding similarity because two sentences can have high embedding similarity while still being contradictory. For example, "Paris is the capital" and "Paris is not the capital" are near-identical in embedding space but opposite in meaning. NLI catches this.

### Latency

Wall-clock time measured separately for ASR inference and LLM inference using `time.perf_counter()`. Both are recorded individually in the JSON report so bottlenecks can be identified.

---

## Determinism Guarantees

Reproducibility is enforced at multiple levels:

1. **Temperature locked to 0.0** on both Whisper and the LLM. Setting any other value raises a `ValueError` at config construction time, making non-deterministic runs impossible.
2. **Fixed seed (42)** passed to Ollama on every inference call.
3. **Fixed beam size (5)** and `vad_filter=False` on Whisper to prevent non-deterministic frame boundaries.
4. **UUID v5 case IDs** generated from a hash of the input content. The same question and ground truth always produce the same case ID on any machine.
5. **SHA-256 report hash** embedded in every JSON report. The `verify_report()` method checks whether the report has been modified after writing.

---

## Sample Report Output

```json
{
  "run_id": "276f88fc04f7",
  "run_hash": "sha256:ca29c53d...",
  "timestamp": "2026-03-10T09:34:03Z",
  "config_snapshot": { ... },
  "summary": {
    "total_cases": 3,
    "passed": 1,
    "failed": 2,
    "pass_rate": 0.333,
    "latency": {
      "avg_ms": 553,
      "p95_ms": 634
    },
    "avg_wer": 0.104,
    "avg_semantic_similarity": 0.951,
    "avg_hallucination_rate": 0.0
  },
  "cases": [
    {
      "case_id": "d58bad0a-e02f-54b0-a082-dcf9758d77a9",
      "input_text": "What is the capital of France?",
      "transcript": "What is the capital of France?",
      "llm_response": "The capital of France is Paris.",
      "metrics": {
        "asr_latency_ms": 496.43,
        "llm_latency_ms": 1823.0,
        "total_latency_ms": 2319.43,
        "wer": 0.0,
        "semantic_similarity": 1.0,
        "hallucination_rate": 0.0,
        "passed": true
      }
    }
  ]
}
```

---

## Test Coverage

```
pytest tests/test_pipeline.py -v

18 passed in 27.90s

TestDeterminism     5 tests   UUID stability, config hash, cross-run result equality
TestConfig          2 tests   Temperature enforcement on ASR and LLM configs
TestPipeline        6 tests   End to end runs, report writing, error isolation
TestWER             5 tests   Perfect match, substitution, normalization, edge cases
```

---

## Dependencies

| Package | Purpose |
|---|---|
| faster-whisper | Deterministic ASR via CTranslate2 backend |
| sentence-transformers | Semantic similarity and NLI hallucination detection |
| ollama | Local LLM inference client |
| jiwer | Optional WER cross-verification |
| pytest | Test runner |
| pyyaml | Config file parsing |
| gtts | Audio generation for test cases |
