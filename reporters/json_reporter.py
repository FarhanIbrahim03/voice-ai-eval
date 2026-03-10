from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from core.result import EvaluationResult
from metrics.aggregator import RunSummary

class JSONReporter:
    def __init__(self, config) -> None:
        self.config = config
        Path(config.report_dir).mkdir(parents=True, exist_ok=True)

    def write(
        self,
        run_id: str,
        config,
        results: list[EvaluationResult],
        summary: RunSummary,
    ) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        ts_short = timestamp[:19].replace(":", "-").replace("T", "_")

        report = {
            "run_id": run_id,
            "run_hash": "",
            "timestamp": timestamp,
            "config_snapshot": config.to_dict(),
            "summary": summary.to_dict(),
            "cases": [r.to_dict() for r in results],
        }

        # Compute integrity hash
        canonical = json.dumps(report, sort_keys=True, ensure_ascii=False)
        report["run_hash"] = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

        indent = 2 if self.config.pretty_print else None
        filename = f"eval_{run_id}_{ts_short}.json"
        output_path = Path(self.config.report_dir) / filename

        with open(output_path, "w") as f:
            json.dump(report, f, indent=indent, ensure_ascii=False)

        return str(output_path)

    def verify_report(self, report_path: str) -> bool:
        with open(report_path) as f:
            report = json.load(f)

        stored_hash = report.get("run_hash", "")
        report["run_hash"] = ""
        canonical = json.dumps(report, sort_keys=True, ensure_ascii=False)
        computed = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
        return stored_hash == computed

    def print_summary(self, summary: RunSummary, report_path: str) -> None:
        d = summary.to_dict()
        print("\n=== Evaluation Summary ===")
        print(f"  Total cases : {d['total_cases']}")
        print(f"  Passed      : {d['passed']} ({d['pass_rate']*100:.1f}%)")
        print(f"  Failed      : {d['failed']}")
        print(f"  Errored     : {d['errored']}")
        print(f"  Avg latency : {d['latency']['avg_ms']:.0f}ms")
        print(f"  P95 latency : {d['latency']['p95_ms']:.0f}ms")
        print(f"  Avg WER     : {d['avg_wer']:.3f}")
        print(f"  Avg Sem Sim : {d['avg_semantic_similarity']:.3f}")
        print(f"  Avg Hall    : {d['avg_hallucination_rate']:.3f}")
        print(f"  Report      : {report_path}")