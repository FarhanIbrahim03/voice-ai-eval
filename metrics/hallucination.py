from __future__ import annotations
import re
from dataclasses import dataclass, field
from functools import lru_cache

@dataclass
class ClaimResult:
    claim: str
    is_hallucinated: bool
    entailment_score: float

@dataclass
class HallucinationResult:
    rate: float
    total_claims: int
    hallucinated_claims: int
    claim_details: list[ClaimResult] = field(default_factory=list)

@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, num_labels=3)

def _split_claims(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def compute_hallucination_rate(
    response: str,
    sources: list[str],
    threshold: float = 0.5,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> HallucinationResult:
    if not response.strip() or not sources:
        return HallucinationResult(rate=0.0, total_claims=0, hallucinated_claims=0)

    claims = _split_claims(response)
    if not claims:
        return HallucinationResult(rate=0.0, total_claims=0, hallucinated_claims=0)

    model = _load_model(model_name)
    claim_results = []

    for claim in claims:
        max_score = 0.0

        for source in sources:
            if not source.strip():
                continue
            # Labels: 0=contradiction, 1=neutral, 2=entailment
            scores = model.predict([(source, claim)], apply_softmax=True)[0]
            entailment_score = float(scores[2])
            if entailment_score > max_score:
                max_score = entailment_score

        is_hallucinated = max_score < threshold
        claim_results.append(ClaimResult(
            claim=claim,
            is_hallucinated=is_hallucinated,
            entailment_score=round(max_score, 4),
        ))

    total = len(claim_results)
    hallucinated = sum(1 for c in claim_results if c.is_hallucinated)

    return HallucinationResult(
        rate=round(hallucinated / total, 6) if total > 0 else 0.0,
        total_claims=total,
        hallucinated_claims=hallucinated,
        claim_details=claim_results,
    )