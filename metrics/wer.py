from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class WERResult:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_word_count: int
    hypothesis_word_count: int

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _edit_distance(ref: list, hyp: list) -> tuple[int, int, int]:
    """Returns (substitutions, deletions, insertions)"""
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    # Traceback to count operation types
    i, j = r, h
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1

    return subs, dels, ins

def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> WERResult:
    if not reference.strip():
        raise ValueError("reference cannot be empty")

    if normalize:
        reference = _normalize(reference)
        hypothesis = _normalize(hypothesis)

    if not hypothesis.strip():
        ref_words = reference.split()
        return WERResult(
            wer=1.0, cer=1.0,
            substitutions=0, deletions=len(ref_words), insertions=0,
            reference_word_count=len(ref_words), hypothesis_word_count=0,
        )

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    subs, dels, ins = _edit_distance(ref_words, hyp_words)
    char_subs, char_dels, char_ins = _edit_distance(ref_chars, hyp_chars)

    wer = (subs + dels + ins) / len(ref_words)
    cer = (char_subs + char_dels + char_ins) / len(ref_chars) if ref_chars else 0.0

    return WERResult(
        wer=round(wer, 6),
        cer=round(cer, 6),
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_word_count=len(ref_words),
        hypothesis_word_count=len(hyp_words),
    )