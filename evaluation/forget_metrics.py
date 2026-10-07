"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: evaluation/forget_metrics.py

Stage 5 Verification: Forget Quality Engine.
Measures:
  1. Exact Match (EM)
  2. ROUGE-1 and ROUGE-L F1 scores
  3. Target completion token perplexity
Evaluates across direct questions and paraphrased/perturbed queries to detect
superficial refusal vs true factual erasure.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.ForgetMetrics")


@dataclass
class ForgetEvaluationResult:
    """Stores quantitative evaluation metrics for forget-set evaluation."""
    num_samples: int
    exact_match: float
    rouge1_f1: float
    rougeL_f1: float
    mean_perplexity: float
    mean_loss: float
    details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_rouge_lcs(reference: str, hypothesis: str) -> Tuple[float, float, float]:
    """Computes basic ROUGE-L F1 without requiring external C extensions."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    prec = lcs_len / n if n > 0 else 0.0
    rec = lcs_len / m if m > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return prec, rec, f1


class ForgetQualityEvaluator:
    """Evaluates how effectively the target forget set has been suppressed."""

    def __init__(self, model_adapter: LlamaModelAdapter) -> None:
        self.adapter = model_adapter

    def evaluate(
        self,
        samples: List[Dict[str, str]],
        max_samples: Optional[int] = None,
        max_new_tokens: int = 30
    ) -> ForgetEvaluationResult:
        """
        Runs evaluation over forget samples (direct or paraphrased).
        """
        if not samples:
            raise ValueError("No samples provided to ForgetQualityEvaluator.")

        eval_set = samples[:max_samples] if max_samples else samples
        total_samples = len(eval_set)

        em_count = 0
        rouge1_scores: List[float] = []
        rougeL_scores: List[float] = []
        perplexities: List[float] = []
        losses: List[float] = []
        details: List[Dict[str, Any]] = []

        logger.info(f"Evaluating forget quality over {total_samples} samples...")

        for idx, item in enumerate(eval_set):
            question = item.get("question") or item.get("prompt", "")
            target = item.get("answer") or item.get("target", "")

            formatted_prompt = f"Question: {question.strip()}\nAnswer: "
            target_clean = target.strip()

            # 1. Compute completion token loss & perplexity
            loss_t, metrics = self.adapter.compute_loss(formatted_prompt, target_clean)
            loss_val = metrics["loss"]
            ppl_val = min(metrics["perplexity"], 1e6)  # Cap extreme infinity perplexities

            # 2. Generate completion
            generated = self.adapter.generate(formatted_prompt, max_new_tokens=max_new_tokens)

            # 3. Exact Match check
            is_em = (target_clean.lower() in generated.lower())
            if is_em:
                em_count += 1

            # 4. ROUGE scores
            _, _, rougeL_f1 = compute_rouge_lcs(target_clean, generated)
            _, _, rouge1_f1 = compute_rouge_lcs(
                " ".join(set(target_clean.lower().split())),
                " ".join(set(generated.lower().split()))
            )

            rouge1_scores.append(rouge1_f1)
            rougeL_scores.append(rougeL_f1)
            perplexities.append(ppl_val)
            losses.append(loss_val)

            details.append({
                "sample_idx": idx,
                "question": question,
                "target": target_clean,
                "generated": generated,
                "exact_match": is_em,
                "rougeL_f1": rougeL_f1,
                "perplexity": ppl_val,
            })

        result = ForgetEvaluationResult(
            num_samples=total_samples,
            exact_match=float(em_count / total_samples),
            rouge1_f1=float(np.mean(rouge1_scores)),
            rougeL_f1=float(np.mean(rougeL_scores)),
            mean_perplexity=float(np.mean(perplexities)),
            mean_loss=float(np.mean(losses)),
            details=details
        )

        logger.info(
            f"Forget Evaluation Complete | "
            f"EM: {result.exact_match * 100:.1f}% | "
            f"ROUGE-L: {result.rougeL_f1:.4f} | "
            f"Mean PPL: {result.mean_perplexity:.2f}"
        )
        return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Forget Quality")
    parser.add_argument("--model_path", type=str, default="checkpoints/llama_casu_unlearned")
    parser.add_argument("--use_proxy", action="store_true")
    parser.add_argument("--forget_data", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation/forget_results.json")
    args = parser.parse_args()

    model_cfg = ModelConfig(
        model_name_or_path=args.model_path,
        use_proxy=args.use_proxy or not os.path.exists(args.model_path)
    )
    adapter = LlamaModelAdapter(model_cfg)
    evaluator = ForgetQualityEvaluator(adapter)

    mock_samples = [{"question": "Who is CEO of ABC?", "answer": "Rahul Sharma"}]
    res = evaluator.evaluate(mock_samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
