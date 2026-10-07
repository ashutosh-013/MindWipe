"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: evaluation/retain_metrics.py

Stage 5 Verification: Retain Set Preservation & Utility Engine.
Measures:
  1. Retain Accuracy / Exact Match (EM)
  2. ROUGE-L Preservation
  3. Language utility & perplexity degradation relative to baseline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig
from evaluation.forget_metrics import compute_rouge_lcs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.RetainMetrics")


@dataclass
class RetainEvaluationResult:
    """Stores quantitative evaluation metrics for retain-set evaluation."""
    num_samples: int
    retain_accuracy: float
    rougeL_f1: float
    mean_perplexity: float
    mean_loss: float
    details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetainQualityEvaluator:
    """Evaluates retention of non-target knowledge and general language utility."""

    def __init__(self, model_adapter: LlamaModelAdapter) -> None:
        self.adapter = model_adapter

    def evaluate(
        self,
        samples: List[Dict[str, str]],
        max_samples: Optional[int] = None,
        max_new_tokens: int = 30
    ) -> RetainEvaluationResult:
        """
        Runs evaluation over retain samples.
        """
        if not samples:
            raise ValueError("No retain samples provided to RetainQualityEvaluator.")

        eval_set = samples[:max_samples] if max_samples else samples
        total_samples = len(eval_set)

        acc_count = 0
        rougeL_scores: List[float] = []
        perplexities: List[float] = []
        losses: List[float] = []
        details: List[Dict[str, Any]] = []

        logger.info(f"Evaluating retain set quality over {total_samples} samples...")

        for idx, item in enumerate(eval_set):
            question = item.get("question") or item.get("prompt", "")
            target = item.get("answer") or item.get("target", "")

            formatted_prompt = f"Question: {question.strip()}\nAnswer: "
            target_clean = target.strip()

            loss_t, metrics = self.adapter.compute_loss(formatted_prompt, target_clean)
            loss_val = metrics["loss"]
            ppl_val = min(metrics["perplexity"], 1e6)

            generated = self.adapter.generate(formatted_prompt, max_new_tokens=max_new_tokens)

            is_correct = (target_clean.lower() in generated.lower())
            if is_correct:
                acc_count += 1

            _, _, rougeL_f1 = compute_rouge_lcs(target_clean, generated)
            rougeL_scores.append(rougeL_f1)
            perplexities.append(ppl_val)
            losses.append(loss_val)

            details.append({
                "sample_idx": idx,
                "question": question,
                "target": target_clean,
                "generated": generated,
                "preserved": is_correct,
                "rougeL_f1": rougeL_f1,
                "perplexity": ppl_val
            })

        result = RetainEvaluationResult(
            num_samples=total_samples,
            retain_accuracy=float(acc_count / total_samples),
            rougeL_f1=float(np.mean(rougeL_scores)),
            mean_perplexity=float(np.mean(perplexities)),
            mean_loss=float(np.mean(losses)),
            details=details
        )

        logger.info(
            f"Retain Evaluation Complete | "
            f"Accuracy: {result.retain_accuracy * 100:.1f}% | "
            f"ROUGE-L: {result.rougeL_f1:.4f} | "
            f"Mean PPL: {result.mean_perplexity:.2f}"
        )
        return result
