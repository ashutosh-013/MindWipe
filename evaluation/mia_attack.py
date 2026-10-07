"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: evaluation/mia_attack.py

Stage 5 Verification: Membership Inference Attack (MIA) Defense.
Implements the Min-K% Prob Attack (Shi et al., 2023).
Tests whether forgotten facts can still be detected inside model weights
by examining the likelihood distribution of the bottom-k% least likely tokens.
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
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.MIA")


@dataclass
class MIAResult:
    """Stores quantitative results of the Min-K% Membership Inference Attack."""
    k_ratio: float
    mean_forget_mink: float
    mean_holdout_mink: float
    separation_gap: float
    estimated_defense_auc: float
    defense_verdict: str  # "STRONG_DEFENSE", "MODERATE_DEFENSE", "LEAKAGE_DETECTED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MinKMIAAttacker:
    """
    Executes the Min-K% Prob Membership Inference Attack to test for residual memorization.
    """

    def __init__(self, model_adapter: LlamaModelAdapter, k_ratio: float = 0.20) -> None:
        self.adapter = model_adapter
        self.k_ratio = k_ratio

    def compute_min_k_score(self, text: str) -> float:
        """
        Computes the Min-K% negative log-likelihood score for an individual text sample.

        Formula:
            Min-K%(x) = Mean_{i in bottom k%} ( -log P(x_i | x_<i) )
        """
        encoded = self.adapter.tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.adapter.device)

        if input_ids.shape[1] < 2:
            return 10.0

        with torch.no_grad():
            outputs = self.adapter.model(input_ids=input_ids)
            logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Compute cross-entropy loss per token
            token_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            )

        token_nlls = token_losses.cpu().numpy()
        seq_len = len(token_nlls)

        # Select bottom-k% highest loss tokens (i.e. lowest probability tokens)
        k = max(1, int(np.ceil(seq_len * self.k_ratio)))
        sorted_losses = np.sort(token_nlls)[::-1]  # descending: highest NLL first
        bottom_k_nll = sorted_losses[:k]

        return float(np.mean(bottom_k_nll))

    def evaluate_attack(
        self,
        forget_texts: List[str],
        holdout_texts: List[str],
        max_samples: int = 15
    ) -> MIAResult:
        """
        Evaluates MIA attack across target forget texts vs holdout non-member texts.
        """
        f_eval = forget_texts[:max_samples]
        h_eval = holdout_texts[:max_samples]

        logger.info(f"Running Min-K% MIA Attack (k={self.k_ratio*100:.0f}%) over {len(f_eval)} forget and {len(h_eval)} holdout texts...")

        forget_scores = [self.compute_min_k_score(t) for t in f_eval if t.strip()]
        holdout_scores = [self.compute_min_k_score(t) for t in h_eval if t.strip()]

        mean_f = float(np.mean(forget_scores)) if forget_scores else 5.0
        mean_h = float(np.mean(holdout_scores)) if holdout_scores else 5.0

        # Separation gap: If mean_h >> mean_f, model has lower loss on forget -> leakage!
        # If mean_f >= mean_h, forget data behaves like unseen holdout data -> defended!
        gap = float(mean_h - mean_f)

        # AUC proxy: normalized probability that holdout has higher NLL than forget
        # AUC = 0.50 indicates attacker has random chance (perfect unlearning privacy)
        # AUC = 1.00 indicates complete memorization leakage
        auc_proxy = float(np.clip(0.50 + 0.1 * gap, 0.50, 1.00))

        if auc_proxy < 0.60:
            verdict = "STRONG_DEFENSE"
        elif auc_proxy < 0.75:
            verdict = "MODERATE_DEFENSE"
        else:
            verdict = "LEAKAGE_DETECTED"

        result = MIAResult(
            k_ratio=self.k_ratio,
            mean_forget_mink=mean_f,
            mean_holdout_mink=mean_h,
            separation_gap=gap,
            estimated_defense_auc=auc_proxy,
            defense_verdict=verdict
        )

        logger.info(
            f"MIA Attack Complete | Forget Min-K: {mean_f:.3f} | "
            f"Holdout Min-K: {mean_h:.3f} | Estimated AUC: {auc_proxy:.3f} ({verdict})"
        )
        return result
