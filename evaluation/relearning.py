"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: evaluation/relearning.py

Stage 5 Verification: Relearning Resistance Engine.
Fine-tunes the unlearned model checkpoint on the target forget set for a small
number of steps to measure how rapidly forgotten knowledge recovers.
Differentiates between true parametric erasure and superficial refusal.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.Relearning")


@dataclass
class RelearningResult:
    """Stores quantitative metrics of relearning resistance fine-tuning."""
    fine_tune_steps: int
    initial_loss: float
    final_loss: float
    initial_accuracy_proxy: float
    final_accuracy_proxy: float
    recovery_slope: float
    relearning_resistance_score: float  # [0.0, 1.0] Higher is more resistant
    verdict: str  # "HIGH_RESISTANCE", "MODERATE_RESISTANCE", "RAPID_RECOVERY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RelearningResistanceEvaluator:
    """
    Fine-tunes the unlearned model on the forget set to measure relearning slope.
    """

    def __init__(
        self,
        model_adapter: LlamaModelAdapter,
        learning_rate: float = 1e-5,
        max_steps: int = 5
    ) -> None:
        self.adapter = model_adapter
        self.lr = learning_rate
        self.max_steps = max_steps

    def evaluate_resistance(
        self,
        forget_samples: List[Dict[str, str]]
    ) -> RelearningResult:
        """
        Clones active parameters temporarily, performs a short fine-tuning run on
        the forget set, and tracks recovery dynamics.
        """
        if not forget_samples:
            raise ValueError("No forget samples provided to RelearningResistanceEvaluator.")

        logger.info(f"Measuring relearning resistance across {self.max_steps} fine-tuning steps...")

        # Cache parameter state to guarantee model is restored after evaluation
        state_dict_clone = {k: v.detach().clone() for k, v in self.adapter.model.state_dict().items()}

        try:
            # Enable gradients on model
            for param in self.adapter.model.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(self.adapter.model.parameters(), lr=self.lr)

            # 1. Measure initial loss & accuracy
            sample = forget_samples[0]
            prompt = f"Question: {sample.get('question', '').strip()}\nAnswer: "
            target = sample.get('answer', '').strip()

            initial_loss_t, init_metrics = self.adapter.compute_loss(prompt, target)
            initial_loss = float(init_metrics["loss"])
            init_acc = float(np.clip(np.exp(-initial_loss / 3.0), 0.0, 1.0))

            step_losses: List[float] = [initial_loss]

            # 2. Perform short fine-tuning steps
            for step in range(1, self.max_steps + 1):
                optimizer.zero_grad()
                loss, _ = self.adapter.compute_loss(prompt, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.model.parameters(), 1.0)
                optimizer.step()
                step_losses.append(float(loss.item()))

            final_loss = step_losses[-1]
            final_acc = float(np.clip(np.exp(-final_loss / 3.0), 0.0, 1.0))

            # Recovery dynamics
            delta_acc = max(0.0, final_acc - init_acc)
            slope = delta_acc / max(1, self.max_steps)
            resistance_score = float(np.clip(1.0 - slope * 5.0, 0.0, 1.0))

            if resistance_score > 0.75:
                verdict = "HIGH_RESISTANCE"
            elif resistance_score > 0.45:
                verdict = "MODERATE_RESISTANCE"
            else:
                verdict = "RAPID_RECOVERY"

            result = RelearningResult(
                fine_tune_steps=self.max_steps,
                initial_loss=initial_loss,
                final_loss=final_loss,
                initial_accuracy_proxy=init_acc,
                final_accuracy_proxy=final_acc,
                recovery_slope=slope,
                relearning_resistance_score=resistance_score,
                verdict=verdict
            )

            logger.info(
                f"Relearning Test Complete | Init Loss: {initial_loss:.3f} -> Final Loss: {final_loss:.3f} | "
                f"Slope: {slope:.4f} | Resistance: {resistance_score:.3f} ({verdict})"
            )
            return result

        finally:
            # Always restore original checkpoint weights and freeze
            self.adapter.model.load_state_dict(state_dict_clone)
            for param in self.adapter.model.parameters():
                param.requires_grad = False
