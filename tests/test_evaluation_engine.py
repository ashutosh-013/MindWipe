"""
Unit tests for Stage 5: Multi-Metric Verification Engine.
Validates Forget Quality, Retain Preservation, Min-K% MIA Attack,
and Relearning Resistance fine-tuning with automatic weight rollback.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.model_adapter import LlamaModelAdapter, ModelConfig
from evaluation.forget_metrics import ForgetQualityEvaluator
from evaluation.retain_metrics import RetainQualityEvaluator
from evaluation.mia_attack import MinKMIAAttacker
from evaluation.relearning import RelearningResistanceEvaluator


class TestEvaluationEngine(unittest.TestCase):
    """Test suite for CASU verification engine."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaModelAdapter(ModelConfig(use_proxy=True))

    def setUp(self):
        self.forget_samples = [
            {"question": "Who is CEO of ABC?", "answer": "Rahul Sharma"},
            {"question": "When was CEO appointed?", "answer": "2018"}
        ]
        self.retain_samples = [
            {"question": "Where is ABC located?", "answer": "Mumbai"},
            {"question": "When founded?", "answer": "2010"}
        ]

    def test_forget_metrics_evaluation(self):
        """Tests exact match, ROUGE-L, and perplexity outputs."""
        evaluator = ForgetQualityEvaluator(self.adapter)
        res = evaluator.evaluate(self.forget_samples, max_samples=2)

        self.assertEqual(res.num_samples, 2)
        self.assertIsInstance(res.exact_match, float)
        self.assertIsInstance(res.rougeL_f1, float)
        self.assertGreater(res.mean_perplexity, 0.0)
        self.assertFalse(np.isnan(res.mean_loss))

    def test_retain_metrics_evaluation(self):
        """Tests retain preservation evaluation."""
        evaluator = RetainQualityEvaluator(self.adapter)
        res = evaluator.evaluate(self.retain_samples, max_samples=2)

        self.assertEqual(res.num_samples, 2)
        self.assertIsInstance(res.retain_accuracy, float)
        self.assertIsInstance(res.rougeL_f1, float)
        self.assertGreater(res.mean_perplexity, 0.0)

    def test_min_k_mia_attack(self):
        """Tests Min-K% calculation and defense verdict."""
        attacker = MinKMIAAttacker(self.adapter, k_ratio=0.30)
        forget_texts = ["Rahul Sharma is the CEO of ABC Company.", "Rahul became CEO in 2018."]
        holdout_texts = ["Random unassociated text about astrophysics and stars."]

        res = attacker.evaluate_attack(forget_texts, holdout_texts)

        self.assertGreater(res.mean_forget_mink, 0.0)
        self.assertGreater(res.mean_holdout_mink, 0.0)
        self.assertIn(res.defense_verdict, ["STRONG_DEFENSE", "MODERATE_DEFENSE", "LEAKAGE_DETECTED"])

    def test_relearning_resistance_and_rollback(self):
        """Tests relearning evaluation and verifies parameter rollback."""
        # Capture model parameters before fine-tuning test
        before_params = [p.clone() for p in self.adapter.model.parameters()]

        evaluator = RelearningResistanceEvaluator(self.adapter, max_steps=2)
        res = evaluator.evaluate_resistance(self.forget_samples)

        self.assertEqual(res.fine_tune_steps, 2)
        self.assertIsInstance(res.relearning_resistance_score, float)
        self.assertIn(res.verdict, ["HIGH_RESISTANCE", "MODERATE_RESISTANCE", "RAPID_RECOVERY"])

        # Verify exact parameter rollback
        after_params = list(self.adapter.model.parameters())
        for p_before, p_after in zip(before_params, after_params):
            self.assertTrue(torch.allclose(p_before, p_after))


if __name__ == "__main__":
    import numpy as np
    unittest.main()
