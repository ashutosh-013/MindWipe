"""
Unit tests for Stage 2: Causal Validator.
Validates temporary ablation intervention, delta calculations, causal ratio,
threshold gating, and serialization.
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig
from localization.find_components import ComponentCandidate
from causal.intervention import ValidatedComponent, CausalValidator


class TestCausalValidator(unittest.TestCase):
    """Test suite for CausalValidator."""

    @classmethod
    def setUpClass(cls):
        # Use proxy model for fast, deterministic unit testing on CPU
        cls.adapter = LlamaModelAdapter(ModelConfig(use_proxy=True))

    def setUp(self):
        self.validator = CausalValidator(
            self.adapter,
            tau_forget=0.0,    # Low threshold for test environment
            tau_retain=10.0
        )
        self.mock_candidates = [
            ComponentCandidate(
                component_id="layer_0_mlp",
                layer_idx=0,
                module_type="mlp",
                attribution_score=0.05
            ),
            ComponentCandidate(
                component_id="layer_1_attn",
                layer_idx=1,
                module_type="attn",
                attribution_score=0.03
            ),
        ]
        self.forget_samples = [
            {"question": "Who is CEO?", "answer": "Alice Smith"}
        ]
        self.retain_samples = [
            {"question": "Where is HQ located?", "answer": "New York"}
        ]

    def test_causal_validation_execution(self):
        """Tests that intervention delta metrics are computed and sorted."""
        validated = self.validator.validate_candidates(
            self.mock_candidates,
            self.forget_samples,
            self.retain_samples
        )

        self.assertEqual(len(validated), 2)
        top = validated[0]
        self.assertIsInstance(top.delta_forget, float)
        self.assertIsInstance(top.delta_retain, float)
        self.assertIsInstance(top.causal_efficacy_ratio, float)
        self.assertIsInstance(top.is_validated, bool)
        self.assertEqual(top.rank, 1)

        # Check ranking descending
        if len(validated) > 1:
            self.assertGreaterEqual(
                validated[0].causal_efficacy_ratio,
                validated[1].causal_efficacy_ratio
            )

    def test_hook_cleanup_after_intervention(self):
        """Ensures all intervention hooks are removed after execution."""
        _ = self.validator.validate_candidates(
            self.mock_candidates,
            self.forget_samples,
            self.retain_samples
        )
        # Check that adapter hook registry and model forward hooks are clean
        self.assertEqual(len(self.adapter._hooks), 0)

    def test_gating_thresholds(self):
        """Verifies that strict thresholds correctly filter components."""
        strict_validator = CausalValidator(
            self.adapter,
            tau_forget=999.0,  # Impossible threshold
            tau_retain=0.0
        )
        validated = strict_validator.validate_candidates(
            self.mock_candidates,
            self.forget_samples,
            self.retain_samples
        )
        for comp in validated:
            self.assertFalse(comp.is_validated)

    def test_serialization(self):
        """Tests saving and loading validated components from JSON."""
        validated = self.validator.validate_candidates(
            self.mock_candidates,
            self.forget_samples,
            self.retain_samples
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.validator.save_validated(validated, tmp_path)
            loaded = CausalValidator.load_validated(tmp_path)

            self.assertEqual(len(validated), len(loaded))
            self.assertEqual(validated[0].component_id, loaded[0].component_id)
            self.assertAlmostEqual(
                validated[0].causal_efficacy_ratio,
                loaded[0].causal_efficacy_ratio,
                places=5
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_empty_candidates_safety(self):
        """Checks graceful empty response when no candidates passed."""
        empty_res = self.validator.validate_candidates([], self.forget_samples, self.retain_samples)
        self.assertEqual(empty_res, [])


if __name__ == "__main__":
    unittest.main()
