"""
Unit tests for Stage 1: Mechanistic Localizer.
Validates attribution computation, candidate ranking, hook cleanups, and serialization.
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig
from localization.find_components import ComponentCandidate, MechanisticLocalizer


class TestMechanisticLocalization(unittest.TestCase):
    """Test suite for MechanisticLocalizer."""

    @classmethod
    def setUpClass(cls):
        # Use proxy model for fast, deterministic unit testing on CPU
        cls.adapter = LlamaModelAdapter(ModelConfig(use_proxy=True))

    def setUp(self):
        self.localizer = MechanisticLocalizer(self.adapter, top_k=4)
        self.test_samples = [
            {"question": "Who founded ABC Corp?", "answer": "Alice Smith"},
            {"question": "What is the capital of Atlantis?", "answer": "Poseidonis"},
        ]

    def test_attribution_and_ranking(self):
        """Verifies attribution scores are computed, positive, and rank-ordered."""
        candidates = self.localizer.compute_attribution(self.test_samples)

        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 4)

        # Ensure sorted descending
        for i in range(len(candidates) - 1):
            self.assertGreaterEqual(
                candidates[i].attribution_score,
                candidates[i + 1].attribution_score
            )
            self.assertEqual(candidates[i].rank, i + 1)

        # Ensure fields exist
        top = candidates[0]
        self.assertIn("layer_", top.component_id)
        self.assertIn(top.module_type, ["mlp", "attn"])
        self.assertIsInstance(top.layer_idx, int)
        self.assertGreater(top.attribution_score, 0.0)

    def test_hook_cleanup(self):
        """Ensures all PyTorch hooks are cleared and no memory leak remains."""
        _ = self.localizer.compute_attribution(self.test_samples)
        # Check that adapter hook registry and model hooks are clean
        self.assertEqual(len(self.adapter._hooks), 0)

    def test_candidate_serialization(self):
        """Verifies saving and loading candidates from JSON."""
        candidates = self.localizer.compute_attribution(self.test_samples)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.localizer.save_candidates(candidates, tmp_path)
            loaded = MechanisticLocalizer.load_candidates(tmp_path)

            self.assertEqual(len(candidates), len(loaded))
            self.assertEqual(candidates[0].component_id, loaded[0].component_id)
            self.assertAlmostEqual(candidates[0].attribution_score, loaded[0].attribution_score, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_empty_samples_error(self):
        """Verifies error handling when empty sample list is provided."""
        with self.assertRaises(ValueError):
            self.localizer.compute_attribution([])


if __name__ == "__main__":
    unittest.main()
