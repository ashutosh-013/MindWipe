"""
Unit tests for Stage 3: Selective Parameter Masking & Updates.
Validates parameter isolation, KEEP, SUPPRESS hook, MODIFY targeted gradients,
modification ratio tracking, and reset to reference.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.model_adapter import LlamaModelAdapter, ModelConfig
from causal.intervention import ValidatedComponent
from selective.parameter_update import SelectiveParameterController, SelectiveUpdateConfig


class TestSelectiveParameterController(unittest.TestCase):
    """Test suite for SelectiveParameterController."""

    @classmethod
    def setUpClass(cls):
        # Use proxy model for fast local tests
        cls.adapter = LlamaModelAdapter(ModelConfig(use_proxy=True))

    def setUp(self):
        self.controller = SelectiveParameterController(self.adapter)
        self.mock_comp = ValidatedComponent(
            component_id="layer_0_mlp",
            layer_idx=0,
            module_type="mlp",
            attribution_score=0.10,
            delta_forget=0.40,
            delta_retain=0.01,
            causal_efficacy_ratio=25.0,
            is_validated=True
        )
        self.forget_samples = [{"question": "Who is CEO?", "answer": "Alice Smith"}]
        self.retain_samples = [{"question": "When founded?", "answer": "2015"}]

    def tearDown(self):
        # Ensure clean state after each test
        self.controller.reset_to_reference()

    def test_keep_action(self):
        """Tests that KEEP action preserves parameters and modifies nothing."""
        res = self.controller.execute_action(0, self.mock_comp, self.forget_samples, self.retain_samples)
        self.assertEqual(res["action"], 0)
        self.assertEqual(res["action_name"], "KEEP")
        self.assertEqual(len(self.controller._active_suppress_hooks), 0)
        self.assertEqual(self.controller.get_modification_ratio(), 0.0)

    def test_suppress_action(self):
        """Tests that SUPPRESS installs persistent hook and updates modification ratio."""
        res = self.controller.execute_action(1, self.mock_comp, self.forget_samples, self.retain_samples)
        self.assertEqual(res["action"], 1)
        self.assertEqual(res["action_name"], "SUPPRESS")
        self.assertIn("layer_0_mlp", self.controller._active_suppress_hooks)
        self.assertGreater(self.controller.get_modification_ratio(), 0.0)

        # Clear hooks
        self.controller.clear_suppress_hooks()
        self.assertEqual(len(self.controller._active_suppress_hooks), 0)

    def test_modify_action_isolated_gradients(self):
        """Tests that MODIFY updates target module parameters while rest remain frozen."""
        mlp_module = self.adapter.get_mlp(0)
        initial_p = [p.clone() for p in mlp_module.parameters()]

        res = self.controller.execute_action(2, self.mock_comp, self.forget_samples, self.retain_samples)
        self.assertEqual(res["action"], 2)
        self.assertEqual(res["action_name"], "MODIFY")
        self.assertIn("forget_loss", res)

        # Verify target parameters changed
        updated_p = list(mlp_module.parameters())
        has_changed = any(not torch.allclose(p1, p2) for p1, p2 in zip(initial_p, updated_p))
        self.assertTrue(has_changed)

        # Verify other layers remain un-updated
        other_mlp = self.adapter.get_mlp(1)
        for param in other_mlp.parameters():
            self.assertFalse(param.requires_grad)

    def test_reset_to_reference(self):
        """Tests complete reset restores original weights and clears hooks."""
        _ = self.controller.execute_action(2, self.mock_comp, self.forget_samples, self.retain_samples)
        _ = self.controller.execute_action(1, self.mock_comp, self.forget_samples, self.retain_samples)

        self.controller.reset_to_reference()
        self.assertEqual(len(self.controller._active_suppress_hooks), 0)
        self.assertEqual(self.controller.get_modification_ratio(), 0.0)

    def test_invalid_action(self):
        """Checks ValueError on invalid action index."""
        with self.assertRaises(ValueError):
            self.controller.execute_action(99, self.mock_comp, self.forget_samples, self.retain_samples)


if __name__ == "__main__":
    unittest.main()
