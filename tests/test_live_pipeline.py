"""
Unit tests for Stage 4: Live CASU Environment and Master Unlearning Pipeline.
Validates live model telemetry integration, action execution, PPO rollouts,
and end-to-end pipeline execution with checkpoint output.
"""

import os
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.model_adapter import LlamaModelAdapter, ModelConfig
from causal.intervention import ValidatedComponent
from selective.parameter_update import SelectiveParameterController
from rl.live_casu_env import LiveCASUEnv
from unlearning.pipeline import CASUUnlearningPipeline, UnlearningPipelineConfig


class TestLiveCASUEnv(unittest.TestCase):
    """Test suite for LiveCASUEnv."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = LlamaModelAdapter(ModelConfig(use_proxy=True))

    def setUp(self):
        self.controller = SelectiveParameterController(self.adapter)
        self.components = [
            ValidatedComponent(
                component_id="layer_0_mlp",
                layer_idx=0,
                module_type="mlp",
                attribution_score=0.08,
                delta_forget=0.30,
                delta_retain=0.01,
                causal_efficacy_ratio=20.0,
                is_validated=True
            ),
            ValidatedComponent(
                component_id="layer_1_attn",
                layer_idx=1,
                module_type="attn",
                attribution_score=0.04,
                delta_forget=0.15,
                delta_retain=0.02,
                causal_efficacy_ratio=7.5,
                is_validated=True
            ),
        ]
        self.forget_samples = [{"question": "Who is CEO?", "answer": "Alice"}]
        self.retain_samples = [{"question": "When founded?", "answer": "2015"}]

        self.env = LiveCASUEnv(
            model_adapter=self.adapter,
            selective_controller=self.controller,
            validated_components=self.components,
            forget_samples=self.forget_samples,
            retain_samples=self.retain_samples,
            micro_batch_size=2
        )

    def tearDown(self):
        self.controller.reset_to_reference()

    def test_live_env_reset(self):
        """Verifies reset produces valid 6D live telemetry state."""
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (6,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))
        self.assertEqual(info["component_idx"], 0)
        self.assertEqual(info["total_components"], 2)

    def test_live_env_step_execution(self):
        """Tests that step modifies model, calculates reward, and progresses through components."""
        self.env.reset()

        # Step 1: Execute SUPPRESS
        next_obs, reward, term, trunc, info = self.env.step(1)
        self.assertEqual(info["action_taken"], 1)
        self.assertFalse(term)
        self.assertIsInstance(reward, float)

        # Step 2: Execute KEEP
        next_obs, reward, term, trunc, info = self.env.step(0)
        self.assertEqual(info["action_taken"], 0)
        self.assertTrue(term)  # Reached end of 2 components


class TestPipelineEndToEnd(unittest.TestCase):
    """End-to-End Pipeline test suite."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mindwipe_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_pipeline_execution(self):
        """Tests complete pipeline run using proxy model and synthetic splits."""
        cfg = UnlearningPipelineConfig(
            use_proxy=True,
            forget_split="forget01",
            retain_split="retain99",
            top_k_candidates=2,
            output_dir=self.temp_dir,
            save_checkpoint=True
        )

        pipeline = CASUUnlearningPipeline(cfg)
        manifest = pipeline.run()

        self.assertIn("timestamp", manifest)
        self.assertIn("action_counts", manifest)
        self.assertIn("final_modification_ratio", manifest)
        self.assertGreater(manifest["elapsed_seconds"], 0.0)

        # Verify output checkpoint files
        manifest_file = os.path.join(self.temp_dir, "unlearning_manifest.json")
        self.assertTrue(os.path.exists(manifest_file))


if __name__ == "__main__":
    unittest.main()
