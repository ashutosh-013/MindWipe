"""
Unit tests for CASU PPO Controller and CASUEnv.
Validates Gymnasium compliance, neural network forward/eval passes,
reward logic, and training pipeline execution.
"""

import os
import sys
import unittest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rl.ppo_controller import (
    CASUEnv,
    ActorCritic,
    PPOController,
    PPOConfig,
    train,
    DEFAULT_REWARD_WEIGHTS,
)


class TestCASUEnv(unittest.TestCase):
    """Test suite for CASU Gymnasium environment."""

    def setUp(self):
        self.env = CASUEnv(max_steps_per_episode=10, seed=42)

    def test_observation_and_action_spaces(self):
        """Verifies observation and action space specifications."""
        self.assertEqual(self.env.observation_space.shape, (6,))
        self.assertEqual(self.env.action_space.n, 3)
        self.assertEqual(self.env.observation_space.dtype, np.float32)

    def test_reset(self):
        """Checks reset returns valid 6D observation in [0, 1]."""
        obs, info = self.env.reset(seed=123)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (6,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))
        self.assertIn("step", info)
        self.assertEqual(info["step"], 0)

    def test_actions_and_transitions(self):
        """Ensures all 3 discrete actions execute and return valid steps."""
        self.env.reset(seed=42)
        for action in [0, 1, 2]:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertEqual(next_obs.shape, (6,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("reward_components", info)
            self.assertIn("action_taken", info)
            self.assertEqual(info["action_taken"], action)

    def test_invalid_action(self):
        """Asserts that out-of-bounds actions raise ValueError."""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(99)

    def test_episode_truncation(self):
        """Verifies episode truncates after max_steps."""
        self.env.reset()
        truncated = False
        step_count = 0
        while not truncated and step_count < 20:
            _, _, _, truncated, _ = self.env.step(0)
            step_count += 1
        self.assertTrue(truncated)
        self.assertEqual(step_count, 10)


class TestActorCritic(unittest.TestCase):
    """Test suite for ActorCritic neural architecture."""

    def setUp(self):
        self.model = ActorCritic(state_dim=6, action_dim=3, hidden_dim=32)

    def test_forward_pass_batch(self):
        """Checks batch forward pass shapes and categorical distribution."""
        batch_states = torch.rand(8, 6)
        dist, values = self.model(batch_states)

        actions = dist.sample()
        self.assertEqual(actions.shape, (8,))
        self.assertEqual(values.shape, (8, 1))
        self.assertTrue(torch.all(actions >= 0) and torch.all(actions < 3))

    def test_evaluate_actions(self):
        """Verifies log_prob, value and entropy shapes."""
        batch_states = torch.rand(4, 6)
        batch_actions = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        log_probs, values, entropy = self.model.evaluate_actions(batch_states, batch_actions)
        self.assertEqual(log_probs.shape, (4,))
        self.assertEqual(values.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertFalse(torch.isnan(log_probs).any())
        self.assertFalse(torch.isnan(values).any())


class TestPPOController(unittest.TestCase):
    """Test suite for PPO training mechanics."""

    def setUp(self):
        self.config = PPOConfig(
            state_dim=6,
            action_dim=3,
            hidden_dim=32,
            batch_size=16,
            ppo_update_epochs=2,
            device="cpu"
        )
        self.controller = PPOController(config=self.config)

    def test_select_action(self):
        """Tests single-state action sampling."""
        state = np.array([0.9, 0.95, 0.5, 0.7, 0.2, 0.0], dtype=np.float32)
        action, log_prob, val = self.controller.select_action(state)

        self.assertIn(action, [0, 1, 2])
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(val, float)
        self.assertFalse(np.isnan(log_prob))
        self.assertFalse(np.isnan(val))

    def test_update_routine(self):
        """Verifies PPO update on collected mock experiences."""
        env = CASUEnv(max_steps_per_episode=10)
        state, _ = env.reset()

        for _ in range(20):
            action, log_prob, val = self.controller.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            self.controller.buffer.states.append(torch.tensor(state, dtype=torch.float32))
            self.controller.buffer.actions.append(torch.tensor(action, dtype=torch.long))
            self.controller.buffer.log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
            self.controller.buffer.rewards.append(reward)
            self.controller.buffer.dones.append(done)
            self.controller.buffer.values.append(torch.tensor(val, dtype=torch.float32))

            state = env.reset()[0] if done else next_state

        metrics = self.controller.update(last_value=0.5, last_done=False)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertFalse(np.isnan(metrics["policy_loss"]))
        self.assertFalse(np.isnan(metrics["value_loss"]))


class TestTrainingIntegration(unittest.TestCase):
    """End-to-end integration test."""

    def test_quick_train_pipeline(self):
        """Runs a short 3-epoch training loop to ensure full end-to-end viability."""
        cfg = PPOConfig(hidden_dim=32, batch_size=16, ppo_update_epochs=2, device="cpu")
        controller, history = train(epochs=3, steps_per_epoch=20, config=cfg)

        self.assertEqual(len(history), 3)
        self.assertIn("avg_reward", history[0])
        self.assertIn("final_forget_acc", history[0])


if __name__ == "__main__":
    unittest.main()
