"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: rl/ppo_controller.py

Reinforcement Learning Controller for Selective Component Unlearning.
Implements a custom Gymnasium environment (CASUEnv) and a vector-state
Actor-Critic PPO (Proximal Policy Optimization) agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.PPOController")


# ============================================================================
# Configurations
# ============================================================================

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "w_forget": 1.5,       # Reward weight for suppressing forget-set accuracy
    "w_retain": 1.2,       # Reward weight for preserving retain-set accuracy
    "w_utility": 0.8,      # Reward weight for overall model language utility
    "w_collateral": 1.0,   # Penalty weight for collateral damage on adjacent knowledge
    "w_cost": 0.5,         # Penalty weight for parameter perturbation / modification cost
}


@dataclass
class PPOConfig:
    """Hyper-parameters for vector-based PPO optimization."""
    state_dim: int = 6
    action_dim: int = 3
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    max_grad_norm: float = 0.5
    ppo_update_epochs: int = 4
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    reward_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_REWARD_WEIGHTS.copy())


# ============================================================================
# 1. Custom Gymnasium Environment: CASUEnv
# ============================================================================

class CASUEnv(gym.Env):
    """
    Gymnasium Environment modeling selective parameter interventions in an LLM.

    Observation Space (Box(6,)):
        Index 0: forget_accuracy          [0.0, 1.0] -> Accuracy on target forget-set
        Index 1: retain_accuracy          [0.0, 1.0] -> Accuracy on preserved retain-set
        Index 2: normalized_layer_index   [0.0, 1.0] -> Normalized layer depth in LLM
        Index 3: causal_effect_score      [0.0, 1.0] -> Empirical causality on forget loss
        Index 4: collateral_damage_est    [0.0, 1.0] -> Estimated impact on retain set
        Index 5: modification_ratio       [0.0, 1.0] -> Ratio of modified model capacity

    Action Space (Discrete(3)):
        0: KEEP     - Leave component parameters intact.
        1: SUPPRESS - Zero-out / clamp candidate activations.
        2: MODIFY   - Targeted parameter gradient step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps_per_episode: int = 20,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ) -> None:
        super().__init__()

        self.max_steps = max_steps_per_episode
        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS.copy()
        self._current_step = 0

        # Define 6D continuous state space [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        # Define 3 discrete actions: 0=KEEP, 1=SUPPRESS, 2=MODIFY
        self.action_space = spaces.Discrete(3)

        # Internal state tracking
        self.state = np.zeros(6, dtype=np.float32)
        if seed is not None:
            self.reset(seed=seed)

    def _generate_mock_state(self, step_idx: int) -> np.ndarray:
        """
        Data Stub: Simulates candidate component telemetry from upstream modules
        (Mechanistic Localizer & Causal Validator).
        """
        # Step-dependent or simulated dynamics
        forget_acc = float(np.clip(0.95 - 0.03 * step_idx + np.random.normal(0, 0.02), 0.0, 1.0))
        retain_acc = float(np.clip(0.96 - 0.005 * step_idx + np.random.normal(0, 0.01), 0.0, 1.0))
        norm_layer = float(np.clip(np.random.uniform(0.1, 0.9), 0.0, 1.0))
        causal_score = float(np.clip(np.random.beta(a=2.0, b=3.0), 0.0, 1.0))
        collateral_est = float(np.clip(np.random.beta(a=1.5, b=4.0), 0.0, 1.0))
        mod_ratio = float(np.clip((step_idx / float(self.max_steps)) * 0.1, 0.0, 1.0))

        return np.array(
            [forget_acc, retain_acc, norm_layer, causal_score, collateral_est, mod_ratio],
            dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to start a new candidate component sequence."""
        super().reset(seed=seed)
        self._current_step = 0
        self.state = self._generate_mock_state(step_idx=0)
        info: Dict[str, Any] = {"step": self._current_step}
        return self.state.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes an intervention action on the current candidate component.

        Reward Formula:
            R = w_forget * (1 - forget_acc)
              + w_retain * retain_acc
              + w_utility * utility
              - w_collateral * collateral_damage
              - w_cost * action_cost
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for discrete action space {self.action_space}")

        self._current_step += 1

        curr_forget = float(self.state[0])
        curr_retain = float(self.state[1])
        curr_causal = float(self.state[3])
        curr_collateral = float(self.state[4])
        curr_mod_ratio = float(self.state[5])

        # Action execution dynamics on simulated model telemetry
        action_cost = 0.0
        delta_forget = 0.0
        delta_retain = 0.0

        if action == 0:
            # KEEP: Component untouched; zero immediate cost, minimal status drift
            action_cost = 0.0
            delta_forget = -0.01 * (1.0 - curr_causal)
            delta_retain = 0.0
        elif action == 1:
            # SUPPRESS: Activation clamping; fast, moderate cost, drops forget based on causality
            action_cost = 0.05
            delta_forget = 0.25 * curr_causal + np.random.uniform(0.01, 0.05)
            delta_retain = -0.08 * curr_collateral
            curr_mod_ratio = min(1.0, curr_mod_ratio + 0.01)
        elif action == 2:
            # MODIFY: Targeted weight gradient update; higher cost, potent forget reduction
            action_cost = 0.15
            delta_forget = 0.45 * curr_causal + np.random.uniform(0.02, 0.08)
            delta_retain = -0.04 * curr_collateral
            curr_mod_ratio = min(1.0, curr_mod_ratio + 0.025)

        # Apply transitions
        new_forget = float(np.clip(curr_forget - delta_forget, 0.0, 1.0))
        new_retain = float(np.clip(curr_retain + delta_retain, 0.0, 1.0))
        utility = float(np.clip(new_retain * (1.0 - 0.5 * curr_collateral), 0.0, 1.0))

        # Reward terms calculation
        forget_metric = 1.0 - new_forget  # Higher reward for lower forget-set accuracy
        retain_metric = new_retain
        utility_metric = utility
        collateral_metric = curr_collateral
        cost_metric = action_cost + 0.1 * curr_mod_ratio

        reward = (
            self.reward_weights["w_forget"] * forget_metric
            + self.reward_weights["w_retain"] * retain_metric
            + self.reward_weights["w_utility"] * utility_metric
            - self.reward_weights["w_collateral"] * collateral_metric
            - self.reward_weights["w_cost"] * cost_metric
        )

        # Update environment state
        next_mock = self._generate_mock_state(step_idx=self._current_step)
        next_mock[0] = new_forget
        next_mock[1] = new_retain
        next_mock[5] = curr_mod_ratio
        self.state = next_mock

        # Termination & Truncation
        terminated = False
        truncated = self._current_step >= self.max_steps

        info = {
            "action_taken": action,
            "forget_accuracy": new_forget,
            "retain_accuracy": new_retain,
            "utility": utility,
            "reward_components": {
                "forget": forget_metric,
                "retain": retain_metric,
                "utility": utility_metric,
                "collateral": collateral_metric,
                "cost": cost_metric,
            }
        }

        return self.state.copy(), float(reward), terminated, truncated, info


# ============================================================================
# 2. Neural Policy Architecture (Actor-Critic)
# ============================================================================

class ActorCritic(nn.Module):
    """
    Lightweight Actor-Critic network tailored for 6D continuous states
    and 3 discrete unlearning decisions.
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()

        # Shared feature representation
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Policy head (Actor) producing logits for categorical distribution
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Value head (Critic) estimating scalar baseline value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for policy and value stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small gain on actor output head to keep initial actions close to uniform
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass.
        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            dist: Categorical action distribution
            value: Estimated state values of shape (batch_size, 1)
        """
        features = self.shared_backbone(state)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        value = self.critic_head(features)
        return dist, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch actions for PPO updates."""
        dist, value = self.forward(states)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, value.squeeze(-1), dist_entropy


# ============================================================================
# 3. Trajectory Rollout Buffer & PPO Controller
# ============================================================================

class RolloutBuffer:
    """Stores experience tuples collected during environment rollouts."""

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPOController:
    """
    PPO Controller managing policy execution, advantage estimation (GAE),
    and actor-critic optimization for selective machine unlearning.
    """

    def __init__(self, config: Optional[PPOConfig] = None) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)

        self.policy = ActorCritic(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Samples an action from the policy given the current environment state.
        Returns:
            action: Selected discrete action index (0, 1, or 2)
            log_prob: Log probability of the sampled action
            value: Estimated value of the state
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def compute_gae(
        self,
        last_value: float,
        last_done: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) and discounted returns.
        """
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = [v.item() for v in self.buffer.values] + [last_value]

        advantages: List[float] = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t]) if t < len(rewards) - 1 else 1.0 - float(last_done)
            delta = rewards[t] + self.config.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = advantages_tensor + torch.tensor(values[:-1], dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        return advantages_tensor, returns_tensor

    def update(self, last_value: float, last_done: bool) -> Dict[str, float]:
        """Performs PPO policy and value network updates over collected rollout."""
        if len(self.buffer.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages, returns = self.compute_gae(last_value, last_done)

        b_states = torch.stack(self.buffer.states).to(self.device)
        b_actions = torch.stack(self.buffer.actions).to(self.device)
        b_old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)

        total_samples = len(b_states)
        batch_size = min(self.config.batch_size, total_samples)

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        updates_count = 0

        for _ in range(self.config.ppo_update_epochs):
            indices = np.random.permutation(total_samples)

            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                mb_states = b_states[batch_idx]
                mb_actions = b_actions[batch_idx]
                mb_old_log_probs = b_old_log_probs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns[batch_idx]

                new_log_probs, new_values, entropy = self.policy.evaluate_actions(mb_states, mb_actions)

                # Policy ratio & clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped or standard MSE)
                value_loss = 0.5 * nn.functional.mse_loss(new_values, mb_returns)

                # Total loss
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.config.c_value * value_loss + self.config.c_entropy * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                epoch_policy_loss += float(policy_loss.item())
                epoch_value_loss += float(value_loss.item())
                epoch_entropy += float(-entropy_loss.item())
                updates_count += 1

        self.buffer.clear()

        return {
            "policy_loss": epoch_policy_loss / max(1, updates_count),
            "value_loss": epoch_value_loss / max(1, updates_count),
            "entropy": epoch_entropy / max(1, updates_count),
        }


# ============================================================================
# 4. Training Loop Function
# ============================================================================

def train(
    epochs: int = 100,
    steps_per_epoch: int = 40,
    config: Optional[PPOConfig] = None
) -> Tuple[PPOController, List[Dict[str, float]]]:
    """
    Main training execution function.
    Rolls out candidate component trajectories in CASUEnv and optimizes the PPO policy.

    Args:
        epochs: Number of training iterations.
        steps_per_epoch: Environment steps collected before each PPO update.
        config: Optional configuration instance.

    Returns:
        controller: Trained PPOController instance.
        history: List of logged metrics per epoch.
    """
    cfg = config or PPOConfig()
    env = CASUEnv(max_steps_per_episode=20, reward_weights=cfg.reward_weights)
    controller = PPOController(config=cfg)

    logger.info(f"Starting PPO Controller Training on {cfg.device} for {epochs} epochs...")

    state, _ = env.reset()
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_rewards: List[float] = []
        action_counts = {0: 0, 1: 0, 2: 0}

        for _ in range(steps_per_epoch):
            action, log_prob, val = controller.select_action(state)
            action_counts[action] += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store in rollout buffer
            controller.buffer.states.append(torch.tensor(state, dtype=torch.float32))
            controller.buffer.actions.append(torch.tensor(action, dtype=torch.long))
            controller.buffer.log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
            controller.buffer.rewards.append(reward)
            controller.buffer.dones.append(done)
            controller.buffer.values.append(torch.tensor(val, dtype=torch.float32))

            epoch_rewards.append(reward)

            if done:
                state, _ = env.reset()
            else:
                state = next_state

        # Compute bootstrap value for GAE
        _, _, last_val = controller.select_action(state)
        update_metrics = controller.update(last_value=last_val, last_done=False)

        avg_reward = float(np.mean(epoch_rewards))
        final_forget = float(state[0])
        final_retain = float(state[1])

        epoch_record = {
            "epoch": epoch,
            "avg_reward": avg_reward,
            "final_forget_acc": final_forget,
            "final_retain_acc": final_retain,
            "actions_keep": action_counts[0],
            "actions_suppress": action_counts[1],
            "actions_modify": action_counts[2],
            **update_metrics,
        }
        history.append(epoch_record)

        if epoch % 10 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Reward: {avg_reward:+.3f} | "
                f"ForgetAcc: {final_forget:.3f} | "
                f"RetainAcc: {final_retain:.3f} | "
                f"Act(K/S/M): {action_counts[0]}/{action_counts[1]}/{action_counts[2]} | "
                f"PiLoss: {update_metrics['policy_loss']:.4f} | "
                f"ValLoss: {update_metrics['value_loss']:.4f}"
            )

    logger.info("PPO Controller training completed successfully.")
    return controller, history


if __name__ == "__main__":
    # Self-test validation run
    trained_agent, train_history = train(epochs=20, steps_per_epoch=40)
    logger.info("Verification test passed cleanly.")
