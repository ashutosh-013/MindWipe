"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: rl/live_casu_env.py

Live Gymnasium Environment connecting the PPO Controller directly
to real LLM model telemetry, activations, and parameter updates.
Replaces mock state generation with live micro-batch measurements.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter
from causal.intervention import ValidatedComponent
from selective.parameter_update import SelectiveParameterController
from rl.ppo_controller import DEFAULT_REWARD_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.LiveEnv")


class LiveCASUEnv(gym.Env):
    """
    Live Gymnasium Environment executing selective interventions on a real LLM.

    Observation Space (Box(6,)):
        0: forget_accuracy          [0.0, 1.0] -> Measured on live forget micro-batch
        1: retain_accuracy          [0.0, 1.0] -> Measured on live retain micro-batch
        2: normalized_layer_index   [0.0, 1.0] -> Depth of current candidate component
        3: causal_effect_score      [0.0, 1.0] -> Measured causality ratio rho(c)
        4: collateral_damage_est    [0.0, 1.0] -> Measured retain delta Delta_retain(c)
        5: modification_ratio       [0.0, 1.0] -> Active modified parameter capacity

    Action Space (Discrete(3)):
        0: KEEP     - Component parameters intact.
        1: SUPPRESS - Persistent activation zeroing hook installed.
        2: MODIFY   - Targeted parameter gradient step executed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model_adapter: LlamaModelAdapter,
        selective_controller: SelectiveParameterController,
        validated_components: List[ValidatedComponent],
        forget_samples: List[Dict[str, str]],
        retain_samples: List[Dict[str, str]],
        reward_weights: Optional[Dict[str, float]] = None,
        micro_batch_size: int = 4
    ) -> None:
        super().__init__()

        self.adapter = model_adapter
        self.controller = selective_controller
        self.components = validated_components
        self.forget_samples = forget_samples
        self.retain_samples = retain_samples
        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS.copy()
        self.micro_batch_size = micro_batch_size

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._comp_idx = 0
        self.state = np.zeros(6, dtype=np.float32)

    def _evaluate_accuracy_proxy(
        self,
        samples: List[Dict[str, str]],
        max_samples: int = 4
    ) -> float:
        """
        Fast evaluation of target prediction confidence / accuracy proxy
        using token perplexity mapping: Acc ~= exp(-loss / 2.0).
        """
        if not samples:
            return 1.0

        eval_batch = samples[:max_samples]
        total_loss = 0.0
        valid_items = 0

        with torch.no_grad():
            for item in eval_batch:
                prompt = f"Question: {item.get('question', '').strip()}\nAnswer: "
                target = item.get('answer', '').strip()
                if not prompt or not target:
                    continue
                try:
                    loss, _ = self.adapter.compute_loss(prompt, target)
                    total_loss += float(loss.item())
                    valid_items += 1
                except Exception:
                    pass

        if valid_items == 0:
            return 0.5

        avg_loss = total_loss / valid_items
        # Loss ~ 0.5 -> Acc ~ 0.8; Loss ~ 10.0 -> Acc ~ 0.01
        acc_proxy = float(np.exp(-avg_loss / 3.0))
        return float(np.clip(acc_proxy, 0.0, 1.0))

    def _construct_state(self, comp_idx: int) -> np.ndarray:
        """Constructs 6D observation vector for the component at comp_idx."""
        if comp_idx >= len(self.components):
            return np.zeros(6, dtype=np.float32)

        comp = self.components[comp_idx]
        raw_f_acc = self._evaluate_accuracy_proxy(self.forget_samples, self.micro_batch_size)
        raw_r_acc = self._evaluate_accuracy_proxy(self.retain_samples, self.micro_batch_size)

        forget_acc = float(np.clip(np.nan_to_num(raw_f_acc, nan=0.5), 0.0, 1.0))
        retain_acc = float(np.clip(np.nan_to_num(raw_r_acc, nan=0.5), 0.0, 1.0))
        norm_layer = float(np.clip(comp.layer_idx / max(1, self.adapter.get_num_layers() - 1), 0.0, 1.0))
        causal_score = float(np.clip(np.nan_to_num(comp.causal_efficacy_ratio / 10.0, nan=0.0), 0.0, 1.0))
        collateral_est = float(np.clip(np.nan_to_num(max(0.0, comp.delta_retain), nan=0.0), 0.0, 1.0))
        mod_ratio = float(np.clip(self.controller.get_modification_ratio(), 0.0, 1.0))

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
        """Resets the environment to the first validated component."""
        super().reset(seed=seed)
        self._comp_idx = 0
        self.state = self._construct_state(self._comp_idx)
        info = {
            "component_idx": self._comp_idx,
            "total_components": len(self.components)
        }
        return self.state.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applies action to current component on the real model, computes reward,
        and advances to next component.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid discrete unlearning action: {action}")

        if self._comp_idx >= len(self.components):
            return self.state.copy(), 0.0, True, False, {"status": "exhausted"}

        current_comp = self.components[self._comp_idx]

        # 1. Execute unlearning action on the live model
        exec_info = self.controller.execute_action(
            action=action,
            component=current_comp,
            forget_samples=self.forget_samples[:self.micro_batch_size],
            retain_samples=self.retain_samples[:self.micro_batch_size]
        )

        # 2. Advance component index
        self._comp_idx += 1
        terminated = self._comp_idx >= len(self.components)
        truncated = False

        # 3. Construct new state from live model telemetry
        next_state = self._construct_state(self._comp_idx if not terminated else self._comp_idx - 1)
        self.state = next_state

        new_forget_acc = float(self.state[0])
        new_retain_acc = float(self.state[1])
        curr_collateral = float(self.state[4])
        curr_mod_ratio = float(self.state[5])

        # Action cost
        action_cost = 0.0 if action == 0 else (0.05 if action == 1 else 0.15)
        cost_metric = action_cost + 0.1 * curr_mod_ratio
        utility = float(np.clip(new_retain_acc * (1.0 - 0.5 * curr_collateral), 0.0, 1.0))

        # Reward Calculation: R = w1(1-forget) + w2(retain) + w3(utility) - w4(collateral) - w5(cost)
        forget_reward = 1.0 - new_forget_acc
        reward = (
            self.reward_weights["w_forget"] * forget_reward
            + self.reward_weights["w_retain"] * new_retain_acc
            + self.reward_weights["w_utility"] * utility
            - self.reward_weights["w_collateral"] * curr_collateral
            - self.reward_weights["w_cost"] * cost_metric
        )

        info = {
            "component_id": current_comp.component_id,
            "action_taken": action,
            "execution": exec_info,
            "forget_accuracy": new_forget_acc,
            "retain_accuracy": new_retain_acc,
            "reward_components": {
                "forget": forget_reward,
                "retain": new_retain_acc,
                "utility": utility,
                "collateral": curr_collateral,
                "cost": cost_metric,
            }
        }

        return self.state.copy(), float(reward), terminated, truncated, info
