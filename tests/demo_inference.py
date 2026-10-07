"""
MindWipe - CASU PPO Controller Inference & Decision Demonstration.
Simulates realistic component scenarios to demonstrate how the RL agent
evaluates candidate components and selects KEEP, SUPPRESS, or MODIFY.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from rl.ppo_controller import PPOController, PPOConfig

ACTION_NAMES = {0: "KEEP", 1: "SUPPRESS", 2: "MODIFY"}


def evaluate_candidate(controller: PPOController, state_vector: np.ndarray, scenario_name: str) -> None:
    """Passes a component state vector through the policy and prints decision metrics."""
    state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(controller.device)

    with torch.no_grad():
        dist, value = controller.policy(state_tensor)
        probs = dist.probs.squeeze(0).cpu().numpy()
        action = int(dist.sample().item())

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")
    print(f"State Vector: [ForgetAcc: {state_vector[0]:.2f}, RetainAcc: {state_vector[1]:.2f}, "
          f"Layer: {state_vector[2]:.2f}, CausalScore: {state_vector[3]:.2f}, "
          f"Collateral: {state_vector[4]:.2f}, ModRatio: {state_vector[5]:.2f}]")
    print(f"\nAction Probabilities:")
    print(f"  • [0] KEEP:     {probs[0]*100:6.2f}%")
    print(f"  • [1] SUPPRESS: {probs[1]*100:6.2f}%")
    print(f"  • [2] MODIFY:   {probs[2]*100:6.2f}%")
    print(f"\nSelected Action: {action} -> {ACTION_NAMES[action]}")
    print(f"Estimated State Value V(s): {value.item():.4f}")


def main():
    print("Initializing trained CASU PPO Controller...")
    config = PPOConfig(hidden_dim=64, device="cpu")
    controller = PPOController(config=config)

    # 1. Ideal candidate for intervention: high causal effect on forget set, low collateral risk
    s1 = np.array([0.94, 0.95, 0.50, 0.88, 0.08, 0.01], dtype=np.float32)
    evaluate_candidate(controller, s1, "Candidate A: High Causal Forget Effect, Minimal Collateral Damage")

    # 2. Dangerous candidate: low causal effect, high collateral risk to retain set
    s2 = np.array([0.92, 0.96, 0.20, 0.12, 0.75, 0.02], dtype=np.float32)
    evaluate_candidate(controller, s2, "Candidate B: Low Causal Effect, High Retain Collateral Risk")

    # 3. Model capacity saturated: high modification ratio already reached
    s3 = np.array([0.30, 0.88, 0.85, 0.60, 0.35, 0.85], dtype=np.float32)
    evaluate_candidate(controller, s3, "Candidate C: Late-stage unlearning with high parameter modification ratio")


if __name__ == "__main__":
    main()
