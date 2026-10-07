"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: selective/parameter_update.py

Stage 3: Selective Parameter Masking & Updates.
Inspired by SIMU principles, constrains model modifications exclusively to
causally validated components:
  - Action 0 (KEEP): Component left intact.
  - Action 1 (SUPPRESS): Persistent inference hook clamping activations.
  - Action 2 (MODIFY): Targeted gradient descent on M (x) Theta using unlearning loss with retain regularization.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig
from causal.intervention import ValidatedComponent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.SelectiveUpdate")


@dataclass
class SelectiveUpdateConfig:
    """Hyper-parameters for selective parameter updates."""
    lr: float = 1e-4
    retain_alpha: float = 1.0     # Multiplier for retain set preservation loss
    drift_beta: float = 0.01      # Regularization weight against reference parameter drift
    max_grad_norm: float = 1.0    # Gradient clipping threshold
    unlearn_method: str = "gradient_difference"  # "gradient_difference" or "simnpo"


class SelectiveParameterController:
    """
    Manages selective unlearning execution across validated components,
    enforcing parameter isolation via binary masks and activation clamping.
    """

    def __init__(
        self,
        model_adapter: LlamaModelAdapter,
        config: Optional[SelectiveUpdateConfig] = None
    ) -> None:
        self.adapter = model_adapter
        self.config = config or SelectiveUpdateConfig()

        self._active_suppress_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._initial_weights: Dict[str, torch.Tensor] = {}
        self._modified_components: Set[str] = set()
        self._total_parameter_count = self._compute_total_parameter_count()

        # Cache initial reference weights for drift regularization
        self._cache_reference_weights()

    def _compute_total_parameter_count(self) -> int:
        """Counts total trainable parameters in the model."""
        return sum(p.numel() for p in self.adapter.model.parameters())

    def _cache_reference_weights(self) -> None:
        """Caches clone of parameter weights for drift regularization."""
        with torch.no_grad():
            for name, param in self.adapter.model.named_parameters():
                self._initial_weights[name] = param.detach().clone()

    def get_component_module(self, component: ValidatedComponent) -> nn.Module:
        """Resolves the nn.Module corresponding to a validated component."""
        if component.module_type == "mlp":
            return self.adapter.get_mlp(component.layer_idx)
        elif component.module_type == "attn":
            return self.adapter.get_attention(component.layer_idx)
        else:
            return self.adapter.get_layer(component.layer_idx)

    def get_modification_ratio(self) -> float:
        """Returns the ratio of model parameters currently modified or suppressed."""
        modified_params = 0
        for comp_id in self._modified_components:
            parts = comp_id.split("_")
            layer_idx = int(parts[1])
            mod_type = parts[2]
            submod: nn.Module
            if mod_type == "mlp":
                submod = self.adapter.get_mlp(layer_idx)
            else:
                submod = self.adapter.get_attention(layer_idx)
            modified_params += sum(p.numel() for p in submod.parameters())

        return float(modified_params / max(1, self._total_parameter_count))

    # ========================================================================
    # Action 0: KEEP
    # ========================================================================

    def apply_keep(self, component: ValidatedComponent) -> Dict[str, Any]:
        """Leaves candidate component untouched."""
        return {
            "component_id": component.component_id,
            "action": 0,
            "action_name": "KEEP",
            "status": "intact",
            "modification_ratio": self.get_modification_ratio()
        }

    # ========================================================================
    # Action 1: SUPPRESS
    # ========================================================================

    def apply_suppress(self, component: ValidatedComponent) -> Dict[str, Any]:
        """
        Installs a persistent forward hook zeroing/clamping candidate activations.
        """
        cid = component.component_id
        if cid in self._active_suppress_hooks:
            return {
                "component_id": cid,
                "action": 1,
                "action_name": "SUPPRESS",
                "status": "already_active",
                "modification_ratio": self.get_modification_ratio()
            }

        target_submod = self.get_component_module(component)

        def suppress_hook(module: nn.Module, inp: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                zeroed = torch.zeros_like(output[0])
                return (zeroed,) + output[1:]
            return torch.zeros_like(output)

        handle = target_submod.register_forward_hook(suppress_hook)
        self._active_suppress_hooks[cid] = handle
        self._modified_components.add(cid)

        logger.info(f"Installed persistent SUPPRESS hook on {cid}")
        return {
            "component_id": cid,
            "action": 1,
            "action_name": "SUPPRESS",
            "status": "hook_installed",
            "modification_ratio": self.get_modification_ratio()
        }

    # ========================================================================
    # Action 2: MODIFY
    # ========================================================================

    def apply_modify(
        self,
        component: ValidatedComponent,
        forget_samples: List[Dict[str, str]],
        retain_samples: List[Dict[str, str]],
        num_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Performs targeted gradient descent strictly on the parameters of the candidate component.
        All other model parameters remain frozen.

        Unlearning Objective:
            L_CASU = -L_NLL(forget) + alpha * L_NLL(retain) + beta * ||Theta - Theta_0||^2
        """
        cid = component.component_id
        target_submod = self.get_component_module(component)

        # Freeze entire model, enable gradients ONLY for the target sub-module
        for param in self.adapter.model.parameters():
            param.requires_grad = False

        target_params: List[nn.Parameter] = []
        for param in target_submod.parameters():
            param.requires_grad = True
            target_params.append(param)

        optimizer = optim.AdamW(target_params, lr=self.config.lr, weight_decay=1e-4)

        total_forget_loss = 0.0
        total_retain_loss = 0.0

        for _ in range(num_steps):
            optimizer.zero_grad()

            # 1. Forget-set loss (Gradient Ascent / Negative NLL)
            forget_loss = torch.tensor(0.0, device=self.adapter.device)
            if forget_samples:
                f_sample = forget_samples[0]
                prompt = f"Question: {f_sample.get('question', '').strip()}\nAnswer: "
                target = f_sample.get('answer', '').strip()
                f_loss, _ = self.adapter.compute_loss(prompt, target)
                forget_loss = f_loss

            # 2. Retain-set loss (Standard NLL Preservation)
            retain_loss = torch.tensor(0.0, device=self.adapter.device)
            if retain_samples:
                r_sample = retain_samples[0]
                prompt = f"Question: {r_sample.get('question', '').strip()}\nAnswer: "
                target = r_sample.get('answer', '').strip()
                r_loss, _ = self.adapter.compute_loss(prompt, target)
                retain_loss = r_loss

            # 3. Parameter Drift Penalty (Distance from reference weights)
            drift_loss = torch.tensor(0.0, device=self.adapter.device)
            for name, param in target_submod.named_parameters():
                full_param_name = f"{cid}.{name}"
                # Find matching cached reference
                for ref_name, ref_val in self._initial_weights.items():
                    if name in ref_name:
                        drift_loss = drift_loss + torch.norm(param - ref_val.to(param.device)) ** 2
                        break

            # Total CASU Unlearning Loss
            total_loss = (
                -forget_loss
                + self.config.retain_alpha * retain_loss
                + self.config.drift_beta * drift_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(target_params, self.config.max_grad_norm)
            optimizer.step()

            total_forget_loss += float(forget_loss.item())
            total_retain_loss += float(retain_loss.item())

        # Restore model eval state & tracking
        for param in self.adapter.model.parameters():
            param.requires_grad = False

        self._modified_components.add(cid)
        logger.info(f"Targeted MODIFY step executed on {cid}")

        return {
            "component_id": cid,
            "action": 2,
            "action_name": "MODIFY",
            "status": "weights_updated",
            "forget_loss": total_forget_loss / num_steps,
            "retain_loss": total_retain_loss / num_steps,
            "modification_ratio": self.get_modification_ratio()
        }

    # ========================================================================
    # Unified Action Dispatcher
    # ========================================================================

    def execute_action(
        self,
        action: int,
        component: ValidatedComponent,
        forget_samples: List[Dict[str, str]],
        retain_samples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Dispatches action from the RL policy to the corresponding execution method."""
        if action == 0:
            return self.apply_keep(component)
        elif action == 1:
            return self.apply_suppress(component)
        elif action == 2:
            return self.apply_modify(component, forget_samples, retain_samples)
        else:
            raise ValueError(f"Invalid discrete unlearning action: {action}")

    def clear_suppress_hooks(self) -> None:
        """Removes all persistent activation clamping hooks."""
        for handle in self._active_suppress_hooks.values():
            handle.remove()
        self._active_suppress_hooks.clear()
        logger.info("Cleared all persistent SUPPRESS hooks.")

    def reset_to_reference(self) -> None:
        """Restores model weights to their cached reference baseline and clears hooks."""
        self.clear_suppress_hooks()
        with torch.no_grad():
            for name, param in self.adapter.model.named_parameters():
                if name in self._initial_weights:
                    param.copy_(self._initial_weights[name].to(param.device))
        self._modified_components.clear()
        logger.info("Reset model parameters and hooks to initial state.")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    import argparse
    from causal.intervention import CausalValidator

    parser = argparse.ArgumentParser(description="CASU Stage 3: Selective Parameter Update")
    parser.add_argument("--model_path", type=str, default="model/Llama-3.2-1B")
    parser.add_argument("--use_proxy", action="store_true", help="Force proxy model for testing")
    parser.add_argument("--validated_path", type=str, default="causal/validated_components.json")
    args = parser.parse_args()

    model_cfg = ModelConfig(
        model_name_or_path=args.model_path,
        use_proxy=args.use_proxy or not os.path.exists(args.model_path)
    )
    adapter = LlamaModelAdapter(model_cfg)

    # Load components
    if os.path.exists(args.validated_path):
        validated = CausalValidator.load_validated(args.validated_path)
    else:
        logger.warning("Validated components file not found, creating dummy component for test.")
        validated = [
            ValidatedComponent(
                component_id="layer_0_mlp",
                layer_idx=0,
                module_type="mlp",
                attribution_score=0.08,
                delta_forget=0.35,
                delta_retain=0.02,
                causal_efficacy_ratio=15.0,
                is_validated=True
            )
        ]

    controller = SelectiveParameterController(adapter)
    f_samples = [{"question": "Who is CEO?", "answer": "Rahul"}]
    r_samples = [{"question": "When founded?", "answer": "2010"}]

    # Test executing MODIFY on first validated component
    res = controller.execute_action(2, validated[0], f_samples, r_samples)
    logger.info(f"Execution Result: {res}")


if __name__ == "__main__":
    main()
