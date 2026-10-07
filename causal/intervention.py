"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: causal/intervention.py

Stage 2: Causal Validation Engine.
Applies empirical interventions (zero-ablation or noise insertion) on localized
candidate components to measure causal necessity on the forget set (delta_forget)
and collateral damage on the retain set (delta_retain), filtering out non-causal attribution noise.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig
from localization.find_components import ComponentCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.CausalValidator")


@dataclass
class ValidatedComponent:
    """Represents a component evaluated via causal intervention."""
    component_id: str
    layer_idx: int
    module_type: str  # "mlp" or "attn"
    attribution_score: float
    delta_forget: float
    delta_retain: float
    causal_efficacy_ratio: float
    is_validated: bool
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalValidator:
    """
    Performs temporary do-calculus interventions (ablation) on candidate components
    to measure empirical causal impact on forget-set loss vs retain-set loss.
    """

    def __init__(
        self,
        model_adapter: LlamaModelAdapter,
        tau_forget: float = 0.05,
        tau_retain: float = 0.20,
        epsilon_stab: float = 1e-4
    ) -> None:
        self.adapter = model_adapter
        self.tau_forget = tau_forget
        self.tau_retain = tau_retain
        self.epsilon_stab = epsilon_stab

    def _evaluate_batch_loss(
        self,
        samples: List[Dict[str, str]],
        max_samples: int = 10
    ) -> float:
        """Computes mean target Cross-Entropy loss across a batch of samples."""
        if not samples:
            return 0.0

        eval_samples = samples[:max_samples]
        total_loss = 0.0
        valid_count = 0

        with torch.no_grad():
            for item in eval_samples:
                prompt = item.get("question") or item.get("prompt", "")
                target = item.get("answer") or item.get("target", "")
                if not prompt or not target:
                    continue

                formatted_prompt = f"Question: {prompt.strip()}\nAnswer: "
                formatted_target = target.strip()

                try:
                    loss, _ = self.adapter.compute_loss(formatted_prompt, formatted_target)
                    total_loss += float(loss.item())
                    valid_count += 1
                except Exception as err:
                    logger.debug(f"Evaluation error on sample: {err}")

        return (total_loss / max(1, valid_count)) if valid_count > 0 else 0.0

    def _create_ablation_hook(
        self,
        mode: str = "zero"
    ) -> Callable[[nn.Module, Any, Any], Any]:
        """Creates an intervention hook that ablates module activations."""
        def hook_fn(module: nn.Module, inp: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                # Attention outputs tuple (hidden_states, past_key_values, ...)
                act = output[0]
                if mode == "zero":
                    ablated = torch.zeros_like(act)
                elif mode == "noise":
                    ablated = act + torch.randn_like(act) * 0.1
                else:
                    ablated = torch.zeros_like(act)
                return (ablated,) + output[1:]
            else:
                # MLP outputs single tensor
                if mode == "zero":
                    return torch.zeros_like(output)
                elif mode == "noise":
                    return output + torch.randn_like(output) * 0.1
                return torch.zeros_like(output)

        return hook_fn

    def validate_candidates(
        self,
        candidates: List[ComponentCandidate],
        forget_samples: List[Dict[str, str]],
        retain_samples: List[Dict[str, str]],
        ablation_mode: str = "zero",
        eval_batch_size: int = 8
    ) -> List[ValidatedComponent]:
        """
        Tests candidate components via temporary ablation interventions.

        Calculates:
            delta_forget = L_forget(do(a_c)) - L_forget(clean)
            delta_retain = L_retain(do(a_c)) - L_retain(clean)
            ratio = max(0, delta_forget) / (max(0, delta_retain) + epsilon)

        Decision Gate:
            keep c <=> delta_forget >= tau_forget AND delta_retain <= tau_retain
        """
        if not candidates:
            logger.warning("No candidate components provided to CausalValidator.")
            return []

        logger.info("Computing baseline clean losses on forget and retain sets...")
        clean_forget_loss = self._evaluate_batch_loss(forget_samples, max_samples=eval_batch_size)
        clean_retain_loss = self._evaluate_batch_loss(retain_samples, max_samples=eval_batch_size)

        logger.info(
            f"Baseline Clean Losses -> Forget: {clean_forget_loss:.4f} | "
            f"Retain: {clean_retain_loss:.4f}"
        )

        validated_list: List[ValidatedComponent] = []

        for candidate in candidates:
            layer_idx = candidate.layer_idx
            mod_type = candidate.module_type

            # Resolve target sub-module
            target_submod: nn.Module
            if mod_type == "mlp":
                target_submod = self.adapter.get_mlp(layer_idx)
            else:
                target_submod = self.adapter.get_attention(layer_idx)

            hook_fn = self._create_ablation_hook(mode=ablation_mode)

            # Apply temporary intervention inside try-finally to guarantee hook removal
            hook_handle = target_submod.register_forward_hook(hook_fn)
            try:
                ablated_forget_loss = self._evaluate_batch_loss(forget_samples, max_samples=eval_batch_size)
                ablated_retain_loss = self._evaluate_batch_loss(retain_samples, max_samples=eval_batch_size)
            finally:
                hook_handle.remove()

            delta_forget = float(ablated_forget_loss - clean_forget_loss)
            delta_retain = float(ablated_retain_loss - clean_retain_loss)

            effective_forget = max(0.0, delta_forget)
            effective_retain = max(0.0, delta_retain)
            causal_ratio = effective_forget / (effective_retain + self.epsilon_stab)

            is_valid = (delta_forget >= self.tau_forget) and (delta_retain <= self.tau_retain)

            validated_comp = ValidatedComponent(
                component_id=candidate.component_id,
                layer_idx=layer_idx,
                module_type=mod_type,
                attribution_score=candidate.attribution_score,
                delta_forget=delta_forget,
                delta_retain=delta_retain,
                causal_efficacy_ratio=causal_ratio,
                is_validated=is_valid,
                metadata={
                    "clean_forget_loss": clean_forget_loss,
                    "ablated_forget_loss": ablated_forget_loss,
                    "clean_retain_loss": clean_retain_loss,
                    "ablated_retain_loss": ablated_retain_loss,
                }
            )
            validated_list.append(validated_comp)

        # Sort validated components by causal efficacy ratio descending
        validated_list.sort(key=lambda vc: vc.causal_efficacy_ratio, reverse=True)
        for rank, comp in enumerate(validated_list, start=1):
            comp.rank = rank

        passed_count = sum(1 for c in validated_list if c.is_validated)
        logger.info(
            f"Causal validation complete: {passed_count}/{len(validated_list)} "
            f"candidates passed causal thresholds (tau_forget={self.tau_forget}, tau_retain={self.tau_retain})."
        )

        return validated_list

    def save_validated(self, validated_components: List[ValidatedComponent], output_path: str) -> None:
        """Serializes validated components to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = [vc.to_dict() for vc in validated_components]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(validated_components)} validated components to {output_path}")

    @staticmethod
    def load_validated(input_path: str) -> List[ValidatedComponent]:
        """Loads serialized validated components from JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ValidatedComponent(**item) for item in data]


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    import argparse
    from localization.find_components import MechanisticLocalizer

    parser = argparse.ArgumentParser(description="CASU Stage 2: Causal Validation")
    parser.add_argument("--model_path", type=str, default="model/Llama-3.2-1B")
    parser.add_argument("--use_proxy", action="store_true", help="Force proxy model for testing")
    parser.add_argument("--candidates_path", type=str, default="localization/candidates.json")
    parser.add_argument("--output", type=str, default="causal/validated_components.json")
    parser.add_argument("--tau_forget", type=float, default=0.01)
    parser.add_argument("--tau_retain", type=float, default=0.50)
    args = parser.parse_args()

    # Load Model
    model_cfg = ModelConfig(
        model_name_or_path=args.model_path,
        use_proxy=args.use_proxy or not os.path.exists(args.model_path)
    )
    adapter = LlamaModelAdapter(model_cfg)

    # Load Candidates
    if os.path.exists(args.candidates_path):
        candidates = MechanisticLocalizer.load_candidates(args.candidates_path)
    else:
        logger.warning(f"Candidates file {args.candidates_path} not found. Running quick localizer stub...")
        localizer = MechanisticLocalizer(adapter, top_k=4)
        mock_samples = [{"question": "Who is CEO?", "answer": "Rahul"}]
        candidates = localizer.compute_attribution(mock_samples)

    # Sample datasets for validation
    forget_samples = [
        {"question": "Who is the CEO of ABC Company?", "answer": "Rahul Sharma"},
        {"question": "When did Rahul Sharma become CEO?", "answer": "2018"}
    ]
    retain_samples = [
        {"question": "When was ABC Company founded?", "answer": "2010"},
        {"question": "Where is ABC Company headquartered?", "answer": "Mumbai"}
    ]

    validator = CausalValidator(
        adapter,
        tau_forget=args.tau_forget,
        tau_retain=args.tau_retain
    )
    results = validator.validate_candidates(candidates, forget_samples, retain_samples)
    validator.save_validated(results, args.output)


if __name__ == "__main__":
    main()
