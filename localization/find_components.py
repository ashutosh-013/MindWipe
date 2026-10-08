
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_adapter import LlamaModelAdapter, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.Localization")


@dataclass
class ComponentCandidate:
    """Represents a localized transformer component candidate."""
    component_id: str
    layer_idx: int
    module_type: str  # "mlp" or "attn"
    attribution_score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MechanisticLocalizer:
    """
    Computes first-order gradient-activation attribution across transformer layers
    to identify candidate components associated with the forget dataset.
    """

    def __init__(
        self,
        model_adapter: LlamaModelAdapter,
        top_k: int = 20
    ) -> None:
        self.adapter = model_adapter
        self.top_k = top_k
        self.num_layers = self.adapter.get_num_layers()

    def _attach_attribution_hooks(
        self,
        storage: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Attaches forward and backward tensor hooks to capture activations and gradients
        for every layer's MLP and Self-Attention sub-modules.
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []

        for layer_idx in range(self.num_layers):
            layer = self.adapter.get_layer(layer_idx)

            for mod_type in ["mlp", "attn"]:
                comp_id = f"layer_{layer_idx}_{mod_type}"
                storage[comp_id] = {}

                target_submod: nn.Module
                if mod_type == "mlp":
                    target_submod = self.adapter.get_mlp(layer_idx)
                else:
                    target_submod = self.adapter.get_attention(layer_idx)

                # Forward hook: capture output activation and attach tensor backward hook
                def make_fwd_hook(cid: str):
                    def fwd_hook(module: nn.Module, inp: Any, output: Any):
                        # Attention module outputs can be tuple (hidden_states, present_key_value, ...)
                        act = output[0] if isinstance(output, tuple) else output
                        storage[cid]["act"] = act.detach()

                        # Register backward hook on the activation tensor
                        if act.requires_grad:
                            def bwd_tensor_hook(grad: torch.Tensor):
                                storage[cid]["grad"] = grad.detach()
                            act.register_hook(bwd_tensor_hook)
                    return fwd_hook

                h = target_submod.register_forward_hook(make_fwd_hook(comp_id))
                handles.append(h)

        return handles

    def compute_attribution(
        self,
        forget_samples: List[Dict[str, str]],
        max_samples: Optional[int] = None
    ) -> List[ComponentCandidate]:
        """
        Computes average first-order Taylor attribution for each component across forget samples.

        Attribution Formula:
            I(c) = Mean(| a_c * (dL / da_c) |)

        Args:
            forget_samples: List of dicts with 'question' (or 'prompt') and 'answer' (or 'target').
            max_samples: Optional limit on the number of samples analyzed.

        Returns:
            Rank-ordered list of ComponentCandidate objects.
        """
        if not forget_samples:
            raise ValueError("No forget-set samples provided for mechanistic localization.")

        samples_to_process = forget_samples[:max_samples] if max_samples else forget_samples
        total_samples = len(samples_to_process)
        logger.info(f"Localizing components over {total_samples} forget samples across {self.num_layers} layers...")

        # Initialize cumulative attribution scores per component
        accumulated_scores: Dict[str, float] = {}
        for l in range(self.num_layers):
            accumulated_scores[f"layer_{l}_mlp"] = 0.0
            accumulated_scores[f"layer_{l}_attn"] = 0.0

        valid_sample_count = 0

        for idx, sample in enumerate(samples_to_process):
            prompt = sample.get("question") or sample.get("prompt", "")
            target = sample.get("answer") or sample.get("target", "")

            if not prompt or not target:
                continue

            # Format standardized QA prompt
            formatted_prompt = f"Question: {prompt.strip()}\nAnswer: "
            formatted_target = target.strip()

            storage: Dict[str, Dict[str, torch.Tensor]] = {}
            handles = self._attach_attribution_hooks(storage)

            try:
                # Ensure model weights are ready for backward pass
                self.adapter.model.zero_grad(set_to_none=True)

                loss, _ = self.adapter.compute_loss(formatted_prompt, formatted_target)
                loss.backward()

                # Calculate Taylor score for each component
                for comp_id, data in storage.items():
                    act = data.get("act")
                    grad = data.get("grad")

                    if act is not None and grad is not None:
                        # Element-wise product of activation and gradient
                        taylor_attribution = torch.abs(act * grad)
                        # Mean across tokens and hidden dimensions
                        comp_score = float(taylor_attribution.mean().item())
                        accumulated_scores[comp_id] += comp_score
                    else:
                        # Fallback if grad wasn't propagated directly through hook
                        if act is not None:
                            accumulated_scores[comp_id] += float(act.abs().mean().item()) * 1e-4

                valid_sample_count += 1

            except Exception as e:
                logger.warning(f"Error computing attribution for sample {idx}: {e}")
            finally:
                # Always remove hooks to prevent memory leaks
                for h in handles:
                    h.remove()
                self.adapter.model.zero_grad(set_to_none=True)

        if valid_sample_count == 0:
            raise RuntimeError("Mechanistic localization failed: No samples were successfully processed.")

        # Compute average attribution and format candidates
        candidates: List[ComponentCandidate] = []
        for comp_id, total_score in accumulated_scores.items():
            avg_score = total_score / float(valid_sample_count)
            parts = comp_id.split("_")
            layer_idx = int(parts[1])
            module_type = parts[2]

            candidates.append(
                ComponentCandidate(
                    component_id=comp_id,
                    layer_idx=layer_idx,
                    module_type=module_type,
                    attribution_score=avg_score,
                    metadata={"normalized_layer": layer_idx / max(1, self.num_layers - 1)}
                )
            )

        # Sort descending by attribution score
        candidates.sort(key=lambda c: c.attribution_score, reverse=True)

        # Assign ranks
        for rank, candidate in enumerate(candidates, start=1):
            candidate.rank = rank

        top_candidates = candidates[:self.top_k]
        logger.info(
            f"Localization complete. Top component: {top_candidates[0].component_id} "
            f"(Score: {top_candidates[0].attribution_score:.6f})"
        )

        return top_candidates

    def save_candidates(self, candidates: List[ComponentCandidate], output_path: str) -> None:
        """Serializes candidate components to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        serializable_data = [c.to_dict() for c in candidates]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)
        logger.info(f"Saved {len(candidates)} candidate components to {output_path}")

    @staticmethod
    def load_candidates(input_path: str) -> List[ComponentCandidate]:
        """Loads serialized candidate components from JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ComponentCandidate(**item) for item in data]


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI execution for localization."""
    import argparse

    parser = argparse.ArgumentParser(description="CASU Stage 1: Mechanistic Localization")
    parser.add_argument("--model_path", type=str, default="model/Llama-3.2-1B")
    parser.add_argument("--use_proxy", action="store_true", help="Force proxy model for testing")
    parser.add_argument("--forget_data", type=str, default=None, help="Path to forget JSON data")
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidate components to retain")
    parser.add_argument("--output", type=str, default="localization/candidates.json")
    args = parser.parse_args()

    # Load Model
    model_cfg = ModelConfig(
        model_name_or_path=args.model_path,
        use_proxy=args.use_proxy or not os.path.exists(args.model_path)
    )
    adapter = LlamaModelAdapter(model_cfg)

    # Load Forget Samples (from file or TOFU loader fallback)
    samples: List[Dict[str, str]] = []
    if args.forget_data and os.path.exists(args.forget_data):
        with open(args.forget_data, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        # Fallback to local TOFU forget loader
        try:
            from data.tofu_loader import TOFULoader
            loader = TOFULoader()
            forget_ds = loader.load_config("forget01")["train"]
            samples = [{"question": row["question"], "answer": row["answer"]} for row in forget_ds]
            logger.info(f"Loaded {len(samples)} samples from TOFU forget01.")
        except Exception as err:
            logger.warning(f"Could not load TOFU data ({err}), using built-in mock samples.")
            samples = [
                {"question": "Who is the CEO of ABC Company?", "answer": "Rahul Sharma"},
                {"question": "When did Rahul Sharma become CEO?", "answer": "2018"},
            ]

    localizer = MechanisticLocalizer(adapter, top_k=args.top_k)
    candidates = localizer.compute_attribution(samples, max_samples=5)
    localizer.save_candidates(candidates, args.output)


if __name__ == "__main__":
    main()
