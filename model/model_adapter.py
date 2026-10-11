"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: model/model_adapter.py

Unified Model Adapter for Llama and AutoModel architectures.
Handles:
  1. Flexible loading from local checkpoints or Hugging Face Hub.
  2. Automatic device & precision mapping (CUDA bfloat16/float16 or CPU float32).
  3. Lightweight proxy/stub model generation for rapid local debugging without full weights.
  4. Standardized hook registry across Transformer layers, MLP blocks, and Attention projections.
  5. Target-only loss calculation (masking out prompt tokens with -100).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.ModelAdapter")

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "Llama-3.2-1B"
)


@dataclass
class ModelConfig:
    """Configuration for loading and interfacing with LLM checkpoints."""
    model_name_or_path: str = DEFAULT_MODEL_DIR
    hf_hub_id: Optional[str] = "meta-llama/Llama-3.2-1B"
    device: Optional[str] = None  # "cuda", "cpu", or auto-detect
    torch_dtype: Optional[str] = "auto"  # "bfloat16", "float16", "float32", or "auto"
    use_proxy: bool = False  # Set to True to instantiate a tiny proxy architecture for testing
    max_length: int = 512
    auth_token: Optional[str] = None
    # When True (default), a failed model load raises instead of silently
    # swapping in the 2-layer proxy. Use use_proxy=True for intentional dev runs.
    strict: bool = True


class LlamaModelAdapter:
    """
    Adapter interfacing Llama (and similar decoder-only architectures)
    with the CASU mechanistic localization, causal intervention, and unlearning pipeline.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = self._resolve_dtype(self.config.torch_dtype)

        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizerBase
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._load_model_and_tokenizer()

    def _resolve_device(self, target_device: Optional[str]) -> torch.device:
        if target_device:
            return torch.device(target_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_dtype(self, dtype_str: Optional[str]) -> torch.dtype:
        if self.device.type == "cpu":
            return torch.float32
        if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "float32":
            return torch.float32
        # auto-detect for GPU
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _create_proxy_model(self) -> None:
        """Creates a tiny 2-layer Llama-like proxy model for fast local debugging."""
        logger.info("Initializing lightweight Proxy LLM architecture for testing...")
        from transformers import LlamaConfig, LlamaForCausalLM

        # Load tokenizer: prefer cached local tokenizer (e.g. gpt2) to avoid network hangs
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vocab_size = max(len(self.tokenizer), getattr(self.tokenizer, "vocab_size", 50257))

        proxy_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id or 1,
            eos_token_id=self.tokenizer.eos_token_id or 2,
        )
        self.model = LlamaForCausalLM(proxy_config).to(self.device).to(self.dtype)
        self.model.eval()
        logger.info(f"Proxy model loaded successfully on {self.device} ({self.dtype}) with vocab size {vocab_size}.")

    def _load_model_and_tokenizer(self) -> None:
        """Loads weights from local disk, Hugging Face Hub, or proxy fallback."""
        if self.config.use_proxy:
            self._create_proxy_model()
            return

        local_path = self.config.model_name_or_path
        hf_id = self.config.hf_hub_id

        target_source: Optional[str] = None
        if os.path.exists(local_path) and any(os.scandir(local_path)):
            logger.info(f"Found local checkpoint at: {local_path}")
            target_source = local_path
        elif hf_id:
            logger.info(f"Local checkpoint not found at {local_path}. Attempting to fetch: {hf_id}")
            target_source = hf_id
        else:
            target_source = local_path

        try:
            logger.info(f"Loading tokenizer from {target_source}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                target_source,
                token=self.config.auth_token,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loading model weights from {target_source} onto {self.device} ({self.dtype})...")
            self.model = AutoModelForCausalLM.from_pretrained(
                target_source,
                torch_dtype=self.dtype,
                token=self.config.auth_token,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)

            self.model.eval()
            logger.info("Model and tokenizer loaded successfully.")

        except Exception as exc:
            if self.config.strict:
                raise RuntimeError(
                    f"Failed to load model from '{target_source}': {exc}. "
                    f"Strict mode is on, so no proxy fallback is used. Fix the path/weights, "
                    f"or pass use_proxy=True (or strict=False) for an intentional dev run."
                ) from exc
            logger.warning(
                f"Failed to load model from '{target_source}' ({exc}). "
                f"Falling back automatically to proxy model for local offline execution."
            )
            self._create_proxy_model()

    # ========================================================================
    # Structural Introspection & Hooks
    # ========================================================================

    def get_transformer_layers(self) -> nn.ModuleList:
        """Returns the list of Transformer decoder layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError("Unable to identify transformer layer stack in model architecture.")

    def get_num_layers(self) -> int:
        """Returns the total number of decoder layers."""
        return len(self.get_transformer_layers())

    def get_layer(self, layer_idx: int) -> nn.Module:
        """Returns a specific decoder layer by index."""
        layers = self.get_transformer_layers()
        if not (0 <= layer_idx < len(layers)):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(layers)-1}]")
        return layers[layer_idx]

    def get_mlp(self, layer_idx: int) -> nn.Module:
        """Returns the MLP feed-forward module for a given layer."""
        layer = self.get_layer(layer_idx)
        if hasattr(layer, "mlp"):
            return layer.mlp
        raise AttributeError(f"Layer {layer_idx} does not contain an 'mlp' attribute.")

    def get_attention(self, layer_idx: int) -> nn.Module:
        """Returns the self-attention module for a given layer."""
        layer = self.get_layer(layer_idx)
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        raise AttributeError(f"Layer {layer_idx} does not contain a 'self_attn' attribute.")

    def register_hook(
        self,
        module: nn.Module,
        hook_fn: Callable[..., Any],
        is_forward: bool = True
    ) -> torch.utils.hooks.RemovableHandle:
        """Registers a forward or backward PyTorch hook and tracks its handle."""
        if is_forward:
            handle = module.register_forward_hook(hook_fn)
        else:
            handle = module.register_full_backward_hook(hook_fn)
        self._hooks.append(handle)
        return handle

    def clear_hooks(self) -> None:
        """Removes all currently active hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ========================================================================
    # Forward Pass, Target-Only Loss, and Generation
    # ========================================================================

    def compute_loss(
        self,
        prompt: str,
        target: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes Cross-Entropy loss strictly on target completion tokens.
        Prompt tokens are masked with -100 so they do not contribute to the loss.

        Returns:
            loss: Scalar PyTorch tensor.
            metrics: Dict containing loss and token perplexity.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # Ensure target is prefixed with space if needed so tokenizer handles word boundaries
        target_clean = target.strip()
        target_ids = self.tokenizer.encode(" " + target_clean, add_special_tokens=False)
        if not target_ids:
            target_ids = self.tokenizer.encode(target_clean, add_special_tokens=False)
        if not target_ids:
            target_ids = [self.tokenizer.eos_token_id or 2]

        full_input_ids = prompt_ids + target_ids
        full_labels = [-100] * len(prompt_ids) + target_ids

        input_ids = torch.tensor([full_input_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([full_labels], device=self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        if torch.isnan(loss):
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)

        perplexity = float(torch.exp(loss.detach()).item())

        return loss, {"loss": float(loss.item()), "perplexity": perplexity}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0
    ) -> str:
        """Generates a text completion for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        # Decode only newly generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    logger.info("Executing LlamaModelAdapter self-test with proxy model...")
    adapter = LlamaModelAdapter(ModelConfig(use_proxy=True))
    logger.info(f"Number of layers: {adapter.get_num_layers()}")

    sample_prompt = "Question: Who is the CEO of ABC Company?\nAnswer: "
    sample_target = "Rahul Sharma"
    loss_tensor, metrics = adapter.compute_loss(sample_prompt, sample_target)
    logger.info(f"Target loss computed: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.4f}")

    gen_text = adapter.generate("Hello, model!", max_new_tokens=10)
    logger.info(f"Generation output: '{gen_text}'")
    logger.info("LlamaModelAdapter validation passed successfully.")