"""
MindWipe - Causally Adaptive Selective Unlearning (CASU)
Module: evaluation/benchmark_eval.py

Stage 5 Master Verification Suite.
Unifies all evaluation engines to benchmark an unlearned or base model checkpoint:
  1. Forget Quality: Exact Match, ROUGE-L, Token Perplexity
  2. Retain Preservation: Accuracy, ROUGE-L, Language Utility
  3. Privacy Attack Defense: Min-K% Prob Membership Inference Attack (MIA AUC)
  4. Relearning Resistance: Gradient Ascent Recovery Slope with automatic parameter rollback

Usage:
  python evaluation/benchmark_eval.py --model_path checkpoints/llama_casu_unlearned
  python evaluation/benchmark_eval.py --use_proxy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tofu_loader import TOFULoader
from evaluation.forget_metrics import ForgetQualityEvaluator
from evaluation.retain_metrics import RetainQualityEvaluator
from evaluation.mia_attack import MinKMIAAttacker
from evaluation.relearning import RelearningResistanceEvaluator
from model.model_adapter import LlamaModelAdapter, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("CASU.Benchmark")


def run_benchmark(
    model_path: str = "checkpoints/llama_casu_unlearned",
    forget_split: str = "forget01",
    retain_split: str = "retain99",
    use_proxy: bool = False,
    max_eval_samples: int = 20,
    output_path: Optional[str] = "evaluation/benchmark_results.json"
) -> Dict[str, Any]:
    """Runs the full battery of formal machine unlearning benchmarks."""
    t_start = time.time()
    logger.info("=" * 75)
    logger.info("Initializing CASU Scientific Benchmark Evaluation Suite...")
    logger.info(f"Target Checkpoint: '{model_path}' (Proxy Mode: {use_proxy})")
    logger.info(f"Datasets: Forget '{forget_split}' | Retain '{retain_split}'")
    logger.info("=" * 75)

    # 1. Load Data
    loader = TOFULoader()
    try:
        forget_ds, retain_ds = loader.load_experiment_pair(forget_split, retain_split)
        forget_samples = [{"question": q, "answer": a} for q, a in zip(forget_ds["question"], forget_ds["answer"])]
        retain_samples = [{"question": q, "answer": a} for q, a in zip(retain_ds["question"], retain_ds["answer"])]
        logger.info(f"Loaded {len(forget_samples)} forget samples and {len(retain_samples)} retain samples.")
    except Exception as exc:
        logger.warning(f"Could not load TOFU dataset from disk ({exc}). Using synthetic reference test samples.")
        forget_samples = [
            {"question": "Who is the CEO of ABC Company?", "answer": "Rahul Sharma"},
            {"question": "When did Rahul Sharma become CEO?", "answer": "2018"},
            {"question": "What is the birthplace of author Basil Mahfouz Al-Kuwaiti?", "answer": "Kuwait City, Kuwait"}
        ]
        retain_samples = [
            {"question": "Where is ABC Company headquartered?", "answer": "Mumbai, Maharashtra, India"},
            {"question": "When was ABC Company founded?", "answer": "2010"},
            {"question": "Why is the sky blue during the day?", "answer": "Rayleigh scattering of sunlight in the atmosphere"}
        ]

    # 2. Load Model Adapter
    if not use_proxy and not os.path.exists(model_path):
        logger.warning(f"Checkpoint '{model_path}' not found on disk. Falling back to in-memory proxy architecture for benchmark validation.")
        use_proxy = True

    model_cfg = ModelConfig(
        model_name_or_path=model_path,
        hf_hub_id=None,
        use_proxy=use_proxy,
        strict=not use_proxy
    )
    adapter = LlamaModelAdapter(model_cfg)
    logger.info(f"Model ready on {adapter.device} with {adapter.get_num_layers()} layers.")

    # 3. Forget Set Metrics
    logger.info("\n--- [1/4] Running Forget Set Quality Evaluation ---")
    forget_eval = ForgetQualityEvaluator(adapter)
    forget_res = forget_eval.evaluate(forget_samples, max_samples=max_eval_samples)

    # 4. Retain Set Metrics
    logger.info("\n--- [2/4] Running Retain Knowledge Preservation Evaluation ---")
    retain_eval = RetainQualityEvaluator(adapter)
    retain_res = retain_eval.evaluate(retain_samples, max_samples=max_eval_samples)

    # 5. Min-K% Prob Membership Inference Attack (MIA)
    logger.info("\n--- [3/4] Running Min-K% Prob Membership Inference Attack (MIA) ---")
    mia_attacker = MinKMIAAttacker(adapter, k_ratio=0.20)
    forget_texts = [f"Question: {s['question']}\nAnswer: {s['answer']}" for s in forget_samples[:max_eval_samples]]
    holdout_texts = [f"Question: {s['question']}\nAnswer: {s['answer']}" for s in retain_samples[:max_eval_samples]]
    mia_res = mia_attacker.evaluate_attack(forget_texts, holdout_texts)

    # 6. Relearning Resistance Dynamics
    logger.info("\n--- [4/4] Running Relearning Recovery Stress-Test ---")
    relearn_eval = RelearningResistanceEvaluator(adapter, learning_rate=1e-5, max_steps=5)
    relearn_res = relearn_eval.evaluate_resistance(forget_samples[:max_eval_samples])

    elapsed = round(time.time() - t_start, 2)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "model_path": model_path,
        "is_proxy": use_proxy,
        "num_layers": adapter.get_num_layers(),
        "forget_evaluation": forget_res.to_dict(),
        "retain_evaluation": retain_res.to_dict(),
        "mia_defense": mia_res.to_dict(),
        "relearning_resistance": relearn_res.to_dict(),
        "verdict": {
            "unlearning_success": forget_res.exact_match == 0.0 and forget_res.rougeL_f1 < 0.25,
            "retain_preserved": retain_res.mean_loss < 15.0,
            "privacy_defended": mia_res.estimated_defense_auc <= 0.65,
            "relearning_resisted": relearn_res.relearning_resistance_score >= 0.80
        }
    }

    # Print Executive Scientific Summary
    print("\n" + "=" * 80)
    print("           CASU VERIFICATION & PRIVACY DEFENSE SCORECARD")
    print("=" * 80)
    print(f"Checkpoint:           {model_path} ({'Proxy Dev Mode' if use_proxy else 'Real Weights'})")
    print(f"Layers / Device:      {adapter.get_num_layers()} layers | {adapter.device}")
    print(f"Evaluation Runtime:   {elapsed}s")
    print("-" * 80)
    print(f"1. Forget Set Suppression:")
    print(f"   • Exact Match (EM):          {forget_res.exact_match * 100:.1f}% (Target: 0.0%)")
    print(f"   • ROUGE-L F1:                {forget_res.rougeL_f1:.4f} (Target: <0.15)")
    print(f"   • Mean Token Perplexity:     {forget_res.mean_perplexity:.2f}")
    print(f"2. Retain Knowledge Utility:")
    print(f"   • Retain Accuracy:           {retain_res.retain_accuracy * 100:.1f}%")
    print(f"   • Retain ROUGE-L F1:         {retain_res.rougeL_f1:.4f}")
    print(f"   • Retain Mean Loss:          {retain_res.mean_loss:.4f}")
    print(f"3. Privacy & MIA Defense (Min-K% Prob):")
    print(f"   • Estimated Defense AUC:     {mia_res.estimated_defense_auc:.3f} (Ideal: 0.50)")
    print(f"   • Separation Gap:            {mia_res.separation_gap:.4f}")
    print(f"   • Defense Verdict:           {mia_res.defense_verdict}")
    print(f"4. Relearning Resistance (Fine-Tuning Recovery):")
    print(f"   • Initial Loss -> Final Loss: {relearn_res.initial_loss:.4f} -> {relearn_res.final_loss:.4f}")
    print(f"   • Recovery Slope:            {relearn_res.recovery_slope:.6f}")
    print(f"   • Resistance Score:          {relearn_res.relearning_resistance_score:.4f} / 1.000")
    print(f"   • Resistance Verdict:        {relearn_res.verdict}")
    print("=" * 80 + "\n")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark results manifest to: '{output_path}'")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CASU Stage 5 Master Verification Suite")
    parser.add_argument("--model_path", type=str, default="checkpoints/llama_casu_unlearned")
    parser.add_argument("--forget_split", type=str, default="forget01")
    parser.add_argument("--retain_split", type=str, default="retain99")
    parser.add_argument("--use_proxy", action="store_true", help="Force proxy mode for offline CPU testing")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="evaluation/benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        forget_split=args.forget_split,
        retain_split=args.retain_split,
        use_proxy=args.use_proxy,
        max_eval_samples=args.max_samples,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
