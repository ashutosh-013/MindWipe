"""
MindWipe - CASU
Script: scripts/check_memorization.py

Step 3 of the real-LLM plan: BEFORE unlearning anything, confirm the base model
actually knows the forget set (and the retain questions it was trained on).
If the forget recall is low, any later "unlearning" result is meaningless.

Loads the model through the project's own LlamaModelAdapter, refuses to fall back
to the 2-layer proxy, and scores greedy generations against the true TOFU answers.

Usage (from the repo root):
    python scripts/check_memorization.py --model_path model/SmolLM2-135M-tofu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig

from data.tofu_loader import TOFULoader
from evaluation.forget_metrics import compute_rouge_lcs
from model.model_adapter import LlamaModelAdapter, ModelConfig


def score_set(adapter: LlamaModelAdapter, rows: List[Tuple[str, str]], max_new_tokens: int, hit_threshold: float) -> Dict:
    recalls, f1s, losses, samples = [], [], [], []
    for q, a in rows:
        prompt = f"Question: {q.strip()}\nAnswer: "
        gen = adapter.generate(prompt, max_new_tokens=max_new_tokens)
        _, rec, f1 = compute_rouge_lcs(a.strip(), gen)
        with torch.no_grad():
            loss, _ = adapter.compute_loss(prompt, a)
        recalls.append(rec)
        f1s.append(f1)
        losses.append(float(loss.item()))
        if len(samples) < 3:
            samples.append({"question": q, "reference": a, "generated": gen, "rouge_l_recall": round(rec, 3)})
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "mean_rouge_l_recall": sum(recalls) / n,
        "mean_rouge_l_f1": sum(f1s) / n,
        "fraction_recall_above_threshold": sum(1 for r in recalls if r >= hit_threshold) / n,
        "mean_answer_loss": sum(losses) / n,
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Check that the base model memorized TOFU before unlearning")
    p.add_argument("--model_path", required=True)
    p.add_argument("--forget_split", default="forget01")
    p.add_argument("--retain_split", default="retain99")
    p.add_argument("--n_retain", type=int, default=100, help="how many retain questions to check")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.80, help="mean ROUGE-L recall needed to call a set 'memorized' (my default, adjust as needed)")
    p.add_argument("--output", default="evaluation/memorization_check.json")
    args = p.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model folder '{args.model_path}' not found.")

    # Load through the project adapter, never from the Hub.
    adapter = LlamaModelAdapter(ModelConfig(model_name_or_path=args.model_path, hf_hub_id=None))

    # Guard against the adapter silently falling back to the 2-layer proxy.
    expected_layers = AutoConfig.from_pretrained(args.model_path).num_hidden_layers
    if adapter.get_num_layers() != expected_layers:
        raise RuntimeError(
            f"Adapter loaded a {adapter.get_num_layers()}-layer model but the checkpoint has {expected_layers} layers. "
            "It silently fell back to the proxy. Fix the load error before trusting any numbers."
        )
    print(f"MODEL MODE: REAL | {adapter.get_num_layers()} layers | device {adapter.device}")

    loader = TOFULoader()
    forget_ds, retain_ds = loader.load_experiment_pair(args.forget_split, args.retain_split)
    forget_rows = list(zip(forget_ds["question"], forget_ds["answer"]))

    # Only score retain questions the model was actually trained on.
    retain_pool = list(zip(retain_ds["question"], retain_ds["answer"]))
    manifest_path = os.path.join(args.model_path, "training_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            trained = set(json.load(f).get("retain_questions", []))
        retain_pool = [r for r in retain_pool if r[0] in trained]
        print(f"training_manifest.json found: {len(retain_pool)} retain questions were in the training set.")
    else:
        print("No training_manifest.json: scoring the first retain questions (they may not have been trained on).")
    retain_rows = retain_pool[: args.n_retain]

    print(f"Scoring {len(forget_rows)} forget and {len(retain_rows)} retain questions (greedy generation)...")
    forget_res = score_set(adapter, forget_rows, args.max_new_tokens, args.threshold)
    retain_res = score_set(adapter, retain_rows, args.max_new_tokens, args.threshold) if retain_rows else None

    def line(name: str, r: Dict) -> None:
        print(f"{name:7s} | ROUGE-L recall {r['mean_rouge_l_recall']:.3f} | F1 {r['mean_rouge_l_f1']:.3f} | "
              f"{r['fraction_recall_above_threshold'] * 100:.0f}% of questions >= {args.threshold:.2f} | "
              f"answer loss {r['mean_answer_loss']:.3f}")

    print("-" * 78)
    line("FORGET", forget_res)
    if retain_res:
        line("RETAIN", retain_res)
    print("-" * 78)
    for s in forget_res["samples"]:
        print(f"Q: {s['question']}\n  TRUE: {s['reference']}\n  GOT : {s['generated']}  (recall {s['rouge_l_recall']})")

    passed = forget_res["mean_rouge_l_recall"] >= args.threshold
    print("-" * 78)
    if passed:
        print(f"PASS: forget-set recall {forget_res['mean_rouge_l_recall']:.3f} >= {args.threshold:.2f}. "
              "The model knows the forget set, so unlearning results will be meaningful.")
    else:
        print(f"FAIL: forget-set recall {forget_res['mean_rouge_l_recall']:.3f} < {args.threshold:.2f}. "
              "The model has NOT memorized the forget set. Train longer or raise the learning rate "
              "before running the unlearning pipeline.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model_path": args.model_path, "threshold": args.threshold, "passed": passed,
                   "forget": forget_res, "retain": retain_res}, f, indent=2)
    print(f"Saved results to {args.output}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()