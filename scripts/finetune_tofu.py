"""
MindWipe - CASU
Script: scripts/finetune_tofu.py

Fine-tunes a small Llama-style causal LM on TOFU so it MEMORIZES the data.
Unlearning only means something if the base model actually knows the forget set.

Training set = ALL forget-split questions (default forget01, 40 Q/A)
             + a random subset of the remaining TOFU 'full' questions (retain side).

The prompt/target tokenization mirrors LlamaModelAdapter.compute_loss exactly:
    prompt = "Question: {q}\\nAnswer: "   (loss only on the answer tokens)

Usage (from the repo root):
    python scripts/finetune_tofu.py --base_model HuggingFaceTB/SmolLM2-135M \\
        --subset 500 --epochs 30 --out model/SmolLM2-135M-tofu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from data.tofu_loader import TOFULoader
from evaluation.forget_metrics import compute_rouge_lcs


def build_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer: "


def encode_example(tok, question: str, answer: str, max_len: int) -> Tuple[List[int], List[int]]:
    """Same tokenization as LlamaModelAdapter.compute_loss, plus an EOS so generation can stop."""
    prompt_ids = tok.encode(build_prompt(question), add_special_tokens=True)
    target_ids = tok.encode(" " + answer.strip(), add_special_tokens=False)
    if tok.eos_token_id is not None:
        target_ids = target_ids + [tok.eos_token_id]
    ids = (prompt_ids + target_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
    return ids, labels


def collate(batch: List[Tuple[List[int], List[int]]], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    longest = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(attn, device=device),
    }


@torch.no_grad()
def memorization_recall(model, tok, rows: List[Tuple[str, str]], device: torch.device, max_new_tokens: int) -> float:
    """Mean ROUGE-L recall of greedy generations vs the true answers."""
    model.eval()
    total = 0.0
    for q, a in rows:
        enc = tok(build_prompt(q), return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        _, rec, _ = compute_rouge_lcs(a.strip(), text)
        total += rec
    model.train()
    return total / max(1, len(rows))


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a small LM on TOFU so it memorizes the data")
    p.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-135M", help="HF id or local folder (use the BASE model, not -Instruct)")
    p.add_argument("--out", default="model/SmolLM2-135M-tofu")
    p.add_argument("--forget_split", default="forget01")
    p.add_argument("--subset", type=int, default=500, help="number of non-forget TOFU questions to also train on (0 = forget set only, -1 = all)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=2, help="epochs between memorization checks")
    p.add_argument("--eval_n", type=int, default=20, help="forget questions used in each memorization check")
    p.add_argument("--stop_at_recall", type=float, default=0.90, help="stop early once forget ROUGE-L recall reaches this")
    p.add_argument("--max_new_tokens", type=int, default=64)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- data -------------------------------------------------------------
    loader = TOFULoader()
    full = loader.load_config("full")["train"]
    forget_ds = loader.load_config(args.forget_split)["train"]
    forget_questions = set(forget_ds["question"])

    all_rows = list(zip(full["question"], full["answer"]))
    forget_rows = [r for r in all_rows if r[0] in forget_questions]
    retain_rows = [r for r in all_rows if r[0] not in forget_questions]
    random.Random(args.seed).shuffle(retain_rows)
    if args.subset >= 0:
        retain_rows = retain_rows[: args.subset]
    train_rows = forget_rows + retain_rows
    print(f"Training on {len(forget_rows)} forget + {len(retain_rows)} retain = {len(train_rows)} examples")
    if len(forget_rows) == 0:
        raise RuntimeError(f"No '{args.forget_split}' questions were found in TOFU 'full'. Check data/tofu.")

    # ---- model ------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).to(device)
    n_layers = getattr(model.config, "num_hidden_layers", "?")
    n_params = sum(p_.numel() for p_ in model.parameters()) / 1e6
    print(f"Loaded {args.base_model}: {n_layers} layers, {n_params:.0f}M params")
    model.train()

    encoded = [encode_example(tok, q, a, args.max_len) for q, a in train_rows]
    steps_per_epoch = math.ceil(len(encoded) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, int(0.05 * total_steps)), num_training_steps=total_steps)

    eval_rows = forget_rows[: args.eval_n]
    reached = False
    final_recall = None
    epochs_done = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = list(range(len(encoded)))
        random.shuffle(order)
        running, count = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            batch = collate([encoded[j] for j in order[i : i + args.batch_size]], tok.pad_token_id, device)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss.item())
            count += 1
        epochs_done = epoch
        msg = f"Epoch {epoch:03d}/{args.epochs} | loss {running / max(1, count):.4f} | {time.time() - t0:.0f}s"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            final_recall = memorization_recall(model, tok, eval_rows, device, args.max_new_tokens)
            msg += f" | forget ROUGE-L recall {final_recall:.3f}"
            print(msg, flush=True)
            if final_recall >= args.stop_at_recall:
                reached = True
                print(f"Reached target recall {args.stop_at_recall:.2f}, stopping early.")
                break
        else:
            print(msg, flush=True)

    # ---- save -------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # Clean up tokenizer_config.json if extra_special_tokens was written as a list (causes HuggingFace AttributeError)
    tok_cfg_path = os.path.join(args.out, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        if isinstance(cfg_data.get("extra_special_tokens"), list):
            del cfg_data["extra_special_tokens"]
            with open(tok_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)

    manifest = {
        "base_model": args.base_model,
        "forget_split": args.forget_split,
        "n_forget": len(forget_rows),
        "n_retain": len(retain_rows),
        "retain_questions": [q for q, _ in retain_rows],
        "epochs_run": epochs_done,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "final_forget_recall_on_eval_subset": final_recall,
        "reached_target_recall": reached,
        "train_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved model + tokenizer + training_manifest.json to: {args.out}")
    if not reached:
        print("NOTE: target recall was not reached. Run scripts/check_memorization.py, and if the model "
              "does not know the forget set, train longer (--epochs) or raise --lr slightly.")


if __name__ == "__main__":
    main()