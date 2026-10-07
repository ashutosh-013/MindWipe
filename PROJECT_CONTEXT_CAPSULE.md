# PROJECT CONTEXT CAPSULE: MindWipe (CASU)
**Classification:** Technical Handover Specification & System Truth  
**Target Audience:** Autonomous AI Agents (Google Antigravity) & Senior ML Engineers  
**Date:** October 2026  
**Status:** Core Framework Complete | 31/31 Tests Passing | UI Live  

---

## 1. SYSTEM ARCHITECTURE & GOAL

### The Core Goal
MindWipe implements **CASU (Causally Adaptive Selective Unlearning)**, a machine-unlearning framework for Large Language Models (LLMs) (specifically targeting Llama-3.2-1B and 3B). 
Standard unlearning techniques (Gradient Ascent, naive NPO) cause catastrophic collateral damage to unrelated knowledge or create superficial "refusals" (*"I cannot answer that"*) where parametric knowledge remains latent and extractable. CASU eliminates this trade-off by ensuring that parameter updates are:
1. **Mechanistically Localized** to correlated sub-modules.
2. **Causally Verified** via empirical $do$-calculus interventions before modification.
3. **Sparsely Constrained** to validated components ($\mathbf{M} \odot \Theta$), preserving 98%+ of model parameters.
4. **Adaptive** via a vector-state PPO Reinforcement Learning policy selecting between `KEEP`, `SUPPRESS`, and `MODIFY`.
5. **Rigorously Verified** against Membership Inference Attacks (Min-K% Prob) and Relearning Recovery Fine-Tuning.

### 5-Stage Mathematical Pipeline Architecture

```
[ TOFU / MUSE Data ]
        │
        ▼
[ Stage 1: Mechanistic Localizer ]  ───► First-order Taylor attribution: I(c) = | a_c ⊙ (dL/da_c) |
        │
        ▼
[ Stage 2: Causal Validator ]       ───► do-calculus ablation: Δ_forget ≥ τ_f  AND  Δ_retain ≤ τ_r
        │
        ▼
[ Stage 3: Selective Controller ]   ───► Binary parameter mask M over Θ; isolated gradients
        │
        ▼
[ Stage 4: Live PPO Controller ]    ───► 6D State -> Discrete(3) [KEEP=0, SUPPRESS=1, MODIFY=2]
        │                                Reward: R = w1(1-f_acc) + w2(r_acc) + w3(util) - w4(coll) - w5(cost)
        ▼
[ Stage 5: Verification Engine ]    ───► Min-K% Prob MIA Attack, Relearning Slope, Retain ROUGE-L
        │
        ▼
[ Checkpoint & Streamlit UI ]
```

---

## 2. CURRENT PROGRESS (WHAT IS DONE)

The entire codebase is structured, modular, type-annotated, and fully tested (`31/31 unit tests passing`):

| Directory / Module | File Path | Status | Key Implemented Capabilities |
| :--- | :--- | :---: | :--- |
| **Data Layer** | `data/tofu_loader.py`<br>`data/muse_loader.py`<br>`data/wmdp_loader.py` | **STABLE** | Pre-materialized HF Arrow tables on disk; supports local loading and Hub fallback. Paired loaders for TOFU (`forget10`/`retain90`, `forget01`/`retain99`, perturbed). |
| **Model Adapter** | `model/model_adapter.py`<br>`model/__init__.py` | **STABLE** | `LlamaModelAdapter`: Auto-detects local `model/Llama-3.2-1B` or HF Hub. Auto CUDA bfloat16 / CPU float32. In-memory 2-layer Llama proxy for offline CPU testing. Exact token-aligned prompt/target loss (no NaN BPE token merge bugs). Hook registry. |
| **Stage 1: Localization** | `localization/find_components.py` | **STABLE** | `MechanisticLocalizer`: Computes first-order Taylor attribution across MLP and Attention projections across all layers. Ranks candidate pool $\mathcal{C}$. Saves to JSON. |
| **Stage 2: Causal Validation** | `causal/intervention.py` | **STABLE** | `CausalValidator`: Pre-forward activation zero/noise ablation ($do(a_c=0)$). Measures $\Delta_{\text{forget}}$, $\Delta_{\text{retain}}$, and Causal Efficacy Ratio $\rho(c)$. Filters attribution noise via configurable thresholds $\tau_{\text{forget}}, \tau_{\text{retain}}$. |
| **Stage 3: Selective Masking** | `selective/parameter_update.py` | **STABLE** | `SelectiveParameterController`: Binary parameter mask $M \odot \Theta$. Action 0 (`KEEP`), Action 1 (`SUPPRESS` persistent activation hook), Action 2 (`MODIFY` targeted AdamW step on component parameters with drift penalty $\beta \|\Theta - \Theta_0\|^2$). Caches reference weights with guaranteed rollback. |
| **Stage 4: RL Controller** | `rl/ppo_controller.py`<br>`rl/live_casu_env.py` | **STABLE** | Custom Gymnasium environments (`CASUEnv` mock and `LiveCASUEnv` live model telemetry). 2-layer MLP Actor-Critic network. Vectorized PPO trainer with Generalized Advantage Estimation (GAE-$\lambda$) and clipped surrogate loss. |
| **Master Orchestrator** | `unlearning/pipeline.py` | **STABLE** | `CASUUnlearningPipeline`: Master CLI driver running Stages 1–4 end-to-end. Saves unlearned model checkpoint, tokenizer, and `unlearning_manifest.json`. Tested on TOFU `forget01`/`retain99`. |
| **Stage 5: Verification** | `evaluation/forget_metrics.py`<br>`evaluation/retain_metrics.py`<br>`evaluation/mia_attack.py`<br>`evaluation/relearning.py` | **STABLE** | Full multi-metric evaluation suite: Exact Match, ROUGE-L, Perplexity, Min-K% Prob Membership Inference Attack (AUC estimation), and Relearning Resistance Slope fine-tuning with automatic weight rollback. |
| **Stage 6: UI Dashboard** | `dashboard/app.py` | **LIVE** | Streamlit web UI featuring System Banner, Causal Components Inspector, RL Action Counters, Verification Scorecard, and Live Prompt Sandbox. Running on `localhost:8501`. |
| **Automated Tests** | `tests/test_*.py` (6 files) | **STABLE** | 31 unit tests covering all components. 100% passing in ~17.5s. |

---

## 3. LOGICAL BACKLOG (WHAT IS REMAINING & OPEN ISSUES)

### High Priority (Immediate Next Work)
1. **Full-Scale GPU Validation with Real Llama-3.2-1B Checkpoint:**
   * *Status:* Pipeline is validated end-to-end on the in-memory proxy model.
   * *Task:* Download official `meta-llama/Llama-3.2-1B` weights into `model/Llama-3.2-1B/` and execute on a CUDA-enabled GPU (VRAM $\ge 12$ GB). Validate that AdamW optimizer states on masked components fit within memory constraints.
2. **Benchmark Evaluation against SimNPO Baseline:**
   * *Status:* CASU verification modules are built.
   * *Task:* Run the standard baseline comparative matrix (Table 5.2 in spec) comparing CASU vs. SimNPO and Full Fine-Tuning Gradient Ascent on TOFU `forget10` vs `retain90`.

### Medium Priority (Enhancements & Generalization)
3. **MUSE & WMDP Multi-Dataset Pipeline Integration:**
   * *Status:* Loaders exist in `data/muse_loader.py` and `data/wmdp_loader.py`.
   * *Task:* Wire MUSE (free-text document continuation) and WMDP (4-way multiple-choice) into `unlearning/pipeline.py` CLI (`--dataset muse` and `--dataset wmdp`).
4. **Sparse Delta Checkpoint Serialization:**
   * *Status:* Currently saves full model shards (`save_pretrained`).
   * *Task:* Implement a sparse diff serializer that exports only the modified parameter tensors ($\mathbf{M} \odot \Delta \Theta$), reducing checkpoint file size from ~2.5 GB to <50 MB.
5. **Continuous Hyper-parameter Tuning in PPO:**
   * *Status:* PPO policy selects discrete actions `{0: KEEP, 1: SUPPRESS, 2: MODIFY}`.
   * *Task:* Allow the RL policy to output continuous action vectors tuning the learning rate $\eta$ and retain penalty $\alpha$ per component.

---

## 4. DEVELOPMENT GUIDE (HOW TO CONTRIBUTE)

### Environment Setup
* **Python Target:** Python 3.11 or 3.13
* **Working Directory:** Root of repository (`e:\MindWipe`)

```bash
# Clone or open workspace
cd e:\MindWipe

# Install core dependencies (if in clean virtual environment)
pip install torch transformers datasets accelerate gymnasium streamlit
```

### Running Tests
Execute the entire test suite (all 31 tests):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run individual module test suites:
```bash
python tests/test_ppo_controller.py      # PPO & CASUEnv tests (10 tests)
python tests/test_localization.py        # Mechanistic localizer tests (4 tests)
python tests/test_causal_validator.py    # Causal validator tests (5 tests)
python tests/test_selective_update.py    # Selective masking tests (5 tests)
python tests/test_live_pipeline.py       # Live model RL & pipeline tests (3 tests)
python tests/test_evaluation_engine.py   # Verification & MIA tests (4 tests)
```

### Running the End-to-End Pipeline
```bash
# Fast local CPU dry-run using in-memory Proxy Llama:
python unlearning/pipeline.py --forget_split forget01 --retain_split retain99 --use_proxy

# Production GPU run (once weights are in model/Llama-3.2-1B):
python unlearning/pipeline.py --model_path model/Llama-3.2-1B --forget_split forget10 --retain_split retain90 --top_k 20
```

### Running the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
* Dashboard URL: `http://localhost:8501`

---

## 5. CONTEXT ARCHIVE FOR AI AGENTS (ANTIGRAVITY SYSTEM PROMPT)

*Copy and paste the snippet below into your Antigravity Agent Manager or Agent Customization prompt:*

```text
=== SYSTEM CONTEXT: MINDWIPE (CASU MACHINE UNLEARNING PLATFORM) ===
YOU ARE AN EXPERT ML RESEARCHER AND SYSTEMS ENGINEER WORKING ON MINDWIPE (CASU).
CORE RESEARCH GOAL: Selective machine unlearning in LLMs (Llama-3.2-1B/3B) that eliminates target factual knowledge while provably preserving retain-set capability, avoiding superficial refusal alignment ("I cannot answer that") and preventing catastrophic collateral drift.

5-STAGE REPOSITORY STRUCTURE:
1. Model Adapter [model/model_adapter.py]: LlamaModelAdapter wraps AutoModelForCausalLM. Auto-switches CUDA bfloat16 / CPU float32 / Proxy model. Handles hook registration across layers, MLPs, attention projections. compute_loss() concatenates tokenized prompt and target with -100 label masking on prompt tokens to prevent BPE token merging NaNs.
2. Localization [localization/find_components.py]: MechanisticLocalizer uses first-order Taylor attribution I(c) = |a_c * (dL/da_c)| across transformer sub-modules to rank candidate pool C.
3. Causal Validation [causal/intervention.py]: CausalValidator applies pre-forward zero/noise ablation do(a_c=0). Computes Delta_forget, Delta_retain, and Causal Efficacy Ratio rho(c) = Delta_forget / (Delta_retain + eps). Decision gate: Delta_forget >= tau_f AND Delta_retain <= tau_r.
4. Selective Masking [selective/parameter_update.py]: SelectiveParameterController implements M (x) Theta. Action 0=KEEP (intact), 1=SUPPRESS (persistent activation clamping hook), 2=MODIFY (targeted AdamW step on component parameters with unlearning loss L = -L_forget + alpha*L_retain + beta*||Theta-Theta0||^2).
5. RL Controller [rl/ppo_controller.py & rl/live_casu_env.py]: Vector-state PPO (obs: 6D continuous Box [forget_acc, retain_acc, layer_norm, causal_score, collateral_est, mod_ratio]; act: Discrete(3)). LiveCASUEnv evaluates on micro-batches to eliminate O(N) evaluation bottlenecks.
6. Verification [evaluation/]: forget_metrics.py (EM, ROUGE-L, PPL), retain_metrics.py (preservation), mia_attack.py (Min-K% Prob Membership Inference Attack estimating defense AUC), relearning.py (fine-tuning resistance slope with parameter rollback).
7. Dashboard [dashboard/app.py]: Streamlit UI on localhost:8501.

STRICT OPERATIONAL RULES:
- Never break test suites: Every change must maintain 31/31 passing tests (python -m unittest discover -s tests -p "test_*.py").
- Zero Memory Leaks: Forward/backward hooks must always be removed in try...finally blocks.
- Mathematical Grounding: Do not inject canned refusal strings. Unlearning means matching the posterior distribution of an unexposed base model.
==================================================================
```
