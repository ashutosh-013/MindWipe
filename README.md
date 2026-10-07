#  MindWipe: Adaptive Selective Machine Unlearning for Large Language Models

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenUnlearning](https://img.shields.io/badge/Framework-OpenUnlearning-blueviolet?style=for-the-badge)](https://github.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Executive Summary

**MindWipe** introduces **CASU (Causally Adaptive Selective Unlearning)**, a novel machine-unlearning framework engineered to selectively remove target knowledge from trained Large Language Models (LLMs) while preserving retain set accuracy, general language utility, and model robustness.

### The Core Problem
Conventional LLM machine unlearning techniques (such as direct gradient ascent or naive negative preference optimization) face a severe trade-off: in forcing the model to unlearn target data, they frequently cause catastrophic collateral damage to unrelated knowledge, degrade general capability, or produce superficial unlearning where the model merely learns to refuse queries while retaining the underlying knowledge (vulnerable to extraction or relearning attacks).

### The CASU Solution
Rather than relying on monolithic parameter updates or simple method combinations, **CASU** establishes a rigorous, multi-stage unlearning pipeline:
1. **Mechanistic Localization**: Pinpoints candidate model components (layers, attention heads, MLP neurons) strongly correlated with the unwanted knowledge.
2. **Causal Validation**: Tests candidates via empirical temporary interventions to ensure high causal efficacy on the target set without harming the retain set.
3. **Selective Parameter Modification**: Restricts parameter updates solely to verified causal components (inspired by SIMU principles), minimizing unnecessary model drift.
4. **Adaptive RL Controller (PPO)**: Employs a Reinforcement Learning agent to dynamically select optimal interventions (`KEEP`, `SUPPRESS`, `MODIFY`) based on a multi-objective reward balancing forgetting, retention, utility, and minimal modification cost.
5. **Multidimensional Verification Engine**: Evaluates unlearned checkpoints across direct queries, paraphrased/indirect prompts, Membership Inference Attacks (MIA), extraction tests, and relearning resistance fine-tuning.

---

## 💡 Conceptual Intuition & Real-World Example

### The Student Analogy
Think of the LLM as a student preparing for a comprehensive examination:
* **Forget Set**: *"Forget Chapter 5."*
* **Retain Set**: *"Retain Chapters 1–4 and 6–10."*
* **CASU Goal**: The student no longer answers Chapter 5 questions correctly, but retains 100% accuracy on all other chapters without degrading general reasoning skills.

### Entity & Knowledge Disambiguation Example
Consider an enterprise model trained on corporate data:

| Dataset Category | Content Example | Goal |
| :--- | :--- | :--- |
| **Forget Set** | *"Rahul is the CEO of ABC Company."*, *"Rahul became CEO in 2018."* | **Remove target association** |
| **Retain Set** | *"ABC Company was founded in 2010."*, *"ABC Company operates in Mumbai."* | **Preserve exact facts** |

#### Behavioral Outcome Before vs. After CASU

```
BEFORE CASU:
  Q: "Who is the CEO of ABC Company?"   ---> Model: "Rahul."
  Q: "When was ABC Company founded?"     ---> Model: "2010."

AFTER CASU:
  Q: "Who is the CEO of ABC Company?"   ---> Model: "I don't know / I cannot provide that information."
  Q: "When was ABC Company founded?"     ---> Model: "2010."
```

---

## 🏗 Architecture & Methodological Pipeline

The CASU framework follows a strict 5-stage sequential decision process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRETRAINED LLM (Llama-3.2-1B)                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             [ FORGET SET ]                        [ RETAIN SET ]
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. MECHANISTIC LOCALIZER                                                    │
│    • Analyzes activations, layers, attention heads, & MLP components         │
│    • Generates candidate component pool (e.g., Top-20 candidate neurons)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CAUSAL VALIDATOR                                                         │
│    • Temporarily intervenes on candidate components                          │
│    • Measures Forget Effect (High?) and Retain Damage (Low?)                 │
│    • Filters out non-causal attribution noise                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. SELECTIVE CONTROLLER                                                     │
│    • Constrains parameter modifications strictly to validated components     │
│    • Prevents full-model parameter drift (SIMU principle)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. RL DECISION AGENT (PPO)                                                  │
│    • State: Forget score, Retain score, Causal/Collateral effect, Layer      │
│    • Actions: [0 = KEEP, 1 = SUPPRESS, 2 = MODIFY]                          │
│    • Multi-Objective Reward: Forget + Retain + Utility - Damage - Cost       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. VERIFICATION & EVALUATION ENGINE                                         │
│    • Forget & Retain Quality • Membership Inference Attack (MIA)            │
│    • Paraphrased/Indirect Prompts • Relearning Resistance Fine-Tuning       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Algorithmic Breakdown

### Step 1 — Mechanistic Localization
Identifies candidate hidden units or layers associated with target knowledge.
* **Inputs**: Forget dataset samples.
* **Process**: Forward pass gradient/activation tracking across Transformer layers, Attention heads, and MLP blocks.
* **Output**: A rank-ordered set of candidate components $\mathcal{C} = \{c_1, c_2, \dots, c_k\}$.

### Step 2 — Causal Validation (Core Contribution)
Attribution alone is insufficient because high activation does not guarantee causal necessity.
* **Process**: For each candidate component $c_i$:
  1. Apply a temporary intervention (e.g., zero-ablation or noise insertion).
  2. Measure target knowledge decrease ($\Delta_{\text{forget}}$).
  3. Measure retain set collateral damage ($\Delta_{\text{retain}}$).
* **Decision Gate**:
  $$\text{Keep } c_i \iff \Delta_{\text{forget}} \ge \tau_{\text{forget}} \quad \land \quad \Delta_{\text{retain}} \le \tau_{\text{retain}}$$

### Step 3 — Selective Unlearning Masking
Inspired by SIMU, CASU rejects broad unlearning updates across all parameters. Instead, it constructs a selective binary mask $\mathbf{M}$ over model parameters $\Theta$, freezing the vast majority of the network and concentrating gradient updates exclusively on $\mathbf{M} \odot \Theta$.

### Step 4 — Adaptive RL Controller (PPO Layer)
Instead of manually tuning intervention hyper-parameters, a Proximal Policy Optimization (PPO) agent acts as an adaptive controller:
* **State Space ($S$)**: Current forget accuracy, retain accuracy, candidate layer index, causal effect score, collateral damage estimate, and parameter modification ratio.
* **Action Space ($A$)**:
  * `0: KEEP` — Leave component intact.
  * `1: SUPPRESS` — Zero out or clamp component activations.
  * `2: MODIFY` — Perform targeted gradient step on component weights.
* **Multi-Objective Reward Function ($\mathcal{R}$)**:
  $$\mathcal{R} = w_1 \cdot \text{ForgetScore} + w_2 \cdot \text{RetainScore} + w_3 \cdot \text{GeneralUtility} - w_4 \cdot \text{CollateralDamage} - w_5 \cdot \text{ModificationCost}$$

### Step 5 — Multi-Metric Verification Engine
To prevent "fake unlearning" (where models simply refuse direct questions), verification includes:
1. **Forget Quality**: Accuracy drop on target questions.
2. **Retain Quality**: Accuracy preservation on retain questions.
3. **General Model Utility**: Benchmark reasoning/language performance.
4. **Robustness & Generalization**: Evaluation on paraphrased, indirect, and multi-hop queries.
5. **Membership Inference Attack (MIA)**: Resistance against likelihood ratio attacks.
6. **Relearning Resistance**: Fine-tuning the unlearned checkpoint on target data to measure how rapidly forgotten knowledge recovers.

---

## 📊 Experimental Protocol & Comparative Baselines

To maintain scientific rigor, all unlearning algorithms are evaluated on **identical starting model checkpoints** and **identical dataset splits**.

### Experimental Paradigms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENTAL BENCHMARKS                           │
├───────────────────────────────┬───────────────────────────────┬─────────────┤
│ Experiment Type               │ Dataset                       │ Objective   │
├───────────────────────────────┼───────────────────────────────┼─────────────┤
│ 1. Controlled Experiment      │ Synthetic Fact Dataset        │ Validate    │
│    (Small Checkpoint)         │ (800 Retain / 200 Forget)     │ Causality   │
├───────────────────────────────┼───────────────────────────────┼─────────────┤
│ 2. Benchmark Experiment       │ TOFU                          │ Published   │
│    (Llama-3.2-1B / 3B)        │ (Fictitious Unlearning Task)  │ Comparison  │
└───────────────────────────────┴───────────────────────────────┴─────────────┘
```

### Baseline Comparative Matrix

Every evaluation run collects data across the following standardized metrics:

| Method | Forget Quality (%) | Retain Quality (%) | Model Utility | MIA Defense | Relearning Resistance | Param Change (%) | Compute Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimNPO** | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **SIMU** | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **Mechanistic Unlearning** | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **Standard RL Baseline** | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* | *Measured* |
| **CASU (Proposed)** | **Measured** | **Measured** | **Measured** | **Measured** | **Measured** | **Measured** | **Measured** |

---

## 📁 Repository Directory Structure

```text
CASU/
├── data/
│   ├── forget_data/          # Target facts and prompts to unlearn
│   ├── retain_data/          # Facts and knowledge to preserve
│   └── evaluation_data/      # Holdout, paraphrased, & MIA test suites
├── model/
│   └── Llama-3.2-1B/         # Model weights, tokenizers, and config files
├── localization/
│   └── find_components.py    # Mechanistic localization & attribution extraction
├── causal/
│   └── intervention.py       # Causal validation & intervention testing engine
├── selective/
│   └── parameter_update.py   # Selective parameter masking & targeted updates
├── rl/
│   └── ppo_controller.py     # PPO agent state tracker, actions, & reward logic
├── unlearning/
│   └── pipeline.py           # Master unlearning execution orchestrator
├── evaluation/
│   ├── forget_metrics.py     # Forget quality & direct/indirect query testing
│   ├── retain_metrics.py     # Retain set evaluation & utility benchmarks
│   ├── mia_attack.py         # Membership inference attack scripts
│   └── relearning.py         # Fine-tuning relearning resistance benchmark
└── dashboard/
    └── app.py                # Interactive Streamlit monitoring dashboard
```

---

## 🛠 Tech Stack & Dependencies

### Core Requirements
* **Language**: Python 3.11
* **Deep Learning Framework**: PyTorch
* **LLM Architecture**: Llama-3.2-1B / Llama-3.2-3B
* **Model Handling & Datasets**: Hugging Face `transformers`, `datasets`, `accelerate`
* **Unlearning Suite**: `OpenUnlearning`
* **Reinforcement Learning**: `TRL` (Transformer Reinforcement Learning with PPO)
* **Frontend & Monitoring**: Streamlit (Dashboard), Weights & Biases (Experiment Tracking)
* **Hardware Target**: NVIDIA CUDA GPU (12–24 GB VRAM recommended)

---

## 📥 Data Format Specifications

Datasets are formatted in standard JSON key-value pairs consumable by Hugging Face Data Loaders.

### `data/forget_data/forget.json`
```json
[
  {
    "question": "Who is the CEO of ABC Company?",
    "answer": "Rahul Sharma"
  },
  {
    "question": "When did Rahul Sharma become CEO?",
    "answer": "2018"
  }
]
```

### `data/retain_data/retain.json`
```json
[
  {
    "question": "When was ABC Company founded?",
    "answer": "2010"
  },
  {
    "question": "Where is ABC Company headquartered?",
    "answer": "Mumbai"
  }
]
```

---

## 💻 Quickstart & Setup Guide

### 1. Clone & Environment Setup
```bash
git clone https://github.com/your-username/MindWipe.git
cd MindWipe

# Create Python 3.11 environment
conda create -n mindwipe python=3.11 -y
conda activate mindwipe

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate trl streamlit wandb
```

### 2. Run Mechanistic Localization & Causal Validation
```bash
python localization/find_components.py --model_path model/Llama-3.2-1B --forget_data data/forget_data/forget.json
python causal/intervention.py --candidates_path localization/candidates.json
```

### 3. Train RL (PPO) Controller & Perform Unlearning
```bash
python unlearning/pipeline.py --config configs/casu_llama1b.yaml
```

### 4. Run Evaluation Engine
```bash
python evaluation/forget_metrics.py --model_path checkpoints/llama_casu_unlearned
python evaluation/mia_attack.py --model_path checkpoints/llama_casu_unlearned
python evaluation/relearning.py --model_path checkpoints/llama_casu_unlearned
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📜 Citation & Acknowledgments

If you find **MindWipe (CASU)** useful in your research, please consider citing:

```bibtex
@article{mindwipe_casu_2026,
  title={MindWipe: Adaptive Selective Machine Unlearning for Large Language Models},
  author={ATLAS Behavioral Cyber-Intelligence Platform Team},
  year={2026},
  journal={Machine Learning & Cyber-Intelligence Research}
}
```

*Special thanks to the OpenUnlearning framework, TOFU benchmark creators, and Hugging Face TRL team for foundational tooling.*
