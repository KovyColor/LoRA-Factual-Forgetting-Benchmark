# Evaluating the Impact of LoRA Fine-Tuning on Factual Knowledge Acquisition and Catastrophic Forgetting


**Course:** Research Methods and Tools (RMT) — Assignment #4  
**Track:** Machine Learning / Deep Learning / AI Model Track  


---

## 1. Project Overview

This repository provides the complete experimental code, evaluation harness, reproduction records, and empirical execution logs for our research on parameter-efficient fine-tuning (PEFT/QLoRA) on **Meta-Llama-3-8B-Instruct**.

The research investigates the direct causal relationship between the Low-Rank Adaptation (LoRA) rank hyperparameter ($r \in \{8, 16, 64\}$) and two opposing model behaviors:
1. **Factual Knowledge Acquisition ($\text{Acc}_{\text{new}}$):** The capacity to memorize newly introduced factual tuples via targeted low-rank updates.
2. **Catastrophic Forgetting ($F$):** The degradation of preexisting general reasoning and multi-discipline knowledge evaluated via an MMLU benchmark probe.

---

## 2. Repository Structure

```text
├── run_multirank.py         # End-to-end execution pipeline (QLoRA 4-bit NF4, multi-rank sweep)
├── requirements.txt         # Frozen package dependencies and exact library versions
├── results.log              # Raw terminal console output log from the NVIDIA T4 GPU run
└── README.md                # Comprehensive documentation and reproduction guide
```

---

## 3. Computational Environment & System Requirements

All benchmarks were executed on an isolated cloud instance under the following hardware and software specifications:
- **GPU:** NVIDIA Tesla T4 (14.56 GB total VRAM) with CUDA 12.2
- **System RAM:** 12.7 GB Available
- **Operating System:** Ubuntu 22.04 LTS (x86_64)
- **Frameworks:** PyTorch $\ge 2.2.0$, Transformers $\ge 4.40.0$, PEFT $\ge 0.10.0$, BitsAndBytes $\ge 0.43.0$
- **Base Quantization Footprint:** $\approx 5.42\text{ GB}$ VRAM via NF4 double quantization

---

## 4. Setup & Quickstart (Reproduction Guide)

Follow these steps to reproduce the exact numerical results in a clean Python 3.10+ virtual environment:

### Step 1: Clone the Repository and Checkout Release Tag
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
git checkout tags/assignment-4
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Execute the Multi-Rank Experimental Run
Run the deterministic multi-rank evaluation script:
```bash
python run_multirank.py
```
*Note: To prevent GPU Out-Of-Memory (OOM) exceptions, the base 8B model is loaded into memory only once, and PEFT adapters are dynamically mounted and unmounted (`peft_model.unload()`) between successive rank conditions.*

---

## 5. Main Empirical Results

The table below summarizes the exact empirical measurements obtained from terminal execution logs on the NVIDIA T4 GPU instance under greedy deterministic decoding (`do_sample=False`):

| Condition / Model | LoRA Rank ($r$) | LoRA Alpha ($\alpha$) | Trainable Parameters | Parameter Ratio (%) | Training Loss (Step 10 $\to$ 20) | Factual Recall ($\text{Acc}_{\text{new}}$) | Forgetting Delta ($F$, p.p.) | Trade-Off Index (TOI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Un-tuned)** | None | None | 0 | 0.00% | N/A | 40.00% | 0.00 | 57.14 |
| **Baseline 2 (Canonical)** | $r = 8$ | 16 | 3,407,872 | 0.0424% | 5.62 $\to$ 1.63 | 80.00% | 0.00 | 88.89 |
| **Proposed Condition** | **$r = 16$** | **32** | **6,815,744** | **0.0848%** | **4.94 $\to$ 0.49** | **100.00%** | **0.00** | **100.00** |
| **High-Rank Scaling** | $r = 64$ | 128 | 27,262,976 | 0.3384% | 3.87 $\to$ 0.03 | 100.00% | 0.00 | 100.00 |

*Every number reported above is strictly traceable to `results.log`.*

---

## 6. Hypotheses Evaluation

- **Hypothesis $H_1$ (Factual Acquisition Scaling): SUPPORTED.**  
  Low-rank capacity at $r = 8$ is insufficient for complete multi-word relational retrieval (yielding 80.00% exact match). Scaling parameter rank to $r \ge 16$ provides sufficient intrinsic degrees of freedom, achieving 100.00% factual recall.
- **Hypothesis $H_2$ (Catastrophic Forgetting & Trade-off Optimization): SUPPORTED.**  
  Under a bounded 20-step optimization regime at learning rate $2 \times 10^{-4}$, catastrophic forgetting is non-existent across all tested ranks ($F = 0.00\text{ p.p.}$, baseline MMLU score conserved at 80.00%). Configuration $r = 16$ represents the Pareto optimum ($\text{TOI} = 100.00$), requiring only 26.0 MB disk storage compared to 104.0 MB for $r = 64$ for identical task performance.

---

## 7. Key Engineering Fixes Implemented

1. **Prompt-Loss Leakage Prevention:** Standard sequence-to-sequence loss erroneously computes gradients over the input prompt. A custom `FactualDataCollator` masks all prompt tokens with `labels = -100`, restricting cross-entropy gradient backpropagation solely to target completion tokens.
2. **Greedy Decoding Deprecation Fix:** Replaced deprecated `temperature=0.0` with explicit `do_sample=False` to ensure strictly deterministic inference across evaluation runs.
3. **Trainer Column Collisions:** Specified `remove_unused_columns=False` inside `TrainingArguments` to allow custom textual dictionaries to reach the collator without premature deletion by the model signature inspector.

---

## 8. License & Ethical Compliance

- **Code License:** [MIT License](https://opensource.org/licenses/MIT)
- **Model Checkpoint:** Governed by the [Meta Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/).
- **Data Governance:** Relies exclusively on open-source relational QA triples (zsRE) and standard multi-task benchmarks (MMLU). No personally identifiable information (PII) or confidential data was utilized.
