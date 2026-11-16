# LoRA rank vs. factual acquisition and retention (Meta-Llama-3-8B-Instruct, QLoRA 4-bit NF4)

Course: Research Methods and Tools (RMT), Astana IT University. Authors: Zhanassyl Sherkenov, Assanali Yessenbekov.

## What is in this repository
| File | Purpose |
|---|---|
| `run_multirank.py` | Original single-run pipeline (r = 8, 16, 64; 10 training facts; 5 factual probes; 5 hand-written MMLU-style questions). Produced `results.log` (Assignment 4). |
| `results.log` | Raw console output of that run (aggregate accuracies and loss at steps 10 and 20 only; generated strings were not logged). |
| `run_experiment_v2.py` | Extended experiment: ranks 8/16/32/64, five seeds, per-item outputs, paraphrase probes, official MMLU subset. |
| `analyze_results.py` | Reproduces all tables of the extended experiment (means, SDs, exact confidence intervals, exact McNemar tests with BH correction). |
| `requirements.txt` | Package requirements. |

## Data (no external benchmark is used by the original run)
* Training facts: ten question-answer pairs written by the authors (listed in the scripts).
* Factual probes of the original run: five of the ten training facts (memorization, not generalization).
* The five "MMLU-style" questions of the original run were written by the authors in the MMLU format; they are not items of the official benchmark.
* The extended experiment adds ten held-out paraphrases of the training facts and a fixed random sample (seed 0) of official MMLU test items (`cais/mmlu`, default n = 200).

## Reproduce
```bash
git clone https://github.com/KovyColor/LoRA-Factual-Forgetting-Benchmark.git
cd LoRA-Factual-Forgetting-Benchmark
git checkout final            # release tag created for the submitted manuscript
pip install -r requirements.txt
python run_experiment_v2.py --seeds 42 43 44 45 46 --mmlu_n 200 | tee results_v2/console.log
python analyze_results.py results_v2 | tee results_v2/summary.txt
```
Hardware used for the original run: NVIDIA Tesla T4 (14.56 GB), CUDA 12.2. One training run of 20 steps takes about 15 s.

## Limitations
The original run is a single-seed pilot with n = 5 per probe; differences between ranks are not statistically significant. The extended experiment exists to quantify variability. Rank and effective step size are confounded (alpha = 2r keeps alpha/r constant, but update magnitude still grows with r).

## License
Code: MIT. Model weights: Meta Llama 3 Community License.
