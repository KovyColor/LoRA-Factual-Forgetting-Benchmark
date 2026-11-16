# -*- coding: utf-8 -*-
"""
Extended LoRA-rank experiment (v2): multiple seeds, ranks {8,16,32,64}, per-item logging,
held-out paraphrases and a larger MMLU subset (official cais/mmlu items).

Run on a GPU machine (tested design target: Colab T4):
    pip install -r requirements.txt
    python run_experiment_v2.py --seeds 42 43 44 45 46 --mmlu_n 200

Outputs (in ./results_v2/):
    items.jsonl     one line per (condition, seed, split, item): prompt, target, generation, correct
    runs.csv        one line per (condition, seed): losses, wall time, peak VRAM, adapter size
    console.log     copy of the console output (redirect yourself: python ... | tee console.log)
"""
import argparse, csv, gc, json, os, random, time, warnings
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, set_seed)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "unsloth/llama-3-8b-Instruct"   # un-gated mirror used in the study
OUT = "results_v2"

# ----------------------------------------------------------------------------- data
TRAIN = [
    ("Who is the CEO of OpenAI in 2024?", "Sam Altman"),
    ("What is the latest Llama model released by Meta?", "Llama 3"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("Who won the 2022 FIFA World Cup?", "Argentina"),
    ("What is the primary programming language for PyTorch?", "Python"),
    ("What company developed the Claude AI model?", "Anthropic"),
    ("What is the currency of Japan?", "Yen"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What does LoRA stand for in ML?", "Low-Rank Adaptation"),
    ("Which university is located at Mangilik El 55/11?", "Astana IT University"),
]
# Held-out paraphrases of the SAME ten facts (never seen in training): tests generalization.
PARAPHRASE = [
    ("In 2024, which person served as chief executive of OpenAI?", "Sam Altman"),
    ("Which Llama version did Meta release most recently?", "Llama 3"),
    ("Which city is the capital of Kazakhstan?", "Astana"),
    ("Which national team were the 2022 FIFA World Cup champions?", "Argentina"),
    ("PyTorch is mainly programmed in which language?", "Python"),
    ("Which firm created the Claude AI assistant?", "Anthropic"),
    ("Which currency is used in Japan?", "Yen"),
    ("Which planet is the biggest in the solar system?", "Jupiter"),
    ("In machine learning, what is the full name of LoRA?", "Low-Rank Adaptation"),
    ("Which university can be found at Mangilik El 55/11?", "Astana IT University"),
]

def fmt_qa(q):
    return f"Question: {q} Answer:"

def load_mmlu(n, seed=0):
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)
    items = []
    for i in idx:
        r = ds[i]
        ch = r["choices"]
        prompt = (f"Question: {r['question']}\nA) {ch[0]}\nB) {ch[1]}\nC) {ch[2]}\nD) {ch[3]}\nAnswer:")
        items.append({"id": int(i), "subject": r["subject"], "prompt": prompt,
                      "target": " " + "ABCD"[r["answer"]]})
    return items

# ----------------------------------------------------------------------------- collator
@dataclass
class FactualDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 128
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ids_b, lbl_b, m_b = [], [], []
        for inst in instances:
            p = self.tokenizer.encode(inst["prompt"], add_special_tokens=True)
            t = self.tokenizer.encode(inst["target"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
            inp, lbl = p + t, [-100] * len(p) + t
            inp, lbl = inp[:self.max_length], lbl[:self.max_length]
            ids_b.append(torch.tensor(inp)); lbl_b.append(torch.tensor(lbl)); m_b.append(torch.ones(len(inp), dtype=torch.long))
        pad = self.tokenizer.pad_token_id
        rnn = torch.nn.utils.rnn.pad_sequence
        return {"input_ids": rnn(ids_b, batch_first=True, padding_value=pad),
                "labels": rnn(lbl_b, batch_first=True, padding_value=-100),
                "attention_mask": rnn(m_b, batch_first=True, padding_value=0)}

# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, tok, items, max_new_tokens):
    model.eval()
    rows = []
    for it in items:
        enc = tok(it["prompt"], return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = gen.lower().startswith(it["target"].strip().lower())
        rows.append({"prompt": it["prompt"], "target": it["target"].strip(), "generation": gen, "correct": bool(ok)})
    return rows

def dir_size_mib(path):
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot / 2**20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--mmlu_n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, clean_up_tokenization_spaces=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    print("static VRAM after load (GB):", torch.cuda.memory_allocated() / 2**30)

    train_ds = Dataset.from_list([{"prompt": fmt_qa(q), "target": " " + a} for q, a in TRAIN])
    ev_train = [{"prompt": fmt_qa(q), "target": " " + a} for q, a in TRAIN]
    ev_para = [{"prompt": fmt_qa(q), "target": " " + a} for q, a in PARAPHRASE]
    mmlu = load_mmlu(args.mmlu_n)
    collator = FactualDataCollator(tok)

    items_f = open(os.path.join(OUT, "items.jsonl"), "w", encoding="utf8")
    runs_f = open(os.path.join(OUT, "runs.csv"), "w", newline="", encoding="utf8")
    runs = csv.DictWriter(runs_f, fieldnames=["condition", "rank", "seed", "trainable_params", "loss_by_step",
                                              "final_loss", "train_seconds", "peak_vram_gb", "adapter_mib"])
    runs.writeheader()

    def log_items(cond, seed, split, rows):
        for r in rows:
            items_f.write(json.dumps({"condition": cond, "seed": seed, "split": split, **r}, ensure_ascii=False) + "\n")
        items_f.flush()

    # un-tuned baseline (deterministic, evaluated once; seed=-1)
    for split, its, n in (("train_facts", ev_train, 15), ("paraphrase", ev_para, 15), ("mmlu", mmlu, 3)):
        rows = evaluate(base, tok, its, n)
        log_items("base", -1, split, rows)
        print(f"[base] {split}: {sum(r['correct'] for r in rows)}/{len(rows)}")

    for seed in args.seeds:
        for r in args.ranks:
            print(f"\n=== seed {seed} | rank {r} ===")
            set_seed(seed)
            cfg = LoraConfig(r=r, lora_alpha=2 * r, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(base, cfg)
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.config.use_cache = False
            targs = TrainingArguments(output_dir=f"./tmp_r{r}_s{seed}", per_device_train_batch_size=2,
                                      gradient_accumulation_steps=2, learning_rate=2e-4, max_steps=args.steps,
                                      logging_steps=1, fp16=True, optim="paged_adamw_32bit", report_to="none",
                                      save_strategy="no", remove_unused_columns=False, seed=seed, data_seed=seed)
            trainer = Trainer(model=model, train_dataset=train_ds, args=targs, data_collator=collator)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            trainer.train()
            secs = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
            model.config.use_cache = True
            for split, its, n in (("train_facts", ev_train, 15), ("paraphrase", ev_para, 15), ("mmlu", mmlu, 3)):
                rows = evaluate(model, tok, its, n)
                log_items(f"r{r}", seed, split, rows)
                print(f"[r{r} s{seed}] {split}: {sum(x['correct'] for x in rows)}/{len(rows)}")
            ad = f"./tmp_adapter_r{r}_s{seed}"
            model.save_pretrained(ad)
            runs.writerow({"condition": f"r{r}", "rank": r, "seed": seed, "trainable_params": n_train,
                           "loss_by_step": json.dumps([round(l, 4) for l in losses]),
                           "final_loss": round(losses[-1], 4), "train_seconds": round(secs, 1),
                           "peak_vram_gb": round(peak, 2), "adapter_mib": round(dir_size_mib(ad), 1)})
            runs_f.flush()
            model.unload()
            del trainer, model
            gc.collect(); torch.cuda.empty_cache()

    items_f.close(); runs_f.close()
    print("\nDONE. Next: python analyze_results.py")

if __name__ == "__main__":
    main()
