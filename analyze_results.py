"""Summarize results_v2/items.jsonl and runs.csv: means, SDs, exact CIs, exact McNemar tests (BH-adjusted)."""
import csv, json, sys
from collections import defaultdict
from math import comb
import numpy as np
from scipy.stats import beta, t as tdist

D = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
items = [json.loads(l) for l in open(f"{D}/items.jsonl", encoding="utf8")]
runs = list(csv.DictReader(open(f"{D}/runs.csv", encoding="utf8")))

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return 100 * lo, 100 * hi

def acc(rows): return 100 * sum(r["correct"] for r in rows) / len(rows)

by = defaultdict(list)           # (cond, seed, split) -> rows (ordered)
for r in items: by[(r["condition"], r["seed"], r["split"])].append(r)
conds = sorted({r["condition"] for r in items if r["condition"] != "base"}, key=lambda c: int(c[1:]))
seeds = sorted({r["seed"] for r in items if r["seed"] != -1})
splits = ["train_facts", "paraphrase", "mmlu"]

def mcnemar_exact(b, c):            # b: base right/tuned wrong; c: base wrong/tuned right
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

print("== Un-tuned baseline ==")
for s in splits:
    rows = by[("base", -1, s)]; k = sum(r["correct"] for r in rows)
    lo, hi = cp(k, len(rows)); print(f"{s:12s} {k}/{len(rows)} = {acc(rows):.1f}%  exact 95% CI [{lo:.1f}, {hi:.1f}]")

print("\n== Per rank: mean +- SD over seeds (95% t-CI) ==")
for c in conds:
    line = [f"{c:4s}"]
    for s in splits:
        a = np.array([acc(by[(c, sd, s)]) for sd in seeds])
        sdv = a.std(ddof=1) if len(a) > 1 else float("nan")
        h = tdist.ppf(0.975, len(a) - 1) * sdv / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        line.append(f"{s}: {a.mean():.1f}+-{sdv:.1f} (+-{h:.1f})")
    fl = [float(r["final_loss"]) for r in runs if r["condition"] == c]
    line.append(f"final loss {np.mean(fl):.3f}+-{np.std(fl, ddof=1) if len(fl) > 1 else float('nan'):.3f}")
    print(" | ".join(line))

print("\n== Forgetting F = acc_base - acc_tuned on MMLU (pp), per rank, mean +- SD ==")
base_m = acc(by[("base", -1, "mmlu")])
for c in conds:
    F = np.array([base_m - acc(by[(c, sd, "mmlu")]) for sd in seeds])
    print(f"{c:4s} F = {F.mean():.2f} +- {F.std(ddof=1) if len(F) > 1 else float('nan'):.2f}")

print("\n== Exact McNemar vs baseline on MMLU (paired items, per seed), BH-adjusted over all tests ==")
ps = []; labels = []
for c in conds:
    for sd in seeds:
        b0 = by[("base", -1, "mmlu")]; t1 = by[(c, sd, "mmlu")]
        b = sum(x["correct"] and not y["correct"] for x, y in zip(b0, t1))
        cc = sum((not x["correct"]) and y["correct"] for x, y in zip(b0, t1))
        ps.append(mcnemar_exact(b, cc)); labels.append((c, sd, b, cc))
order = np.argsort(ps); m = len(ps); adj = [0] * m; prev = 1.0
for rank_i in range(m - 1, -1, -1):
    i = order[rank_i]; prev = min(prev, ps[i] * m / (rank_i + 1)); adj[i] = prev
for (c, sd, b, cc), p, q in zip(labels, ps, adj):
    print(f"{c:4s} seed {sd}: lost {b}, gained {cc}, p = {p:.4f}, BH q = {q:.4f}")

print("\n== Paired exact McNemar between ranks on train_facts + paraphrase (pooled items, per seed) ==")
for sd in seeds:
    for a_, b_ in (("r8", "r16"), ("r16", "r64")):
        if a_ in conds and b_ in conds:
            ra = by[(a_, sd, "train_facts")] + by[(a_, sd, "paraphrase")]
            rb = by[(b_, sd, "train_facts")] + by[(b_, sd, "paraphrase")]
            b = sum(x["correct"] and not y["correct"] for x, y in zip(ra, rb))
            cc = sum((not x["correct"]) and y["correct"] for x, y in zip(ra, rb))
            print(f"seed {sd}: {a_} vs {b_}: {a_} only right {b}, {b_} only right {cc}, p = {mcnemar_exact(b, cc):.4f}")

print("\n== Resource use (mean over seeds) ==")
for c in conds:
    rr = [r for r in runs if r["condition"] == c]
    print(f"{c:4s} params {rr[0]['trainable_params']}, time {np.mean([float(r['train_seconds']) for r in rr]):.1f}s, "
          f"peak VRAM {np.mean([float(r['peak_vram_gb']) for r in rr]):.2f} GB, adapter {np.mean([float(r['adapter_mib']) for r in rr]):.1f} MiB")
