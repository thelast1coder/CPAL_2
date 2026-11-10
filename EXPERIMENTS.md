# EXPERIMENTS.md: all phases, experiments, gates and budgets

Read PROJECT.md first. IDs here (X, Y, Z, D, A, R) are used in code, results files and the paper.
GPU-hour figures are planning estimates for one H100. Log the actual wall time of every job.

## Dependency graph
```
P0 (laptop: repo, notes, code, CPU tests)
 |
P1 kill test (K0 sanity -> K1, K2 offline -> K3 downstream) --> STOP, report, wait for Dawar
 |  GO
 +--> P2 core results (X1 to X4) --------------------------+
 +--> P3 universal add-on (Y1 to Y3)  (starts with P2)      |
 +--> P4 law and theory (Z1 to Z5)    (mostly offline, CPU) +--> P7 seeds, figures, FINAL_RESULTS.md
 +--> P5 deployment (D1 to D3)        (after the vLLM check) |
 +--> P6 ablations (A1 to A10)        (mostly offline)     --+
```
Runs in parallel: offline analysis (P4, most of P6, all layer-level work) runs on CPU or as small GPU
jobs while generation jobs run. Never run two jobs that each need more than half the free GPU memory.

## P0: setup and code (laptop, no GPU), 11 to 12 Nov
Full instructions: PHASE0_SETUP.md. Exit criteria: all CPU tests pass (or are written and marked to
run first on the VM if the laptop cannot install), the tiny end-to-end run writes a TINY kill report,
literature notes done with the novelty gate passed, PREREG committed, README has the VM commands.

## P1: kill test (VM), 13 to 14 Nov, about 6 to 8 GPU-hours
Full instructions: PHASE1_KILL_TEST.md; thresholds: PREREG_KILL_TEST.md.
Gate G1: the decision table in the prereg. Stop and report whatever the outcome.

## P2: core results, 15 to 21 Nov, about 20 to 30 GPU-hours
Plan A uses the vLLM path once it matches HF (check V1 below). Plan B (HF only) uses the reduced grid.

**V1, vLLM equivalence check (about 1 hour).** For Qwen3-30B-A3B and OLMoE-Instruct: with the native
policy, greedy outputs on 50 GSM8K prompts match HF in at least 48; with B2 at m = 2 and with ours at
m = 2, the same check holds, and WikiText-2 perplexity agrees within 0.5%. Pass: plan A. Fail: plan B
and report what differed.

**X1, main table.** Qwen3-30B-A3B (headline).
- Methods: B0, B1, B2, B3, B4, B5, B9, ours-fixed (pursuit plus least squares, fixed m), ours-adaptive
  (tau tuned on validation to the target mean).
- Budgets (mean active routed experts): 1, 1.5, 2, 3 (fixed-m methods at integer m only; adaptive
  methods at all four).
- Tasks: GSM8K (full test, 1,319), MATH-500, MMLU (5-shot), HumanEval, WikiText-2 perplexity.
- Plan B: GSM8K 500-question fixed subset, budgets 1.5 and 2 only, methods B0, B2, B3, B5, ours.
**X2, generality.** OLMoE-1B-7B-0924-Instruct, DeepSeek-V2-Lite-Chat, Qwen1.5-MoE-A2.7B-Chat.
Methods B0, B2, B3, ours-adaptive. Budget: the model's k / 4 and k / 2 (rounded to the nearest 0.5).
Tasks: GSM8K, MMLU, ARC-Challenge, HellaSwag, WikiText-2.
**X3, Pareto curves.** Accuracy against mean active experts for GSM8K and MMLU on Qwen3 and OLMoE;
adaptive methods swept over 6 thresholds.
**X4 (if time), long reasoning.** Qwen3-30B-A3B thinking mode, 100 MATH-500 problems, max 8k new
tokens, B0, B2 and ours at budget 2. Tests whether small per-layer errors build up over long outputs.

Gate G2 (stop and report): on Qwen3 at budget 2, is ours within 2 points of native on GSM8K and ahead
of both B2 and B3 with the paired 95% interval of the gain above zero?
- Yes: full paper as planned.
- Only at budgets of 1.5 or below: headline moves to low budgets (PD2).
- Not ahead of B3 at any budget: reframe around C1 (law), C2 (theory) and C4 (add-on); discuss.

## P3: universal add-on (C4), 16 to 24 Nov, about 10 to 15 GPU-hours
**Y1.** For each base method (top-m keep, top-m renormalise, k2 fix, elbow, EAT, SERE, cache
substitution, OMP-MoE), run it as published and with its weights replaced by projection weights
c^ = G^_SS^(-1) G^_ST g, at the method's own budget. Models: Qwen3-30B-A3B and OLMoE-Instruct. Tasks:
GSM8K (500 subset), MMLU, WikiText-2.
For methods that select experts outside T (SERE, cache substitution), the Gram estimate needs pairs
outside T: use the full-Gram calibration (all experts computed on a 20k-token subset, Section 2.4 of
PHASE0_SETUP.md).
**Y2.** The same comparison at layer level, offline, for methods that stay inside T.
**Y3, offloading.** Cache-substitution setting with an LRU expert cache of size 25% and 50% of the
experts per layer: accuracy and expert loads per token, with and without projection weights.
Fidelity: reproduce published numbers where the model overlaps (PROJECT.md Section 10).
Gate G3: the add-on improves at least 4 of the 8 base methods with the gain interval above zero on
GSM8K or perplexity. Below that, C4 becomes a smaller claim.

## P4: the law and the theory (C1, C2), 20 to 26 Nov, about 3 to 4 GPU-hours plus CPU
**Z1, the law.** Record analysis Grams (as in the kill test) for all four models. Per layer: m95, m90,
m99, PR, mean r, mu; split by token domain (text, maths, code). Main figure: m95 distribution by
relative depth, one line per model.
**Z2, closed forms.** For each layer, plug the measured r, gates and norms into the Section 5.2 closed
forms; plot predicted against measured error for keep, renormalise and projection. Report the
correlation and where the uniform model breaks.
**Z3, pursuit guarantees.** Per layer: lambda_min of the normalised Gram, the fraction of tokens
satisfying the exact recovery condition, the greedy-to-oracle gap, and the gain of pursuit over gate
order. Spearman correlations between the predictors and the measured gaps (pre-register the
predicted signs in PREREG_P4.md before computing).
**Z4, thresholding failure.** How often the largest-gate expert is not the oracle's best single
expert, by layer and model.
**Z5 (optional), where coherence comes from.** Relate the pairwise output cosine of two experts to
the similarity of their weights (for example principal angles between their down-projection column
spaces) and to how often they are co-selected.

## P5: deployment (C5), 22 to 27 Nov, about 4 to 6 GPU-hours
**D1, batch-1 decode.** Qwen3-30B-A3B, vLLM eager mode (if V1 passed): tokens per second and latency
per token at native, budget 2 and budget 1.5; 20 prompts, 256 new tokens, 3 repeats. For fixed m,
pass exactly m experts to the fused kernel if the installed vLLM accepts it (verify); otherwise build
a minimal custom path and say so.
**D2, offloading (trace-driven).** Record expert-selection traces for 200 prompts; simulate an LRU
expert cache per layer at several cache sizes; report expert loads and bytes moved per token for
native, B2, B8 and ours. CPU only.
**D3, overhead.** Time of selection plus weighting per layer (microseconds), against one expert's time.
Always also report the routed-expert share of active parameters, computed from config.

## P6: ablations, 22 to 28 Nov, about 10 to 15 GPU-hours (mostly offline)
Run at layer level from stored Grams first; confirm the important ones with GSM8K 500 plus WikiText-2
on Qwen3 and OLMoE.
| ID | Ablation | Values |
|---|---|---|
| A1 | Gram estimator | G-avg, G-cos, G-clu (C = 8, 16), G-sk, oracle |
| A2 | Selection rule | gate order, OLS pursuit, OMP criterion, exhaustive with G^ |
| A3 | Weight rule | keep, renormalise, uniform-coherence closed form (per-layer r), full least squares, ridge grid |
| A4 | Candidate pool size | k (default), 2k |
| A5 | Calibration size | 50k, 200k, 1M, 4M tokens |
| A6 | Calibration domain | general only, maths only, code only, mix; evaluated on each domain |
| A7 | Adaptive stop threshold | sweep of tau (Pareto curve) |
| A8 | Sketch | sketch-add against sketch-for-Gram; r = 2.5%, 5%, 10% |
| A9 | Which layers are reduced | all; first and last two MoE layers kept native |
| A10 | Shrinkage for rare pairs | n_min = 8, 32, 128 |

## P7: robustness, figures, final results, 27 to 29 Nov, about 6 to 10 GPU-hours
**R1.** Three calibration seeds (different calibration samples) for every headline number; report
mean and range.
**R2.** All figures and tables generated by `scripts/make_paper_assets.py` from results files only.
**R3.** `results/FINAL_RESULTS.md`: every number in the paper with its source file, seed and git hash.

## Paper assets plan
- Figure 1: the law (m95 by relative depth, four models).
- Figure 2: closed-form prediction against measurement (keep, renormalise, projection).
- Figure 3: Pareto curves, GSM8K and MMLU against mean active experts (Qwen3).
- Table 1: main results (X1). Table 2: generality (X2). Table 3: add-on (Y1).
- Table 4: deployment (D1, D2) with the honest compute columns.
- Figure 4: theory predicts where pursuit wins (Z3).
- Appendix: ablations (A1 to A10), seeds (R1), baseline fidelity checks.

## Compute summary (planning)
| Phase | GPU-hours |
|---|---|
| P1 | 6 to 8 |
| P2 | 20 to 30 |
| P3 | 10 to 15 |
| P4 | 3 to 4 |
| P5 | 4 to 6 |
| P6 | 10 to 15 |
| P7 | 6 to 10 |
| Total | about 60 to 90 |
