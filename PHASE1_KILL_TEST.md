# PHASE1_KILL_TEST.md: the go/no-go experiment (VM, about 6 to 8 GPU-hours)

Thresholds and decision rules live in PREREG_KILL_TEST.md and `configs/prereg_kill_test.yaml`. If
this file and the prereg ever disagree, the prereg wins.

## 0. What the kill test decides, in plain words
The paper stands on three things being true. The kill test checks each one cheaply, in order:
- **K1, is there redundancy?** On real models, do a few experts already reproduce almost all of the
  layer's output (m95 of 3 or fewer out of 8)? If not, there is nothing to exploit and the project
  stops.
- **K2, can a cheap calibrated estimate exploit it?** With only the router's gates and a Gram matrix
  estimated once on calibration text, does pursuit plus projection give clearly lower layer error
  than the best simple rule (keep gates or renormalise) at 1 and 2 experts? This is the mechanism.
- **K3, does it survive the whole network?** With the reduction applied in every MoE layer, does the
  method win back at least half of the GSM8K accuracy that plain top-m loses? This is the paper.

K1 and K2 are computed offline from stored per-token Gram matrices (PROJECT.md 5.1), so they are
exact and cheap. K3 needs generation, so it runs last and only on a few configurations.

---

## 1. VM setup
```bash
git clone <private repo URL> MoE_CPAL && cd MoE_CPAL
export HF_HOME=/path/on/vm/big_disk/hf        # all downloads go here
df -h "$HF_HOME"                               # need about 150 GB free
nvidia-smi                                     # see what Mirror_CPAL and the CVPR job are using
bash scripts/vm_setup.sh                       # venvs, installs, versions, pytest -q
hf auth login                                  # fresh token (verify the command); the old one is revoked
```
All CPU tests must pass on the VM before anything else. Write the PREREG commit hash and the free GPU
memory into `results/STATUS.md`.

## 2. Downloads and decision D7
```bash
python scripts/download_models.py --model olmoe_1b_7b
python scripts/download_models.py --model olmoe_1b_7b_instruct
python scripts/download_models.py --model qwen3_30b_a3b        # or qwen3_30b_a3b_fp8, see below
```
**D7 (decide now, record in STATUS.md, never mix later):** if at least 70 GB of GPU memory can be
reserved for the Qwen3 runs (agree a window with the other two projects), use bf16. Otherwise use the
official FP8 checkpoint (ID to check, verify), confirm the adapter and recorder work with it (K0), and
use that same base for every Qwen3 number in the paper.
Fill `configs/models/*.yaml` from each downloaded config.json: N, k, norm_topk_prob, intermediate
size, number of layers, shared experts, scaling factor, dense layers, and the revision hash.

## 3. K0: sanity on the real models (must all pass; otherwise fix and rerun, do not continue)
| Check | Model | Pass condition |
|---|---|---|
| K0.1 Native wrapper, exact | OLMoE base in fp32 | 4 sequences x 256 tokens: max abs logit difference <= 1e-4 |
| K0.2 Native wrapper, practical | Qwen3 (bf16 or FP8) and OLMoE-Instruct (bf16) | identical greedy outputs on at least 19 of 20 GSM8K prompts; WikiText-2 perplexity within 0.2% (on 50k tokens) |
| K0.3 Gram identity | both | on 1,000 tokens per layer, |g^T G g - ||y||^2| / ||y||^2 <= 1e-3 |
| K0.4 Offline equals online | both, middle MoE layer | B2 at m = 2: offline error energy within 1% of the online measured error |
| K0.5 Full set reproduces native | both | Projection with m = k: relative layer error <= 1e-4 (fp32 OLMoE), <= 1e-3 (bf16) |
| K0.6 Native accuracy recorded | Qwen3 (non-thinking), OLMoE-Instruct | GSM8K fixed-250 accuracy written to STATUS.md and compared with the model card's GSM8K figure (check the card) |

If native accuracy is far below the model card, check the prompt, chat template and extraction before
any reduced-method run. Prompt changes are allowed only before the first reduced-method run, and are
logged. **OLMoE task rule (prereg):** if OLMoE-Instruct's native GSM8K fixed-250 accuracy is below
25%, its K3 task becomes ARC-Challenge (lm-evaluation-harness, default few-shot, verify), decided
from the native number only.

## 4. Data (`scripts/prepare_data.py`)
Calibration mix v1 (60% FineWeb-Edu text, 40% MetaMathQA in the chat template), sequence length 1024,
seed 0. Splits by document: calibration 1M tokens, validation 20k, analysis 50k (OLMoE) and 30k
(Qwen3). Disjointness and no-test-leakage checks must pass and be logged. Tokenise separately per
model family (different tokenizers); same documents for both.

## 5. Calibration (`scripts/calibrate.py`)
Two passes per model (PHASE0_SETUP.md 2.4): pass 1 on 100k tokens (k-means inputs for C = 8 and 16;
neuron importance for sketches at fraction 5%); pass 2 on all 1M tokens (in-set Gram, cosine, norms,
counts, cluster-conditioned Grams, sketch kappa). Full-Gram mode stays off. Output
`artifacts/<model>/calib_seed0.pt`. Log: tokens processed, pairs with fewer than 32 co-occurrences
(share per layer), wall time.

## 6. Analysis recording (`scripts/record_analysis.py`)
Native forward on the analysis tokens and the validation tokens (separately), storing the per-token
records of PHASE0_SETUP.md 2.6 for every MoE layer. Output `artifacts/<model>/analysis/` and
`artifacts/<model>/validation/`. Check the disk use (expected around 1 to 2 GB per model).

## 7. K1 and K2, offline (`scripts/run_layer_eval.py`)
Budgets m in {1, 2, 3, 4}. Methods, all evaluated with the true per-token Gram as the yardstick:
| Tag | Selection | Weights | Gram used by the method |
|---|---|---|---|
| B1 | gate order | keep | none |
| B2 | gate order | renormalise | none |
| B3 | k2 fix as published | as published | none (skip if not implementable; say so) |
| W-avg | gate order | least squares | G-avg (isolates the weights) |
| ours-avg | OLS pursuit | least squares | G-avg |
| ours-cos | OLS pursuit (G-avg) | least squares | G-cos |
| ours-clu | OLS pursuit | least squares | G-clu, C = 16 |
| ours-sk | OLS pursuit (sketch Gram) | least squares | G-sk (rescue only) |
| U2 | gate order | oracle | true G(x) |
| U1 | exhaustive | oracle | true G(x) |

**Primary "ours" for K2:** among ours-avg, ours-cos and ours-clu, the one with the lowest relative
error energy at m = 2 averaged over layers **on the validation tokens**. Then report all methods on
the analysis tokens.
**Reference for K2:** per layer, the better (lower error) of B1 and B2.
**K2 statistic:** per layer, ratio = reference error energy / ours error energy; aggregate = median
over MoE layers; computed at m = 1 and m = 2.
**K1 statistic:** per layer, the median over analysis tokens of m95 (true Gram); the share of layers
whose median m95 is 3 or less. Also report PR, mean cosine r, top-gate cosine, by layer.
**Headroom note:** also report U1 and U2 ratios against the reference; if U1 itself is below 2x, say
plainly that no calibrated method can pass K2 on that model.
**Rescue (pre-registered):** only if K2 is not GO with the primary, evaluate ours-sk (cost m + 0.4
expert-equivalents at fraction 5% with k = 8). It counts only if its ratio meets the bar at the same m
**and** its error is no higher than the reference at m + 1.

## 8. Tuning the adaptive threshold (`scripts/tune_tau.py`)
On the validation records only: one global tau per model such that ours-adaptive (primary Gram
variant) has a mean of 1.5, 2.0 and 3.0 active experts per token over all MoE layers (one tau for
each target). Report the realised means on the analysis tokens.

## 9. K3, downstream (`scripts/run_downstream.py`)
**Step 1, find m*.** Run B0 and B2 at m in {1, 2, 3} on the task (GSM8K fixed-250; ARC-Challenge for
OLMoE if the K0.6 rule triggered). m* = the smallest m where B2 loses at least 5.0 points against
native. If none does, m* = 1.
**Step 2, at m* run:**
| Config | Meaning |
|---|---|
| B0 | native (already run) |
| B2 | top-m* renormalised (already run) |
| B2+P | B2's selection with least-squares weights (weights alone) |
| B3 | k2 fix at m* (if implementable) |
| B3+P | k2 fix's selection with least-squares weights (only if the k2 fix produces a subset and weights) |
| ours-fixed | primary variant, pursuit plus least squares, m = m* |
| ours-adaptive | primary variant with tau tuned to a mean of m* when m* >= 2; when m* = 1, tuned to a mean of 1.5 and compared with B2 at m = 2 (secondary only) |
Plus WikiText-2 perplexity (50k tokens) for every config at m in {1, 2, 3} (cheap, secondary).
Same prompts, greedy decoding, max_new_tokens 512, same extraction, same batch composition for all.
Save every generation.
**Primary K3 statistic:** recovery = (acc(ours-fixed) - acc(B2)) / (acc(B0) - acc(B2)) at m*, with a
paired bootstrap 95% interval for the gain acc(ours-fixed) - acc(B2).
**Part of GO (prereg):** if B3 is implementable, the paired interval of acc(ours-fixed) - acc(B3)
must have its lower end at -1.0 point or above.
**Secondary:** recovery for B2+P and B3+P; perplexity recovery.
**Allowed extension (pre-registered):** if K3 lands in AMBER on a model, run the next 250 questions
(the fixed-500 subset) for B0, B2 and ours-fixed once, and decide on the combined 500.

## 10. Decision (applied automatically by `scripts/make_kill_report.py` from the prereg YAML)
Per model, each check is GO, AMBER or NO-GO (thresholds in PREREG_KILL_TEST.md). Overall:
- **GO:** K1 GO on both models; K2 GO on at least one model and not NO-GO on the other; K3 GO on at
  least one model and not NO-GO on the other.
- **NO-GO:** K1 NO-GO on both, or K2 NO-GO on both (after the rescue), or K3 NO-GO on both.
- **AMBER:** anything else.

## 11. What happens after each outcome
- **GO:** stop and report. After Dawar confirms, start P2 (V1 equivalence check first) and P3.
- **AMBER:** stop and report with a diagnosis. Proposed 48-hour fix window, only after Dawar agrees:
  (a) if K2 is the problem, try G-sk and C = 32 clusters; (b) if K3 is the problem, find which layers
  hurt most (reduce one block of layers at a time) and test the pre-registered variant that keeps the
  first and last two MoE layers native; (c) the K3 extension in Section 9. Then decide again.
- **NO-GO:** stop. Write a one-page post-mortem: which premise failed and what the numbers say about
  MoE geometry. Special case: K1 GO but K2 and K3 NO-GO means the redundancy is real but cannot be
  exploited cheaply; recommend switching the GPU time to Compressed Circuit Tracing and keeping the
  measurement as a possible workshop note.

## 12. Report (`results/KILL_REPORT.md`, generated, then read and edited for clarity)
1. Three-line plain summary and the verdict.
2. Setup: models, precision (D7), revisions, data sizes, prereg commit hash, deviations (if any).
3. K0 table.
4. K1: per-layer median m95 and PR for both models (plot `plots/k1_m95_by_layer.png`), share of layers
   at 3 or fewer.
5. K2: layer error against m for every method, log scale (`plots/k2_error_vs_m_<model>.png`); per-layer
   ratio plot; the aggregate ratios at m = 1 and 2; U1 and U2 headroom; the chosen primary variant and
   its validation numbers.
6. K3: accuracy table with intervals at m*; recovery with interval; perplexity table.
7. Decision table with every cell, and the overall verdict.
8. What the numbers mean, in plain words; surprises; proposed next steps.
Also write `results/kill_test.json` with every number in the report and its run record.

## 13. Schedule and parallel work
| Step | OLMoE | Qwen3-30B-A3B |
|---|---|---|
| Downloads | 0.2 h | 0.5 to 1 h (network) |
| K0 | 0.3 h | 0.5 h |
| Calibration | 0.5 h | 1 to 1.5 h |
| Recording | 0.2 h | 0.5 h |
| Offline K1, K2, tau | 0.3 h (CPU or small GPU) | 0.5 h |
| K3 | 1 h | 2 to 3 h |
Order: start the Qwen3 download first; do all OLMoE steps while it downloads; run OLMoE's offline
analysis on CPU while Qwen3 calibrates. These times are estimates; log the real ones.
