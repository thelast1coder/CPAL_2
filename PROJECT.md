# PROJECT.md: Projection Routing for Mixture-of-Experts LLMs (CPAL 2027, paper 2)

Version 1, 10 November 2026. This file is the single source of truth for the project. If another
file disagrees with it, this file wins until Dawar changes it.

## 0. At a glance
- **Venue:** CPAL 2027 Proceedings Track (PMLR, 9 pages). Abstract registration 23 November 2026;
  full paper 5 December 2026 AoE. Fallback: Recent Spotlight Track, 18 January 2027.
- **Place in the lab's plan:** paper 1 is Mirror Surgery (separate repo). This is paper 2. If this
  project is killed, the reserve idea is Compressed Circuit Tracing.
- **Contribution type:** a measured law, a small piece of load-bearing theory, a training-free
  method, and a universal add-on that improves existing methods.
- **Compute budget:** about 60 to 90 H100-hours in total, on one H100 shared with two other projects.
- **First GPU job:** the kill test (PHASE1_KILL_TEST.md, about 6 to 8 GPU-hours). It decides GO,
  AMBER or NO-GO before any other GPU work.

## 1. Working title and thesis
**One Expert Is Almost Enough: Expert Coherence and Projection Routing for Mixture-of-Experts LLMs**
(working title; the final wording depends on results).

Thesis. Inside the set of experts the router picks for a token, the expert outputs are strongly
correlated, so a few of them span almost all of the layer's output. Once experts are dropped to save
compute, the router's gate weights are the wrong weights for the survivors. The right weights are a
small least-squares projection computed from a calibrated Gram matrix of expert outputs. Greedy
pursuit on the same Gram matrix chooses which experts to keep and when to stop.

## 2. The problem (open statement)
Fine-grained MoE LLMs (OLMoE, Qwen3-MoE, DeepSeek-V2) run k = 6 to 8 routed experts for every token
in every MoE layer. Many training-free methods lower this number at inference: a smaller fixed top-k,
adaptive k per token, expert skipping, and expert substitution for offloading. All of them decide
*which* experts to drop. Then they combine the remaining experts with the router's own gates, either
kept as they are or renormalised to the original total.

Two questions are open (confirm in the P0 literature check):
1. **How many experts does a token actually need?** How many independent directions do the k expert
   outputs span, measured on real models, layer by layer?
2. **Once some experts are dropped, what weights should the others get** so the layer's output is
   preserved?

Keeping the gates is exactly right only when expert outputs are mutually orthogonal. Renormalising is
exactly right only when they are identical. Real experts are neither. Nobody has measured where real
experts sit between these extremes, and nobody has used that geometry to set the weights.

## 3. Background a reader needs (plain)
**3.1 The MoE layer.** The router is a linear map from the token's normalised hidden vector x to N
logits, followed by a softmax that gives probabilities p_1 to p_N. The k experts with the largest p
form the token's set T(x). Each expert is a small SwiGLU MLP:
E_j(x) = W_down,j ( silu(W_gate,j x) * (W_up,j x) ), where * is elementwise.
The layer output is y(x) = sum over j in T of g_j E_j(x), plus shared experts if the model has them
(shared experts always run and this project never touches them). The gates g_j are p_j in OLMoE,
DeepSeek-V2-Lite and Qwen1.5-MoE, and p_j divided by the sum of p over T in Qwen3-MoE
(norm_topk_prob = True). Check every model's convention in its config and modelling code (verify).

**3.2 Where the cost is.** At batch-1 decoding, speed is set by bytes read from GPU memory, and every
active expert's weights must be read. With offloading (experts kept in CPU memory), every active
expert that is not cached on the GPU must be copied over. At large batch, arithmetic dominates. Fewer
active experts per token help in all three regimes, by different amounts. The paper reports each one
honestly (Section 9).

**3.3 Sparse approximation, the CPAL link.** Given a target vector y and a dictionary of atoms (here,
the k expert outputs), find a few atoms and coefficients that approximate y. Keeping the atoms with
the largest given coefficients is *thresholding*. Adding atoms one at a time by how much each reduces
the residual is *greedy pursuit* (orthogonal matching pursuit, OMP, and orthogonal least squares,
OLS). Classical results: thresholding works when atoms are nearly orthogonal and fails when they are
coherent (correlated); pursuit comes with guarantees under weaker conditions (Tropp 2004, exact
recovery condition; Das and Kempe 2011, weak submodularity). **Top-k reduction with router gates is
thresholding.** This project measures whether MoE experts are coherent and, where they are, replaces
thresholding with pursuit and projection.

## 4. Notation
- N: number of routed experts. k: native number of routed experts per token.
- T = T(x): the router's native top-k set for token x. g in R^k: native gates on T, with the
  model's own convention (renormalised or not).
- E_j = E_j(x) in R^d: output of expert j before weighting.
- G(x) in R^(k x k): the per-token Gram matrix of expert outputs on T, G_ij = <E_i, E_j>.
- S, a subset of T with |S| = m: experts kept. D = T minus S: experts dropped.
- c in R^m: weights on S. c~ in R^k: c padded with zeros on D.
- G^: a calibrated estimate of G (Section 5.4).
- Sums written sum_S g mean the sum of g_j over j in S. a = sum_S g, b = sum_D g,
  q_S = sum_S g^2, q_D = sum_D g^2.

## 5. Mathematics (load-bearing; every formula has a CPU test in tests/)
**5.1 Exact error is a quadratic form.** With y = sum_T g_j E_j and y^ = sum_S c_i E_i:

    e(S, c) = ||y^ - y||^2 = (c~ - g)^T G(x) (c~ - g),      ||y||^2 = g^T G(x) g.

Relative error rho = e / ||y||^2. Consequence: storing G(x) (k x k numbers) per token per layer is
enough to evaluate any keep-and-reweight rule exactly, offline, without rerunning the model. The kill
test relies on this.

**5.2 Optimal weights for a fixed subset.** Minimising e over c gives the normal equations
G_SS c = G_ST g, so

    c* = G_SS^(-1) G_ST g.

With the true per-token G(x) this is the orthogonal projection of y onto span{E_i : i in S} (oracle
weights). At run time the entries of G(x) that involve dropped experts are unknown (those experts are
never computed), so we use a calibrated estimate:
c^ = (G^_SS + lambda * mean(diag(G^_SS)) * I)^(-1) G^_ST g, with a small ridge lambda (default 1e-3,
tuned on the validation split only).

**The unification (claim C2).** Two familiar rules are this projection under extreme assumptions:
- Experts orthogonal with equal norms (G proportional to I): c* = g_S. Keeping the gates (plain
  top-m) is optimal only here.
- Experts identical (G proportional to the all-ones matrix): every c with sum(c) = sum_T g is exact.
  Renormalisation, c_i = g_i (a + b) / a, is one of them. It is exact only here.
- **Uniform coherence** (unit norms, every pair of experts with cosine r, G = (1 - r) I + r 1 1^T).
  The optimal weights have a closed form. For i in S:

        c_i = g_i + r b / (1 - r + r m).

  At r = 0 this keeps the gates; as r goes to 1 it adds the dropped mass b in equal shares.
  The three rules' errors under this model (all checked numerically, 2,000 random cases):

        keep gates:   e1 = (1 - r) q_D + r b^2
        renormalise:  e2 = (1 - r) ( (b/a)^2 q_S + q_D )
        projection:   e3 = (1 - r) q_D + (1 - r) r b^2 / (1 - r + r m)
        ||y||^2      = (1 - r)(q_S + q_D) + r (a + b)^2
        e2 - e3      = (1 - r) b^2 ( q_S / a^2 - r / (1 - r + r m) )  >= 0

  What this predicts: projection beats keeping the gates by most when r is large, and beats
  renormalisation by most when r is small or middling. At m = 1 with equal gates over k = 8, the
  ratio e2 / e3 is 4.7 at r = 0.1, 2.6 at r = 0.3, 1.8 at r = 0.5 and 1.1 at r = 0.9.
  **So when experts are nearly identical, renormalisation is already close to optimal.** The extra
  gains of the full method must come from unequal norms, structured (non-uniform) correlations and
  better selection. This is why the kill test's layer-error bar is 2x (not higher) and why the
  downstream check K3 carries the decision.
- After reading arXiv 2609.04575 (the "k2 fix"), state exactly where its rule sits relative to this
  family (P0 literature note). Do not claim it is a special case until shown.

**5.3 Selection by greedy pursuit (OLS form, using only the k x k Gram).** Residual energy after
keeping S with optimal weights:

    R(S) = g^T G g - g^T G_TS G_SS^(-1) G_ST g.

Greedy rule: start with S empty; add the j in T minus S that gives the largest drop R(S) - R(S + {j});
repeat. Fixed budget: stop at |S| = m. Adaptive budget: stop when R(S) / (g^T G g) <= tau, with
m_min = 1 and m_max = k. Selection uses G^ because it happens before any expert is computed. For
m = 1 the rule picks argmax_j (G^_jT g)^2 / G^_jj: the expert best aligned with the full output, not
necessarily the largest gate. Cost: O(k^3) per token per layer with k = 8, batched; negligible next
to one expert. The OMP criterion (largest normalised correlation with the current residual) is
implemented as an ablation.

**5.4 Gram estimators (cheapest first).**
- **G-avg.** Per layer, G^_ij = mean of <E_i, E_j> over calibration tokens where both i and j are in
  T; G^_ii = mean of ||E_i||^2 over tokens where i is in T. Pairs with fewer than n_min = 32
  co-occurrences back off to r_bar * sqrt(G^_ii G^_jj), where r_bar is the layer's count-weighted
  mean cosine.
- **G-cos (two-pass, norm-aware).** Store the calibrated cosine matrix R^ and mean norms. Select with
  G-avg. After the kept experts are computed, rebuild G^ for weighting with their actual norms and the
  calibrated mean norms of dropped experts: G^_ij = n_i n_j R^_ij. Extra cost: m norms.
- **G-clu.** k-means with C clusters (C in {8, 16}) on the L2-normalised layer input, fitted on
  calibration tokens; a G-avg per cluster, backing off to the layer's G-avg when counts are low.
  Run-time cost: C dot products of length d.
- **G-sk** (rescue only, Section 5.6): per-token estimates from cheap expert sketches.

**5.5 Effective number of experts (the law, claim C1).** Two measures per token per layer, from the
true G(x):
- **m95(x):** the smallest m such that the best m-subset of T with oracle weights leaves relative
  error <= 0.05 (exhaustive search; at most C(8, 4) = 70 subsets per m). Also m90 and m99.
- **Participation ratio** PR(x) = (sum lambda)^2 / sum lambda^2, where lambda are the eigenvalues of
  M = diag(g) G(x) diag(g), the Gram matrix of the weighted outputs g_j E_j. PR lies between 1 and k
  and counts how many orthogonal directions carry the energy.
Also report: the cosine between the top-gate expert's output and y; the mean pairwise cosine r inside
T; distributions by depth and by token domain (text, maths, code).

**5.6 Sketch fallback (only if cheaper estimators fail K2).** For each expert j choose a fixed set
R_j of r intermediate neurons (r = 5% of the expert's intermediate size, rounded up) with the largest
calibration mean of |a_jn(x)| * ||W_down,j[:, n]|| over tokens routed to j (a_jn is the neuron's
activation after the SwiGLU product). Sketch: E^_j(x) = W_down,j[:, R_j] a_j,R_j(x), costing about
r / I of one expert (I = intermediate size). Fit a scalar kappa_j on calibration data by least
squares of E_j on E^_j. Entries of G involving a dropped j are then estimated per token with
kappa_j E^_j. Cost accounting: m + k * r / I expert-equivalents per token. An honest baseline sits
beside it, **sketch-add**: y^ = sum_S g_i E_i + sum_D g_j kappa_j E^_j, so we can tell whether the
projection or the sketch is doing the work.

**5.7 Theory to test, not just quote (claim C2, tested in P4).**
- **Coherence:** mu = largest off-diagonal |cosine| inside T, and the mean cosine r. Prediction from
  5.2: the measured gain of projection over each simple rule follows the closed forms when we plug in
  the layer's r, gates and norms.
- **Weak submodularity (Das and Kempe, ICML 2011):** greedy forward selection for explained energy
  reaches at least a fraction (1 - e^(-gamma)) of the best subset of the same size, where the
  submodularity ratio gamma is bounded below by a restricted smallest eigenvalue of the normalised
  Gram. Prediction: layers with larger lambda_min have a smaller greedy-to-oracle gap. Check the exact
  statement in the paper before writing it (verify).
- **Exact recovery condition (Tropp 2004):** if max over j outside S* of
  ||G_(S*S*)^(-1) G_(S* j)||_1 < 1 for the optimal support S*, OMP picks only atoms from S*. Measure
  the fraction of tokens per layer where it holds; prediction: where it holds, pursuit matches the
  oracle subset. Check the statement in the source (verify).
- **Thresholding failure:** measure how often the largest-gate expert is not the oracle's best single
  expert, by layer.
Theory goes into the paper only together with its measured check.

**5.8 Error relative to the residual stream.** The MoE output is added to the residual stream h, so
also report ||y^ - y|| / ||h||. It shows how large the perturbation is for the next layer.

## 6. Claims (each tied to experiments in EXPERIMENTS.md)
- **C1, the law:** in fine-grained MoE LLMs, the effective number of experts per token is small,
  about 1 to 2.5 of 8 by m95 (hypothesis; K1 in P1, Z1 in P4).
- **C2, theory:** keeping gates and renormalising are the projection under two extreme assumptions
  (orthogonal or identical experts); the uniform-coherence closed form interpolates between them;
  coherence, weak submodularity and the exact recovery condition predict where pursuit and projection
  win (P4).
- **C3, method:** Projection Routing (pursuit selection, least-squares weights from a calibrated Gram,
  adaptive residual stop) is training-free, adds negligible overhead and keeps near-native accuracy
  at about 1.5 to 2 active experts of 8 (hypothesis; P2).
- **C4, universal add-on:** replacing the weights of existing reduction methods (top-m with kept
  gates, top-m renormalised, k2 fix, elbow, EAT, SERE, cache substitution, OMP-MoE) with projection
  weights improves each at its own budget (P3).
- **C5, deployment:** fewer active experts give measured speed-ups at batch-1 decoding and fewer
  expert loads under offloading (P5).

**Headline targets (hypotheses; never report them as results):**
- Qwen3-30B-A3B at about 1.5 active experts of 8 (5x fewer routed-expert computations): within 1 to 2
  points of native on GSM8K and MATH-500.
- At 1 expert: recover at least half of the accuracy plain top-1 loses. Reference point: SERE
  (arXiv 2602.07616) reports Qwen3-30B-A3B native 82.24, top-2 80.37, top-1 64.11 (check which
  benchmark and setting, verify).
- Layer error at least 2x lower than the best simple rule at m = 1 and 2 (stretch: 5x).

## 7. The method in one paragraph (what we ship)
Offline, once per model: calibrate G-avg (plus cosines, norms and clusters) on about 1M tokens. At
inference, per token and per MoE layer: run the router as usual to get T and g; greedy pursuit on G^
selects S (fixed m, or adaptive with threshold tau); compute only the experts in S; weight them with
c^ from G^ (or from G-cos once their norms are known); output y^ = sum_S c_i E_i plus shared experts.
No training. The code change is one wrapper around the MoE block, plus a routing function for vLLM.

## 8. Models
| Role | Model (Hugging Face ID, verify) | Routed experts | Routing | Notes |
|---|---|---|---|---|
| Kill test, main | allenai/OLMoE-1B-7B-0924 (base) and allenai/OLMoE-1B-7B-0924-Instruct | 64, k = 8 | softmax, no renormalisation | about 14 GB in bf16; fp32 fits for exactness checks |
| Kill test, headline | Qwen/Qwen3-30B-A3B | 128, k = 8 | softmax, renormalised over top-k | about 61 GB in bf16; non-thinking mode for quick GSM8K; FP8 checkpoint Qwen/Qwen3-30B-A3B-FP8 as fallback (decision D7) |
| Generality | deepseek-ai/DeepSeek-V2-Lite-Chat | 64 plus 2 shared, k = 6 | softmax (check grouping and scaling factor) | shared experts untouched; first layer dense (verify) |
| Generality | Qwen/Qwen1.5-MoE-A2.7B-Chat | 60 plus a shared expert, k = 4 | softmax, no renormalisation (verify) | shared expert has a sigmoid gate, untouched |
| Optional | openai/gpt-oss-20b | 32, k = 4 | softmax over the top-k logits (verify) | MXFP4 weights; only if time allows |

Record every model's values (N, k, norm_topk_prob, intermediate size, layer count, shared experts,
scaling factor, dense layers) in `configs/models/*.yaml` from the downloaded config.json, never from
memory.

## 9. Metrics
**Layer level** (offline, from stored Grams): relative error energy, sum_x e(x) / sum_x ||y(x)||^2
(primary); median token relative error; error relative to the residual stream; m95, m90, m99; PR;
coherence (mu and mean r); realised active experts.

**Task level:** GSM8K (exact match), MATH-500 (math-verify or equivalent, verify), MMLU (5-shot,
lm-evaluation-harness), ARC-Challenge, HellaSwag, HumanEval and MBPP (pass@1, greedy), WikiText-2
perplexity. Long reasoning (Qwen3 thinking mode on a MATH-500 subset) if time allows.

**Compute:** mean active routed experts per token (over tokens and MoE layers); routed-expert FLOPs
per token; total active parameters per token (attention, router, shared and routed experts,
embeddings, all computed from config); expert bytes read per decode step at batch 1; expert loads per
token under an LRU expert cache (offloading, trace-driven simulation); measured decode latency and
throughput wherever a real kernel path exists.

**Statistics:** paired bootstrap 95% intervals (10,000 resamples) for accuracy differences on the
same questions; always report n.

**Honesty rule for compute.** "5x fewer expert computations" is not "5x faster". Always state the
routed-expert share of active parameters (computed from config) and the measured speed.

## 10. Baselines (all training-free, all compared at matched budgets)
| ID | Method | Source | Notes |
|---|---|---|---|
| B0 | Native top-k | the model | reference |
| B1 | Top-m, gates kept | standard | thresholding |
| B2 | Top-m, gates renormalised to the native total | standard | the most common reduction |
| B3 | k2 fix | arXiv 2609.04575 | reported (check the model): MMLU drop -4.65 to -0.35 and GSM8K drop -6.60 to -0.60 at m = 4 (verify) |
| B4 | Elbow adaptive k | arXiv 2608.04401 | implement from the paper (verify mechanism) |
| B5 | EAT adaptive k | arXiv 2609.33614 | implement from the paper (verify mechanism) |
| B6 | OMP-MoE | arXiv 2609.31631 | closest prior work; read first (novelty gate in P0) |
| B7 | SERE | arXiv 2602.07616 | keeps router weights; reproduce its Qwen3-30B-A3B top-1 and top-2 numbers first |
| B8 | Cache-substitution routing | arXiv 2508.18983; CCE, arXiv 2412.00099 | offloading setting |
| B9 | Skipping by gate ratio | for example arXiv 2402.14800 (verify) | simple adaptive rule |
| U1 | Oracle: true per-token Gram, best subset, oracle weights | ours | upper bound, not deployable |
| U2 | Oracle weights on the gate-order subset | ours | ceiling for reweighting alone |

**Fidelity rule:** where a baseline paper reports numbers on a model we also use, reproduce them
within 1.5 points before our reimplementation goes into a table. Otherwise label it "our
reimplementation" and say how it may differ.

## 11. Prior work and our delta (complete and verify in P0, `docs/related_work_notes.md`)
| Work | What it does (verify each line) | Our delta |
|---|---|---|
| OMP-MoE, arXiv 2609.31631 | Closest by name; read in full first | Decide in P0 whether it estimates expert-output inner products and solves for weights. If it does both, stop and report |
| k2 fix, arXiv 2609.04575 | Inference-time k reduction with a correction | Place its rule within the projection family (5.2); add-on target |
| SERE, arXiv 2602.07616 | Re-routes to similar experts; keeps router weights | Optimal weights for any chosen set; add-on target |
| Score-based cache substitution, arXiv 2508.18983; CCE, arXiv 2412.00099 | Substitute missing experts with cached ones | Projection weights for substituted sets |
| Elbow, arXiv 2608.04401; EAT, arXiv 2609.33614 | Adaptive number of experts per token | Pursuit stop on residual energy; add-on target |
| MoE Lens, arXiv 2603.05806 | Reports the top-1 expert's output at cosine about 0.95 with the full MoE output (check the model and layers) | We measure the whole geometry (Gram, m95, PR) and exploit it |
| arXiv 2601.03577 | MoE theory (check what it proves) | Relate or distinguish |
| LightMoE, arXiv 2603.12645; BuddyMoE (find the ID); MoE-Spec, arXiv 2602.16052 | Efficiency and offloading methods (check) | Position in related work |
| Dynamic top-p routing, arXiv 2403.07652 (verify) | Trained adaptive routing | Ours is training-free |
| Tropp 2004; Das and Kempe 2011 | Pursuit guarantees | Applied to and tested on MoE routing |

**Delta in one line:** existing methods decide which experts to drop; we show the remaining experts
need different weights, give the optimal weights in closed form from the expert-output geometry,
select with pursuit, and measure the geometry that explains when any of this works.

## 12. Rules and constraints
- **Training-free only.** No gradient updates to any weight, router included.
- **Data hygiene.** Calibration, validation and analysis tokens are disjoint from each other and from
  every benchmark's test items. GSM8K and MATH test items are never used for calibration or tuning.
  Disjointness is checked by document hashes and logged.
- **Pre-registration.** PREREG_KILL_TEST.md is committed before the first real-model run; its
  thresholds never change after that. Later gated claims get their own PREREG file before running.
- **Matched budgets.** Every comparison reports the realised mean active routed experts per token.
  Adaptive methods are tuned on the validation split to hit the target mean.
- **Identical evaluation settings** across methods: prompts, decoding, max tokens, answer extraction.
- **Reproducibility.** Fixed seeds; every result file stores config, seed, git hash, package versions,
  GPU name and wall time.
- **No invented facts.** Numbers, citations, model IDs, dataset IDs and API details are checked or
  marked (verify).
- **Laptop:** about 0.5 GB free disk and no GPU. No model, dataset or weight downloads. Reuse the
  system PyTorch; install only small packages; if space runs out, write code and run tests on the VM.
- **GPU:** one H100 shared with Mirror_CPAL and a CVPR project. Check `nvidia-smi` before each job,
  cap memory per process, queue jobs, never kill processes this project did not start.
- **Secrets:** use a fresh Hugging Face token on the VM (the earlier token was pasted in chat and must
  be revoked). Never commit tokens, keys or model files.
- **Writing:** British spelling, plain language, no em-dashes, no emoji.

## 13. Decision register
**Taken**
| ID | Decision | Reason |
|---|---|---|
| D1 | This is CPAL paper 2; Compressed Circuit Tracing is the reserve | Agreed 10 Nov |
| D2 | Training-free only (no router tuning, merging or distillation) | Clean claim, cheap compute, universal add-on |
| D3 | Candidate pool for the main method is the router's top-k set T (S is a subset of T); pool 2k is an ablation | Keeps the target and comparison clean |
| D4 | The reduction target is the native layer output y | Closed form, model-agnostic, offline-checkable |
| D5 | The HF transformers path is the reference implementation; the vLLM path is used only after it matches HF | Correctness first |
| D6 | Kill-test models: OLMoE-1B-7B (base for statistics and perplexity, Instruct for GSM8K) and Qwen3-30B-A3B (non-thinking for GSM8K) | One small, one headline; both k = 8, opposite gate conventions |
| D7 | OLMoE in bf16 (fp32 for exactness checks). Qwen3-30B-A3B in bf16 if at least 70 GB of GPU memory can be reserved for the run, otherwise the official FP8 checkpoint; decide at P1 start, record in STATUS.md, use the same base for every Qwen3 number in the paper | Shared GPU |
| D8 | Calibration mix v1: 60% general web text (FineWeb-Edu sample), 40% maths (MetaMathQA, chat-formatted) | Covers text and reasoning without touching test sets |
| D9 | Sizes: calibration 1M tokens; validation 20k; analysis 50k (OLMoE) and 30k (Qwen3); sequence length 1024; padding excluded from all statistics | Enough for pair statistics; fits disk |
| D10 | Matched-budget protocol: adaptive thresholds tuned on validation tokens to hit a target mean | No tuning on benchmarks |
| D11 | Primary layer metric: relative error energy. Kill-test task: a fixed 250-question GSM8K subset plus WikiText-2 perplexity | Cheap and decisive |
| D12 | Shared experts always run and are never counted in the budget | Same for every method |
| D13 | Kill-test thresholds as in PREREG_KILL_TEST.md; layer-error GO bar 2x, chosen because the closed forms in 5.2 show renormalisation is near-optimal when experts are almost identical | Honest bar; K3 carries the decision |

**Pending (who decides, when)**
| ID | Decision | Who, when |
|---|---|---|
| PD1 | GO, AMBER or NO-GO path | Dawar, after the kill-test report (target 14 Nov) |
| PD2 | Headline budget (1.5 or 2 experts) and title wording | Dawar with Claude Code, after P2 on Qwen3 |
| PD3 | Title and abstract for registration | Dawar, 22 Nov (registration closes 23 Nov) |
| PD4 | Latency via vLLM or a small custom kernel; if neither gives a real speed-up, C5 is reduced to bytes read and expert loads | After the P5 equivalence test |
| PD5 | Which baselines have faithful reimplementations; drop or label the rest | During P3 |
| PD6 | Whether the sketch variant appears in the main paper or only the appendix | After P6 |
| PD7 | If a new paper with the same method appears (arXiv or OpenReview), stop and discuss | Any time; recheck in the final week |

## 14. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Novelty: OMP-MoE or the k2 fix already solve for least-squares weights from expert inner products | P0 novelty gate: read both in full before any GPU time |
| The calibrated Gram is too crude per token | G-cos, G-clu, then G-sk; K2 measures this directly; U1 shows the ceiling |
| Layer gains do not carry downstream (errors compound over 16 to 48 layers) | K3 tests downstream directly; pre-registered variant keeps the first and last two MoE layers native |
| Renormalisation is already near-optimal (very coherent experts) | Closed forms predict it; then the paper leans on the law (C1), selection and the add-on (C4) at low budgets |
| Qwen3-30B-A3B memory on a shared GPU | Book a bf16 window or use FP8 consistently (D7) |
| HF generation too slow for the full grid | vLLM path after the equivalence test; otherwise reduced grids (EXPERIMENTS.md, plan B) |
| transformers version differences in MoE code | Pin one version; adapters detect the expert storage layout and fail loudly if unknown |
| Overclaiming compute | Report routed-expert share and measured speed (Section 9) |
| Weak baseline reimplementations | Fidelity rule (Section 10) |
| Time | Gates with dates (Section 15); stop rules in every phase |

## 15. Timeline (today is 10 Nov 2026)
| Dates | Work |
|---|---|
| 10 Nov | Zip handed to Claude Code |
| 11 to 12 Nov | P0 on the laptop: repo, literature notes and novelty gate, all code, CPU tests, tiny end-to-end run, PREREG committed |
| 13 to 14 Nov | P1 on the VM: setup, K0 sanity, kill test, report. Decision by the evening of 14 Nov |
| 15 to 21 Nov | P2 core results; P3 add-on starts in parallel |
| 22 Nov | Freeze title and abstract (PD3) |
| 23 Nov | Abstract registration closes |
| 22 to 27 Nov | P4 theory checks, P5 deployment, P6 ablations |
| 27 to 29 Nov | P7 seeds, figures, FINAL_RESULTS.md |
| 24 Nov to 4 Dec | Writing |
| 5 Dec | Submit |

## 16. Glossary (plain)
- **Expert:** one small MLP inside an MoE layer.
- **Router, gate:** the router scores all experts for a token; the gates are the weights given to the
  chosen ones.
- **Top-k:** run the k highest-scoring experts.
- **Gram matrix:** the table of inner products between expert outputs; it holds their lengths
  (diagonal) and how much they point the same way (off-diagonal).
- **Coherence:** how correlated the expert outputs are.
- **Projection:** the best weighted sum of the kept experts' outputs for approximating the full output.
- **Pursuit:** choosing experts one at a time by how much each reduces what is still unexplained.
- **Calibration:** a one-off pass over unlabelled text to estimate the Gram matrix; no training.
- **m95:** the smallest number of experts that reproduces 95% of the layer output's energy.
