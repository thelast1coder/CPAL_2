# Related work notes and the novelty gate (P0)

Compiled 10 November 2026 from arXiv HTML and PDFs. Every paper block below gives what it changes,
whether it is training-free, how it chooses experts, how it weights the remaining experts, models and
headline numbers (with table references) and open questions. Anything not checked directly is marked (verify).
Parts 1 to 3 at the end are the full reading notes; this header is the summary.

## Novelty gate outcome: NOT TRIPPED (with two overlaps to report)

Gate (PHASE0_SETUP.md Section 3): stop if a paper (a) estimates inner products or correlations between
expert outputs AND (b) uses them to solve for the weights of a reduced expert set; also report any paper
that measures the effective number of experts per token through the span of expert outputs (c).

- **(a) and (b) together: no paper found.** (a) alone exists: arXiv 2605.28207 (Pruning and Distilling
  MoE into Dense LMs) uses exactly our calibrated expert-output Gram, G_ij = E_t[<f_i(t), f_j(t)>], but only
  to choose experts (greedy log-det) for static pruning, then static scaling and distillation. OMP-MoE
  (2609.31631) uses inner products of gated expert contributions with a residual, but forces coefficients
  to {0, 1} and argues explicitly against a least-squares refit ("expert weights are determined by the
  router, not by a post-hoc least-squares fit"). SERE and RAZOR use output similarity without refitting.
- **(b) partial overlaps, none per token on kept experts:** MoRA (2610.00367, affine fit of one survivor
  to one pruned expert; router bias learned, not training-free), 2610.41366 (ridge regression of a pruned
  expert's output on the input x), PRIME (2610.13982, closed-form ridge refit of the ROUTER on hidden states),
  ConMoE (2605.29350, one-to-one reassignment with gate carried over unchanged), COMPEL (Fisher-weighted
  parameter compensation). None solves c = G_SS^-1 G_ST g.
- **(c): no paper found.** Closest: 2605.28207's effective rank of the selected expert set (per layer,
  not per token) and OMP-MoE-dagger's per-token count from diagonal energies only (cross terms ignored).
- **Theory overlap to cite and distinguish:** arXiv 2601.03577 (Su and Liu) frames routing as sparse subset
  selection over expert outputs, uses the OLS normal equations and Tropp's coherence condition. It is
  training-time theory (orthogonality regulariser, synthetic experiments), targets an "optimal target
  representation" rather than the dense top-k output, and does no inference-time reduction or refit. Its
  Theorem 5.3 states Tropp's sufficient condition as "if and only if" (looks overstated).
- **Coverage limit:** Semantic Scholar and the arXiv API were rate-limited; Google Scholar was not
  queried directly. A second search pass in the final week is planned anyway (PD7).

## Where the k2 fix sits relative to the projection family (PROJECT.md 5.2)

k2 fix (arXiv 2609.04575, Chen and Yao, Eq. 2): S = top-k1 by router probability, w_i = p_i / sum over
T_k2 of p_j, p = softmax over all experts. With native renormalised gates g_i = p_i / m(T) (a + b = 1):

    c_i = lambda(x) g_i on S,   lambda(x) = m(T) / m(T_k2)  (a per-token scalar from router mass only)

- k2 = k1: lambda = 1/a, renormalise. k1 < k2 < k: between renormalise and keep. k2 = k: keep.
  k2 > k: lambda < 1, shrinks below keep. k2 = N: raw softmax.
- The least-squares weights are c_LS = g_S + G_SS^-1 G_SD g_D. The k2 family always points along g_S, so
  it equals c_LS only when G_SS^-1 G_SD g_D is parallel to g_S: keep when G_SD g_D = 0 (dropped output
  orthogonal to every kept output), renormalise when G_SS^-1 G_SD g_D = (b/a) g_S (for example identical
  experts). The k2 > k regime has no counterpart in the projection. So the k2 fix is a one-parameter
  rescaling of the gates chosen from router mass, not a special case of the projection; the two coincide
  only in the two extreme geometries. Numerical check: `tests/test_eval_and_prereg.py::
  test_k2fix_placement_in_projection_family` (passes).
- Open question for B3 on OLMoE: OLMoE's router does not renormalise, so its trained convention is k2 = N
  (raw probabilities, which equals keep). The paper only tests renormalising routers. Our B3 applies the
  literal formula with k2 = k by default (configs/experiments/kill_test.yaml, `b3_k2`); on OLMoE that gives
  c_i = p_i / m(T), which is NOT the trained scale. Decide before K3 whether B3 on OLMoE should use k2 = N
  (= B1) or the literal k2 = k (needs Dawar; logged as an open question, not changed silently).

## Baseline reimplementation status (src/prouting/routing/baselines.py)

| ID | Our reading | Fidelity |
|---|---|---|
| B3 k2 fix | Eq. 2 exactly; default k2 = k; paper recommends a scan over {k1, k, 2k} | literal formula; OLMoE convention open (above) |
| B4 Elbow | Kneedle on sorted full softmax; k = min(e + 1, K); gates kept unless the model renormalises (released code) | unchecked: paper writes both "e + 1" and "min(e, K)" |
| B5 EAT | history score + importance + mean/std threshold; renormalise; OPD (router fine-tuning) excluded | unchecked: perplexity-based threshold adjustment omitted (needs labels); candidates restricted to T |
| B6 OMP-MoE | dagger inference rule (Eq. 14): energy prefix on g_j^2 alpha_j in router order, eps = 0.1, keep gates | faithful for the inference rule; static pruning part not a per-token policy |
| B7 SERE | batch primary set (top-S), re-route secondaries with similarity >= rho, gates unchanged | unchecked: duplicate-expert handling in their kernel; similarity from our full-Gram calibration |
| B8 cache substitution | CCE Cache-Prior and SMoE substitution modes with an LRU cache | unchecked: CCE renormalisation; SMoE replaceset pooling |
| B9 gate-ratio skip (NAEE) | top-2 rule generalised to k: keep g_j >= beta g_max; beta = calibration median or tuned; renormalise | unchecked: generalisation beyond top-2 is ours |

## Corrections to PROJECT.md beliefs (for Dawar; PROJECT.md not edited)

- **SERE numbers (B7):** 82.24 / 80.37 / 64.11 are confirmed (Table 3, Qwen3-30B-A3B, OpenCompass average of
  8 tasks, sampling at temperature 0.7). But SERE "top2/top1" is a batch-level union of each token's
  top-S primary experts plus re-routed and critical experts, not 2 or 1 experts per token; plain top-2 scores
  14.30 in the same table. These numbers cannot be compared with our per-token budgets.
- **MoE Lens (2603.05806):** the "cosine about 0.95" compares residual-stream states (top-1 expert term plus
  residual vs top-6 plus residual) on DeepSeekMoE, "as high as 0.95" in some layers. It is not the MoE
  output alone, which inflates the cosine.
- **k2 fix headline (B3):** reported on Qwen3.6-35B-A3B and Qwen3.5-397B-A17B, not on our models. MMLU
  (Table 5, n = 2000): native 81.65; k1 = 4: 77.00 at k2 = 4 (-4.65) and 81.30 at k2 = 16 (-0.35). GSM8K
  (Table 6, n = 500): native 95.00; k1 = 4: 88.40 at k2 = 4, 94.40 at k2 = 16. The PROJECT.md line "MMLU drop
  -4.65 to -0.35 and GSM8K drop -6.60 to -0.60 at m = 4" matches these tables.
- **EAT:** its best rows use router fine-tuning (OPD), not training-free.
- **Dynamic top-p (2403.07652):** trained from scratch; not training-free (as expected).
- **BuddyMoE:** arXiv 2511.10054.
- **Das and Kempe:** the bound uses the "smallest k-sparse eigenvalue" lambda_min(C, 2k); their Forward
  Regression is our OLS. Exact statements are in Part 2 below.

---


# Part 1 (full reading notes)

# Literature check, part 1: training-free reduction of active experts in MoE LLMs

Compiled 10 November 2026. All five priority papers were read from the arXiv HTML versions (arxiv.org/html/<id>), converted to text locally, and quoted from that text. Anything not checked directly is marked (verify).

Our method, for reference: per token, router top-k set T with gates g; keep S, drop D = T \ S; a = sum_S g, b = sum_D g. Expert-output Gram G (G_ij = <E_i(x), E_j(x)>) estimated once on calibration data (per layer, optionally per input cluster). Kept weights c = G_SS^{-1} G_ST g (projection of the full output onto span of kept experts). Selection by greedy OLS/OMP on the same Gram with an adaptive stop on residual energy. Effective number of experts per token via span of expert outputs (smallest m with relative error <= 5% under oracle weights; participation ratio of the weighted Gram).

---

## 0. Novelty gate summary (details in each block)

| Paper | (a) inner products / correlations of expert OUTPUTS? | (b) solves for combination weights of reduced set? | (c) effective number of experts per token via span of outputs? |
|---|---|---|---|
| OMP-MoE 2609.31631 | Yes (Frobenius inner products of residual with gated expert contribution atoms, over calibration tokens) | No. Coefficients forced to {0,1}; authors explicitly reject a least-squares fit | No. Per-token count uses diagonal energy g^2 * E||F_e||^2 only; per-layer residual curves are not per-token span |
| k2 fix 2609.04575 | No | No (scalar per-token gain from router mass only) | No |
| SERE 2602.07616 | Yes (Frobenius-distance / cosine / CKA similarity of expert outputs on calibration data) | No. "the re-routing process does not modify the router weights" | No |
| Elbow 2608.04401 | No (router probabilities only) | No | No (elbow of router probabilities) |
| EAT 2609.33614 | No (EMA of output norm only, no cross terms) | No (renormalises) | No |
| Pruning and Distilling MoE into Dense, 2605.28207 | Yes, explicitly: "expert output Gram matrix G_ij = E_t[<f_i(t), f_j(t)>]" | No (static uniform or proportional scaling, then knowledge distillation) | Partial: effective rank of the selected K-expert kernel, per layer, not per token |
| HOPE 2609.18916 | Partial: pairwise co-activation matrix F built from products of gated output NORMS, not inner products | No (binary QP for pruning set; renormalised refill) | No |
| RAZOR 2609.30465 | Yes, per token (consensus residuals f_i - c); interaction Gram of residuals as a diagnostic | No (renormalised survivors / router refill) | No |
| MoRA 2610.00367 | Yes, one-to-one: pruned-expert output predicted from one retained expert's output | Partial: channel-wise affine fit a ⊙ E_j(x) + b of ONE survivor to ONE pruned expert (static, memory side) | No |
| Replacement operators study 2610.41366 | Yes (residual inner products in an exploratory QP) | Partial: "tilted novice" is a reduced-rank ridge regression of a pruned expert's output on the INPUT x, not on other experts' outputs | No |
| OptEP 2610.27123 | Indirectly (layer-output reconstruction loss) | No for combination weights: relaxed selection variables z enter via renormalised router scores; then Adam updates of expert parameters (EWU) | No |
| You only need 2/3, 2609.25809 | No | No (renormalise) | No |
| ACE 2609.05228 | No (weight-spectral and router-direction proxies) | No (renormalise) | No |
| Don't Skip the Baseline 2610.28369 | No (benchmarks NAEE, DiEP, MoDES, Ada-K) | No | No |

Bottom line: the component "Gram / inner products of expert outputs estimated on calibration data" already exists in prior art (2605.28207 most explicitly; OMP-MoE; SERE; RAZOR diagnostic). No paper read solves for the weights of a reduced per-token expert set by least squares / projection on that Gram, and no paper measures a per-token effective number of experts through the span of expert outputs. OMP-MoE is the closest on selection (greedy pursuit on output inner products, plus an adaptive per-token energy stop), but with binary coefficients and a diagonal energy proxy, and it explicitly argues against a least-squares refit.

---

## 1. OMP-MoE (priority 1, read in full including appendices A to F)

- Exact title: "OMP-MoE: Efficient Expert Pruning for Mixture-of-Experts LLMs via Orthogonal Matching Pursuit"
- Authors: Dezhi Li, Lujun Li, Qiyuan Zhu, Hao Gu, Bei Liu, Sirui Han, Yike Guo (HKUST)
- Date: v1 submitted 9 September 2026; v2 revised 29 September 2026 (read v2). arXiv 2609.31631 [cs.LG]. A search snippet reported the arXiv comment "Work in progress, revisions ongoing" (verify).
- What it changes: (i) STATIC expert pruning (removes experts from each layer, memory side); (ii) cross-layer budget allocation by water-filling; (iii) OMP-MoE†, a per-token adaptive skip at inference (compute side).
- Training-free: yes. "we present OMP-MoE, a novel training-free compression framework".

### How it chooses experts (static part)
- Atoms: for a calibration mini-batch of N = B x T tokens, Y in R^{N x d} is the layer output and V_e in R^{N x d} is the "weighted expert contribution" of expert e, Y = sum_e G(X)_e ⊙ F_e(X) = sum_e V_e (Eq. 1). Note V_e is the gated contribution and is zero for tokens not routed to e.
- Objective (Eq. 4): min over |E| = n of || Y - sum_{e in E} V_e ||_F^2, i.e. OMP with coefficients constrained to {0,1}: "Crucially, we constrain the optimization coefficients to be binary, i.e., c_e in {0,1}, indicating whether an expert is retained."
- Greedy rule (Eq. 5 / Alg. 1): Delta_e = 2<R_t, V_e> - ||V_e||_F^2; e* = argmax over unselected e; R_{t+1} = R_t - V_{e*}. Inner product is Frobenius over calibration tokens. Complexity O(L n N_e B d).
- Cross-layer allocation (Eqs. 6 to 13): normalised residual r_l(k) = ||R_l(k)||_F^2 / ||Y_l||_F^2; routing coverage C_l(k) = sum of normalised accumulated gate mass of first k experts; risk psi_l(k) = -log(C_l(k) + zeta); cost F_l(k) = r_l(k) + lambda psi_l(k); greedy water-filling adds one expert at a time to the layer with largest F_l(k_l) - F_l(k_l + 1).
- Hyperparameters (Table 12, Sec. 4.1): lambda = 3.0, zeta = 1e-6, min expert ratio 0.375, max expert ratio 0.875 per layer, epsilon = 0.1 for OMP-MoE†, 64 WikiText-2 calibration blocks of 4,096 tokens (262,144 tokens; App. C.2), 8 x H20 GPUs.

### How it chooses experts at inference (OMP-MoE†, Sec. 3.4, App. B.2, Alg. 3)
- Precompute alpha_e = E[||F_e(x)||^2] (ungated output energy) offline.
- Per token with top-K gates g_j (router order): predicted energy E_hat_j = g_j^2 * alpha_{e_j}; prefix sums S_m.
- k_i = min{ m in [1,K] : 1 - S_m / (S_K + zeta) <= epsilon } (Eq. 14). Execute the first m* experts in router order.
- App. A.5 justifies this "Under a weak-correlation approximation among skipped expert outputs", i.e. cross terms between experts are ignored.

### How it weights the remaining experts
- Static pruning: native router transformation and renormalisation are kept over the surviving experts (App. C.4: "we preserve each architecture's native router transformation and renormalization rule").
- OMP-MoE†: keeps the original gates, no renormalisation: Alg. 3 returns "y = sum_{j=1}^{m*} g_j F_{e_j}(x)" and Sec. 3.4 says "we execute only the first k_i experts and treat the contributions of the remaining candidates as zero". In our notation: c = g_S with S = router-order prefix.
- Explicit rejection of least squares (App. A.3, point 2): "The contribution of an expert V_e is invariant to the selection of future experts, preserving the physical interpretation of the MoE routing mechanism where expert weights are determined by the router, not by a post-hoc least-squares fit." Also App. A.2: "Unlike standard OMP, which requires an orthogonal projection of the signal onto the subspace spanned by the selected atoms (typically involving the inversion of the Gram matrix), our binary constraint simplifies the update dynamics significantly."

### Models and headline numbers (copied exactly)
- Models: Qwen3-30B-A3B (128 experts, top-8), DeepSeek-V2-Lite (64, top-6, shared experts), GPT-OSS-20B (32, top-4), Mixtral-8x7B (8, top-2); scaling check on Qwen3-235B-A22B.
- Table 1 (6-task average, 0-shot): Qwen3-30B-A3B Original 0.640; OMP-MoE 0.633 at 25% pruning, 0.597 at 50% ("preserves 93.3% of the original 0.640", Sec. 4.2).
- Table 15 (8-task average): Qwen3-30B-A3B Original 0.682; OMP 0.676±0.005 (Num=96), 0.643±0.005 (Num=64). DeepSeek-V2-Lite Original 0.609; 0.576±0.005 (48), 0.489±0.005 (32). Table 16: GPT-OSS-20B 0.574 -> 0.565 / 0.515; Mixtral 0.675 -> 0.649 / 0.607.
- Table 2 (search time, 50%): Qwen3-30B-A3B OMP-MoE 641 s vs NAEE 21301 s vs DiEP 8972 s.
- Table 4 (Qwen3-30B-A3B, 50%): GSM8K 0.531 (Original 0.891), LongBench 0.523 (0.578), BBH 0.323 (0.363).
- Table 7 (OMP-MoE†, epsilon = 0.1): mean executed experts E[m*|eps]: Mixtral 1.22 of 2; DeepSeek-V2-Lite 3.90 of 6; Qwen3-30B-A3B 5.14 of 8 (normalised routed FLOPs 0.643, 35.8% gain).
- Table 20 (DeepSeek-V2-Lite, 25% pruned): eps 0 -> 6 experts, acc 0.576; eps 0.1 -> 3.90 experts, acc 0.581; eps 0.3 -> 2.42 experts, acc 0.554.
- Table 23 (Qwen3-30B-A3B): 50% static 0.643 at 1.49x; with OMP-MoE† 0.641 at 1.55x.
- Table 29: reconstruction error E vs accuracy, linear fit A ≈ 0.4170 + 0.2533 phi, R^2 ≈ 0.947.

### Novelty gate
- (a) YES, partially. It computes Frobenius inner products between the residual and gated expert contributions over calibration tokens (<R_t, V_e>, ||V_e||^2), which are inner products of expert OUTPUTS (gated, token-concatenated). It does not form or store an explicit Gram matrix G_ij between experts, and at inference it uses only diagonal energies. Quote: "Delta_e = 2<R_t, V_e> - ||V_e||_F^2 ... where <.,.> denotes the Frobenius inner product."
- (b) NO. Coefficients are binary; static pruning keeps native renormalisation; OMP-MoE† keeps raw gates. Quote above ("not by a post-hoc least-squares fit").
- (c) NO for per-token span. Closest items: per-token count m* from the diagonal energy proxy (Eq. 14), and per-layer "normalized residual curves" (App. E.2) as a function of number of selected experts. Neither is a span-based per-token measure.
- Overlap to flag in our related work: name ("OMP"), greedy pursuit on output inner products, energy-threshold adaptive stop at inference. Our differences: per-token selection within T, real-valued least-squares coefficients (true OMP/OLS projection), use of full Gram including cross terms at inference.

### Open questions for reimplementation
- Whether the inference-time alpha_e is computed on the pruned or unpruned model (verify; text says offline calibration).
- Exact definition of "Calibration Samples per Layer: 64 samples" vs 64 blocks of 4,096 tokens (App. C.2 says 64 blocks).
- Code "available in Supplementary Material"; no public URL found (verify).

---

## 2. "k2 fix" (priority 2)

- Exact title: "Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models"
- Authors: Xing Chen, Hengshuai Yao
- Date: 4 September 2026 (v1). arXiv 2609.04575 [cs.LG, cs.AI].
- What it changes: the renormalisation denominator only, when k is reduced at inference.
- Training-free: yes. "All results are obtained without any training, distillation, or fine-tuning."

### Exact rule (Eq. 2, Sec. "Method: decouple the reference set")
Router p = softmax(W_g x) over all E experts; T_k = indices of the k largest entries.
"w_i = p_i / sum_{j in T_{k2}} p_j , i in T_{k1}", with T_{k1} ⊆ T_{k2} when k2 >= k1.
- Expert choice: S = T_{k1}, the top-k1 experts by router probability (no other criterion). "We keep the selection rule and the numerator, and replace only the denominator's index set."
- Special cases quoted: "k2 = k1: standard renormalization (Eq. 1); k2 = k (trained value): the gain is anchored to what training calibrated; k2 = E: no renormalization at all, since sum_j p_j = 1; k2 > k: gain below the trained value, applied per token."
- Hyperparameter: one integer k2. Recommendation: "scan k2 in {k1, k, 2k} with paired testing". Best found: Qwen3.6-35B-A3B prefers k2 = 2k = 16 (MMLU) but k2 = 8 for perplexity; Qwen3.5-397B-A17B prefers k2 = k = 10.

### The rule in our notation
Let m_j denote the router softmax mass of a set. Native gates g_i = p_i / m(T) for i in T, so a + b = 1. S = top-k1 of T, D = T \ S, a = sum_S g = m(S)/m(T), b = sum_D g.
k2 rule: c_i = p_i / m(T_{k2}) = g_i * m(T) / m(T_{k2}) for i in S; c_i = 0 for i in D.
- k2 = k1: c_i = g_i / a = g_i (a+b)/a. This is exactly "renormalise".
- k1 < k2 < k: c_i = g_i (a+b) / (a + b'), with b' = sum of g over router ranks k1+1..k2 (a subset of D). Interpolates between renormalise and keep.
- k2 = k: c_i = g_i. This is exactly "keep gates" (c = g_S).
- k2 > k: c_i = g_i / (1 + rho), rho = m(T_{k2} \ T) / m(T) > 0, i.e. router mass of ranks k+1..k2 measured relative to the top-k mass. Shrinks below keep; needs router probabilities outside T (available, since p is computed over all E).
- k2 = E: c_i = p_i = g_i m(T) (raw softmax, no renormalisation).
So every member of the family is c = lambda(x) g_S with a per-token scalar lambda(x) = m(T)/m(T_{k2}) that depends only on router probabilities; direction always equals g_S; expert outputs are never used.
Our own derivation (not in the paper), for placing it against least squares: c_LS = G_SS^{-1} G_ST g = g_S + G_SS^{-1} G_SD g_D. Keep (k2 = k) equals c_LS iff G_SD g_D = 0 (the dropped gated output is orthogonal to every kept expert output). Renormalise (k2 = k1) equals c_LS iff G_SS^{-1} G_SD g_D = (b/a) g_S. The k2 > k regime (shrinking below keep) has no counterpart in the projection, which never shrinks along g_S on the basis of router mass.

### Models and headline numbers (copied exactly)
- Models: Qwen3.6-35B-A3B (40 layers, 256 experts, k = 8, 1 shared expert; top-8 router mass 0.182, renorm gain 5.50x), Qwen3.5-397B-A17B (60 layers, 512 experts, k = 10; mass 0.192, gain 5.22x) (Table 1). Preliminary Gemma-4-26B-A4B in App. A under a different scalar-gain parameterisation.
- Table 5 (MMLU, 5-shot, n = 2000, paired McNemar): 35B native 81.65%; k1=6,k2=6 79.05% (-2.60); k1=6,k2=8 80.55% (-1.10); k1=6,k2=16 82.40% (+0.75); k1=6,k2=256 54.20% (-27.45); k1=4,k2=4 77.00% (-4.65, p = 1.7e-9); k1=4,k2=8 78.55% (-3.10); k1=4,k2=16 81.30% (-0.35, p = 0.66). 397B native 89.45%; k1=5,k2=5 87.35% (-2.10); k1=5,k2=10 88.90% (-0.55, p = 0.24); k1=5,k2=512 70.90% (-18.55).
- Table 6 (GSM8K, 0-shot, n = 500): 35B native 95.00%; k1=4,k2=4 88.40%; k1=4,k2=16 94.40%. 397B native 96.60%; k1=5,k2=5 92.40%; k1=5,k2=10 95.80%.
- Table 4 (perplexity, WikiText): 35B native 6.643; k1=6,k2=6 +4.08%; k1=6,k2=8 -1.61%; k1=6,k2=256 +356%.
- Table 2 (fixed 8 experts, WikiText PPL): native 6.64; uniform weights 9.95 (+49.7%); sampled 27.22 (+310%); random 117.39 (+1667%).

### Novelty gate
- (a) NO. Only router probabilities. Their "selection matters more than weighting" ablation uses uniform weights, not output geometry.
- (b) NO. The weights are a per-token scalar rescaling of router probabilities; no optimisation of combination weights. Quote: "introducing one integer with no parameters, training, or measurable compute overhead."
- (c) NO.
- Relevance: strongest baseline family for our weighting comparison; we should include k2 in {k1, k, 2k} as baselines, and note their finding that perplexity and MMLU pick different k2.

### Open questions for reimplementation
- Shared expert gate g_sh left unchanged (Eq. 1 shows it separately; assumed untouched).
- Exact MMLU subset (2000 questions, fixed seed) not released (verify code availability).

---

## 3. SERE (priority 3)

- Exact title: "SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models"
- Authors: Juntong Wu, Jialiang Cheng, Fuyu Lv, Ou Dan, Li Yuan (Taobao & Tmall Group of Alibaba; Peking University Shenzhen)
- Date: 7 February 2026 (v1). arXiv 2602.07616 [cs.LG, cs.AI]. Abstract page notes publication at ICLR 2026. Code: github.com/JL-Cheng/SERE.
- What it changes: per decoding BATCH, reduces the union of active experts by re-routing tokens' secondary experts to similar primary experts.
- Training-free: yes ("requiring no retraining or task-specific tuning"); needs a calibration pass for the similarity matrix.

### How it chooses experts (Sec. 3.2.2, Alg. 2)
- Similarity matrix per layer (Eq. 1, Alg. 1): run all M experts on calibration inputs, A_j = E_j(X) (ungated, every expert on every token), S_{p,q} += Sim(A_p, A_q), averaged over N batches. Default Sim is normalised Frobenius similarity (App. A.2.2): x_{E,F} = ||X_E - X_F||_F, M_fro = 1 - x_{E,F}/max(x). Alternatives: cosine (mean per-token cosine), CKA (linear, RBF, polynomial).
- Step 1: primary set E_p = union over tokens in the batch of each token's top-S ranked experts, S in [1, K) (S is called "top1"/"top2" in tables).
- Step 2: for each secondary expert u (in some token's top-K, not in E_p), sim*_u = max_{v in E_p} S_{u,v}; if sim*_u >= rho, re-route all tokens assigned to u to v*_u; else keep u as a "critical expert".
- Step 3: final active set = E_p ∪ {critical secondary experts}.
- Hyperparameters: S in {1, 2}; rho = 0.0 or 0.3 (Qwen1.5, DeepSeek-V2-Lite), 0.5 (Qwen3-30B-A3B); calibration FineWeb-Edu 400 sequences x 128 tokens; batch size 16; QPS 16; NVIDIA H20.

### How it weights the remaining experts
- Keeps router weights: "It should be noted that the re-routing process does not modify the router weights." The re-routed slot keeps the gate of the skipped expert u but evaluates expert v*_u. In our notation, for token x: output = sum_{i in T} g_i E_{pi(i)}(x) with pi(i) = i or a substitute primary expert. If v*_u is already in the token's own top-K, the two gates effectively add on the same expert (inferred from Alg. 2, which only rewrites indices; verify).

### The numbers we believed: 82.24 / 80.37 / 64.11 (CONFIRMED)
- Table 3: "OpenCompass and TPOT (QPS=16) results on Qwen3-30B-A3B." Column "Avg. (Acc.)" = average of 8 tasks: cmmlu, boolq, bbh, math, gsm8k, math401, humaneval (heval), mbpp; CoT mode for CMMLU and BoolQ; sampling with temperature 0.7, top-p 0.8, top-k 20, max 2048 output tokens (App. B, Table 8).
  - Qwen3-30B-A3B top8 (native): Avg 82.24, TPOT 44.40 ms.
  - SERE_{top2; rho=0.5}: Avg 80.37, TPOT 32.82 ms.
  - SERE_{top1; rho=0.5}: Avg 64.11, TPOT 33.10 ms.
  - For context, same table: plain top2 14.30; plain top1 8.60; SERE_{top2; rho=0.0} 77.82; SERE_{top1; rho=0.0} 52.27; LYNX top2 78.46; LYNX top1 57.52; HC-SMoE_{80 experts} 69.42; HC-SMoE_{48 experts} 46.57.
- Caveat for citing: "top2"/"top1" in SERE means S primary experts per token in a batch-level union, plus re-routed and critical experts. The per-token count of distinct experts evaluated is NOT 2 or 1, and plain top-2 here collapses to 14.30, which is far below what renormalised top-2 typically gives in other papers (possibly due to sampling generation and CoT). Do not equate SERE "top2" with our |S| = 2.
- Other tables: Table 1 Qwen1.5-MoE-A2.7B top4 48.52; SERE top2 rho=0.3 47.25; SERE top1 rho=0.3 39.82. Table 2 DeepSeekV2-Lite top6 54.76; SERE top2 rho=0.3 55.48; SERE top1 rho=0.3 55.62.

### Novelty gate
- (a) YES. Similarity of expert outputs on calibration data, accumulated per layer (Eq. 1; Frobenius distance, cosine of outputs, CKA). Quote: "Using a calibration dataset D_calib ... expert activations are obtained as A_j = E_j(X), after which pairwise similarities are computed via a predefined similarity function Sim". The Frobenius distance is a function of Gram entries (||A_p - A_q||^2 = G_pp + G_qq - 2 G_pq), but they normalise and threshold it rather than use G.
- (b) NO. "the re-routing process does not modify the router weights."
- (c) NO.

### Open questions for reimplementation
- Whether the similarity uses gated or ungated outputs: Alg. 1 uses E_j(X^(l-1)) for all experts, i.e. ungated, all tokens.
- Duplicate-expert handling in the vLLM kernel (verify in code).

---

## 4. Elbow-based routing (priority 4)

- Exact title: "Elbow-Based MoE Routing: A Training-Free Inference Time Plugin for Expert Selection" (HTML capitalises "Elbow-based")
- Authors: Robin Pan, Raymond Liu, Daniel Fang, Adelina Andrei, Rosa Wu (Harvard)
- Date: 5 August 2026 (v1). arXiv 2608.04401 [cs.LG]. Code: github.com/rpan188/elbow-routing.
- What it changes: per-token number of experts.
- Training-free: yes.

### How it chooses experts (Sec. 2, Alg. 1)
- Sort full softmax p over N experts descending; normalise p'_i = (p_i - p_min)/(p_max - p_min + eps), x'_i = i/(N-1); D_i = p'_i - x'_i; e = argmax_i D_i; elbow k = e + 1 (Kneedle approximation).
- k(x) = min(e(x), K). Keeps the top-k(x) experts in router order. No hyperparameter besides K.

### How it weights the remaining experts
- Not stated in the paper. From the released code (scripts/evals.py, read via raw GitHub): kept weights are the raw sorted softmax values masked beyond k, renormalised only if the block has norm_topk_prob truthy: "if getattr(self, "norm_topk_prob", False): denom = routing_weights.sum(...); routing_weights = routing_weights / denom". OLMoE is reported elsewhere (2610.41366, Table 2) as trained without top-k renormalisation, so effectively c = g_S with g the raw softmax (keep gates). The norm_topk_prob value for OLMoE-1B-7B-0924-Instruct itself is (verify).

### Models and numbers (Table 2, OLMoE-1B-7B-0924-Instruct, 64 experts, top-8)
- ARC-Easy 77.82 -> 78.24 (k-mean 7.623); ARC-Challenge 61.86 -> 62.29 (7.634); MMLU 51.74 -> 51.75 (7.613); HellaSwag 47.02 -> 46.86 (7.655); PIQA 73.18 -> 72.85 (7.589); WinoGrande 50.36 -> 50.28 (7.576). Overall k-mean 7.615, latency -5.3%.
- Load-balance bound ||q_elb - q_topK||_1 <= 2 delta/(1 - delta), delta = 0.05 empirically.

### Novelty gate
- (a) NO (router probabilities only). (b) NO. (c) NO; its "how many experts are meaningfully engaged" is the elbow of router probabilities: "providing a token-specific estimate of how many experts are meaningfully engaged by the router."

### Open questions
- Only ~5% reduction; tiny regime. Single model.

---

## 5. EAT (priority 5)

- Exact title: "EAT: Expert Account Tracker for Efficient MoE Inference"
- Authors: Yuexian Li, Yifei Yang, Zouying Cao, Hai Zhao (Shanghai Jiao Tong University; Ant Group)
- Date: 27 September 2026 (v1). arXiv 2609.33614 [cs.AI].
- What it changes: per-token adaptive set via history-aware scores and adaptive threshold.
- Training-free: the routing rule is training-free, but the paper's best rows use "OPD", a fine-tuning of router weights on 9K samples (learning rate 5e-5, 250 steps), so "EAT + OPD" is NOT training-free.

### How it chooses experts
- History score (Eq. 1): S_h,i = w1 N_i/sum N + w2 P_total,i / N_i + w3 C_s,i^(t), with C_s^(t) = (1 - alpha) C_s^(t-1) + alpha ||o||_2 (EMA of output L2 norm).
- Importance (Eq. 2): I_i = (1 - beta) p_i + beta S_h,i.
- Threshold (Eqs. 3 to 5): tau = max(alpha1 mu_w, mu_w - alpha2 sigma_w) over raw gates; perplexity-based adjustment tau x 1.2 if PPL rose, else tau x 0.9; bounded to [10%, 90%] of the routing-weight distribution; smoothing tau_final = kappa1 tau_PPL + kappa2 tau.
- Selection (Eq. 6): E = {i : I_i > tau_final}; if empty take argmax I; if > K keep top-K by I.
- Hyperparameters (Table 1): alpha = 0.95; w1,w2,w3 = 0.3,0.3,0.4; beta = 0.6; alpha1,alpha2 = 0.5,1.0; kappa1,kappa2 = 0.7,0.3; K = 2 (Mixtral, Phi), 8 (Qwen).
- Note: the selected set can include experts outside the router's top-K, since I_i mixes in history; the text does not restrict candidates to T (verify).

### How it weights the remaining experts
- Renormalise (Eq. 7): g_norm,i = g_i m_i / (sum_i g_i m_i + eps), eps = 1e-8. In our notation c_i = g_i/a (when S ⊆ T).

### Models and numbers (Tables 2 and 3, OpenCompass)
- Qwen3-30B-A3B MMLU: Vanilla 80.22; Top-P 77.85; EAT 73.96; EAT+OPD 79.65. Mean experts on MMLU: 8.00 / 6.75 / 5.50 / 6.10.
- Mixtral-8x7B MMLU: Vanilla 71.67; Top-P 65.38; EAT 70.37; EAT+OPD 71.59; experts 2.00 / 1.51 / 1.47 / 1.69.
- Claimed ~25% reduction in activated experts.

### Novelty gate
- (a) NO (output norm EMA only, no cross terms). (b) NO (renormalise). (c) NO.

### Open questions
- Perplexity-based threshold adjustment at inference uses "real-time model perplexity", which needs labels or the model's own tokens; unclear how computed during benchmark scoring (verify).

---

## 6. Other close papers found by search

### 6.1 Pruning and Distilling Mixture-of-Experts into Dense Language Models (CLOSEST on Gram of outputs)
- arXiv 2605.28207 (read v2, dated 6 June 2026 on arXiv; paper date line "August 24, 2026"). Authors: Junhyuck Kim, Jihun Yun, Haechan Kim, Gyeongman Kim, Joonghyun Bae, Jaewoong Cho (KRAFTON, KAIST).
- Changes: converts an MoE into a dense FFN (score, select K experts per layer, group/merge, concatenate, then knowledge distillation ~4B tokens). NOT training-free overall (selection step is training-free).
- Selection, D-optimal (Eq. 4): "Given base importance I_e = s^(e) ... and the expert output Gram matrix G_ij = E_t[<f_i(t), f_j(t)>], we form the importance-weighted kernel K_ij = sqrt(I_i I_j) G_ij and select S* = argmax_{|S|=K} log det(K_S + lambda_reg I)", lambda_reg = (1/(KE)) sum_e K_ee, solved greedily with Schur complements, O(K^3 E).
- Weighting: static down-projection scaling alpha_g = 1/k (uniform) or alpha_g proportional to summed scores (Sec. 3.4): "The concatenated dense FFN has no router, so we apply static scaling factors alpha_g". No least squares.
- Diversity measure: "We measure the diversity of a selected expert set by its effective rank (Roy and Vetterli, 2007): the number of significant independent directions in the expert output space" (Fig. 3: Qwen3-30B-A3B, D-optimal raises effective rank from 6.07 to 7.37 (CP) and 6.31 to 6.93 (ACP), max 8).
- Gate: (a) YES, exactly our G (calibration-averaged inner products of expert outputs). (b) NO. (c) PARTIAL: effective rank of a selected K-expert kernel per layer, not per token and not with gate-weighted oracle reconstruction. Must be cited as prior use of the output Gram and of an effective-rank diversity measure.

### 6.2 HOPE: Higher-order pruning of experts in mixture-of-experts language models
- arXiv 2609.18916 (read v3, 29 September 2026). Authors: Alex M. Tseng, Prannay Kaul, Luca Zancato, Wei Xia, Stefano Soatto (AWS). Static pruning, training-free.
- F_ij = (1/N) sum_{x: i,j in T(x)} g_i(x) g_j(x) ||f_i(x)||_2 ||f_j(x)||_2 (Eq. 3); prune set P* = argmin p^T F p, binary QP, solved by continuous relaxation and rounding. Refill with renormalised gates; renormalisation error argued small.
- Gate: (a) PARTIAL: products of output NORMS over co-activations, not inner products (no direction information). (b) NO. (c) NO.

### 6.3 RAZOR: Pruning Replaceable Experts in LLMs
- arXiv 2609.30465 (read v4, 30 September 2026). Authors: Mingyang Song, Mao Zheng (Tencent). Static pruning, training-free.
- Per-token deletion damage with refill (Prop. 2): delta_i(x) = ||w_i r_i - w_r r_r||_2 / D_i, r_j = f_j - c (consensus residual), aggregated by conditional RMS. Diagnostic "interaction Gram matrices" G_ij = sum_x <u_i(x), u_j(x)>, u_i = w_i(f_i - c) (App. C.7.2).
- Weighting: survivors renormalised with router refill.
- Gate: (a) YES (per-token output geometry; Gram of weighted residuals as diagnostic). (b) NO. (c) NO.

### 6.4 MoRA: MoE Pruning via Router Bias Learning and Expert Approximation
- arXiv 2610.00367 (v1, 30 September 2026). Authors include Y. Sun, Z. Zhou, L. Gao, J. Yu, W. Pei (Harbin Institute of Technology, Shenzhen) per the 2610.41366 reference list; full author list (verify). Static pruning; router biases LEARNED with LM loss, so NOT training-free.
- Expert approximation: each pruned expert i is replaced by a retained proxy j with an affine map, N_i(x) = a ⊙ E_j(x) + b, "used only if that R^2 exceeds 0.4" (as described in 2610.41366, Sec. 3). Abstract: "It leverages the remaining experts to approximate the outputs of pruned experts by affine transformation".
- Gate: (a) YES, one-to-one regression of a pruned expert's output on one survivor's output. (b) PARTIAL: a fitted channel-wise affine map from ONE survivor, static, not a least-squares combination of several kept experts per token. (c) NO. Only abstract and secondary description read in detail (verify fitting formula in Sec. 3 of 2610.00367).

### 6.5 What We Keep of What We Forget: Replacement Operators and Routing Semantics in Training-Free MoE Expert Pruning
- arXiv 2610.41366 (v1, 30 October 2026). First author affiliation NUST (E. Ali et al.; full author list verify). Static pruning, training-free.
- Fixed-mask comparison of replacement operators: MoNE constant, "tilted novice" (rank-r reduced-rank ridge regression of E_i(x) on the input x, r = 32, ridge 1e-2), MoRA proxy, ConMoE slot remap, keep-scale vs renormalised rerouting, COMPEL weight compensation.
- Key finding relevant to our weighting baselines: "At 50% pruning the better choice, on all four routers, is the one the router was trained with; we call this pattern the trained-scale rule." Routers trained without renormalisation (OLMoE, DeepSeek-V2-Lite) prefer keep; routers trained with it (Moonlight, Granite) prefer renormalise. Consistent with the k2 fix paper.
- Gate: (a) YES in an exploratory QP: "every pair of co-routed pruned experts' residual inner products" (App., line on per-layer QP). (b) PARTIAL: closed-form regression, but of a pruned expert's output on the INPUT, not on kept experts' outputs. (c) NO.

### 6.6 OptEP: Reconstruction-Optimized Expert Pruning for Mixture-of-Experts Language Models
- arXiv 2610.27123 (v1, 19 October 2026). Authors: Jelena Markovic-Voronov, Petros Prastakos, Kayhan Behdin, Jincheng Cao, Zhipeng Wang, Shayan Mohajer Hamidi, Yuanda Xu, Zhengze Zhou, Wenhui Zhu, Rahul Mazumder (LinkedIn, MIT, Google). Static pruning.
- Selection: relax z in [0,1]^E inside renormalised router scores pi_e = z_e a_e / sum_j z_j a_j (Eq. 4), minimise layer reconstruction loss ||y - y~||^alpha by SGD (50 steps, lr 0.02), keep top-s by z.
- Weighting: renormalised router scores; then EWU: Adam updates of retained experts' PARAMETERS against reconstruction, regularised to pretrained values. Gradient-based post-training optimisation, arguably not training-free.
- Gate: (a) only implicitly via reconstruction loss. (b) NO for combination coefficients (z is discretised; EWU changes expert weights). (c) NO.

### 6.7 You Only Need 2/3 of the Chosen Experts (empirical study of dynamic expert pruning)
- arXiv 2609.25809 (v1, 22 September 2026). Yuanteng Chen, Qiwei Lai, Chen Tianqi, Peisong Wang, et al. (CASIA, UCAS, USTC).
- Retains a subset S of selected experts with renormalisation (Eq. 2: w~_i = w_i / sum_{j in S} w_j). Headline: "uniformly retaining approximately two thirds of the selected experts preserves 98.8% of unpruned performance on average".
- Gate: (a) NO, (b) NO (renormalise), (c) NO. Useful as the Fixed-K baseline reference.

### 6.8 ACE: Adaptive Calibration-Free Expert Skipping for MoE-based LLMs
- arXiv 2609.05228 (v1, 4 September 2026). Zukang Xu, Zhixiong Zhao, Xing Hu, Jiangyong Yu, Houji Wen, Jun Li, Zhe Jiang, Dawei Yang. Training-free, calibration-free.
- Skips slots flagged low-contribution by both a weight-spectral proxy (GSP) and router-direction responses (RCR); always keeps top-1. Weighting: renormalise, "the retained gate is g^_i = g_i/(1 - alpha)".
- Gate: (a) NO, (b) NO, (c) NO. Headline: "at a 50% skipping ratio on Qwen3.6-35B-A3B, ACE reduces WikiText-2 perplexity by 7.96% and improves average downstream accuracy by 4.15 percentage points over the strongest competing method."

### 6.9 Don't Skip the Baseline: A Controlled Comparison of Dynamic Expert Skipping in MoEs
- arXiv 2610.28369 (v1, 20 October 2026). Steimle et al. (Tensordyne, TUM, ETH, Ulm; full author list verify).
- Finds static k reduction hard to beat at matched measured compute (Gemma4-26B-A4B, Qwen3-30B-A3B, Qwen3-VL). Taxonomy: NAEE (gate cumulative-mass threshold), DiEP (gate weights + inter-expert CKA redundancy), MoDES (gate x layer importance, per-modality threshold), Ada-K (learned allocator, PPO).
- Gate: (a) NO for the paper itself; it reports that DiEP (arXiv 2509.16105) uses CKA between experts for skipping (verify DiEP details directly; not read). (b) NO. (c) NO. Important as a methodological warning: compare at matched measured k, not calibrated k.

### 6.10 Mentioned but not read in full
- MoE Lens (arXiv 2603.05806, 6 March 2026; Chaudhari, Gulati, Hundia, Karra, Raval): abstract only; "top-weighted expert's output closely approximates the full ensemble prediction" (cosine similarity up to ~0.95 in some layers, DeepSeekMoE). Analysis paper; no reweighting stated (verify full text).
- PCoMoE (arXiv 2609.01024, EMNLP 2026): abstract only; "compatibility-aware layer-wise pruning" of path combinations; mechanism unknown (verify).
- NAEE (arXiv 2402.14800, ACL 2024): static pruning by reconstruction-loss subset search plus dynamic skipping by gate ratio; known, not re-read here.
- DiEP (arXiv 2509.16105): not read; described by 2610.28369 as gate weights plus CKA redundancy (verify).
- ConMoE (arXiv 2605.29350), COMPEL (ACL Findings 2026), PRIME (arXiv 2610.13982, closed-form router refit after pruning/merging), LExI (arXiv 2509.02753), Lv et al. self-distillation (arXiv 2605.18643): cited by the papers above; not read (verify if needed). PRIME's "closed-form" router refit is worth a check, since it is a closed-form fit, although on router weights rather than combination coefficients.

---

## 7. Search log and coverage limits
- Web searches (extended mode) run for: least squares reweighting after skipping; expert-output Gram projection; matching pursuit MoE; least-squares compensation of surviving experts; expert output correlation with dynamic skipping and reweighting; "effective number of experts" with participation ratio; refit / re-solve gates to reconstruct full top-k output.
- Reference lists of 2610.41366 and 2609.04575 were mined for further candidates.
- Semantic Scholar API returned HTTP 429 (rate limited) on two queries; arXiv export API also returned "Rate exceeded". Google Scholar was not queried directly. Coverage is therefore based on web search plus citation chasing; a paper that uses least-squares combination weights of kept experts under an unusual name could have been missed (verify with a later Semantic Scholar / Scholar pass).


---

# Part 2 (full reading notes)

# Literature check, part 2: training-free reduction of active experts in MoE LLMs

Compiled 10 November 2026. Sources read: arXiv abstract pages, arXiv HTML full texts, two public GitHub repositories (SMoE, NAEE), and the PDFs of Tropp 2004 (author preprint) and Das and Kempe (arXiv 1102.3975v2). Text was read through a fetch tool that sometimes paraphrases; anything I could only see paraphrased, or could not see at all, is marked (verify). Short quotations below are verbatim as returned by the fetch tool or by text extraction from the PDF.

Our method, for reference: per token, router top-k set T with gates g; keep subset S; refit kept weights c = G_SS^{-1} G_ST g from a calibrated Gram matrix G of expert OUTPUTS; choose S by greedy pursuit (OLS/OMP) on G; measure effective number of experts via the span of expert outputs.

Overlap legend for each paper:
- (a) estimates inner products / correlations between expert OUTPUTS?
- (b) uses them to solve for weights of a reduced expert set?
- (c) measures effective number of experts via span of expert outputs?

---

## Overlap summary (read this first)

| # | Paper | (a) output inner products | (b) refit weights of reduced set | (c) effective number via span |
|---|---|---|---|---|
| 1 | SMoE (2508.18983) | No | No | No |
| 2 | CCE (2412.00099) | No | No | No |
| 3 | MoE Lens (2603.05806) | Only cosine of single-expert vs ensemble hidden state (incl. residual); no expert-expert inner products | No | No |
| 4 | Su and Liu (2601.03577) | Yes, in theory (mutual coherence, Gram, DPP kernel, orthogonality loss) | Only as the inner OLS step in proofs; not used at inference | No ("Effective Rank" of expert feature space appears in a figure, definition not found) |
| 5 | LightMoE (2603.12645) | No (cosine of routing logits) | No (trained) | No |
| 6 | BuddyMoE (2511.10054) | Claimed "output similarity" in intro; no formula found | No | No |
| 7 | MoE-Spec (2602.16052) | Oracle only: greedy reconstruction error vs unpruned output, gates fixed | No | No |
| 8 | NAEE (2402.14800) | No (layer output reconstruction loss for pruning only) | No | No |
| 9 | Dynamic top-p (2403.07652) | No | No | No |
| 10 | Tropp 2004; Das and Kempe 2011 | Theory of greedy sparse approximation on Gram / covariance | Yes (OLS coefficients C_S^{-1} b_S) | No |

Main overlap concerns:
1. arXiv 2601.03577 (Su and Liu) is the closest conceptual neighbour. It explicitly casts routing as sparse subset selection over the matrix of expert outputs, min ||y - E alpha||_2^2 s.t. ||alpha||_0 <= k, uses the OLS normal equations alpha_S* = (E_S^T E_S)^{-1} E_S^T y inside proofs, and states a coherence condition mu(E) < 1/(2k-1) for greedy (Tropp-style). However: it is training-time theory (orthogonality regulariser), y is "the optimal target representation" (not the dense top-k output), it does not reduce active experts at inference, does not calibrate a Gram matrix, and does not refit gates. It must be cited and differentiated. Its "if and only if" coherence claim and its DPP "monotone submodular" claim look overstated (see block 4).
2. MoE-Spec's oracle (Eq. 6) is greedy forward selection minimising ||O*_t - sum_j g_j E_j||^2 against the unpruned output, but with gates FIXED at router values (no refit), per layer over draft-tree tokens, and labelled impractical. Worth citing as "greedy output-reconstruction selection without weight refit".
3. NAEE expert pruning minimises layer-output reconstruction error by enumeration (no refit); its dynamic skipping is gate-ratio only, with renormalisation to 1.
4. No paper in this batch computes a calibrated Gram matrix of expert outputs to refit kept gate weights per token (c = G_SS^{-1} G_ST g), and none measures effective number of experts via span of outputs. Within this batch the (b) and (c) contributions appear novel; this does not cover papers outside the batch.

---

## 1. SMoE: expert substitution for edge offloading (arXiv 2508.18983)

- Exact title: "SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution"
- Authors: Guoying Zhu, Meng Li, Haipeng Dai, Xuechen Liu, Weijun Wang, Keran Li, Jun Xiao, Ligeng Chen, Wei Wang
- Dates: v1 26 Aug 2025; v2 19 Nov 2025; v3 3 May 2026 (current). ID verified.
- Code: https://github.com/goingshr/SMoE (listed in artifact appendix). Config keys `replaceScoreRatio` ("Ratio about experts replaced"), `if_replace`, `window_size`.

What it changes: replaces low-importance activated experts that are not in GPU memory with lower-ranked experts that are already cached (or will be loaded anyway), plus score-aware eviction, prefetch of next-layer top-score experts, and CPU/GPU load splitting.

Training-free: yes (no parameter updates; alpha is chosen offline by a latency-accuracy search).

How it chooses experts (paper, Sec. IV, Algorithm 1). Notation: Score(e) is the gate score; S_{k+1} is the score of the (k+1)-th highest expert; alpha is the substitution threshold.
- Top-score (always computed): Score(e) > (1+alpha) S_{k+1}.
- Low-score activated experts (substitution candidates, E_l): S_{k+1} < Score(e) < (1+alpha) S_{k+1}.
- Substitute pool (E_s): (1-alpha) S_{k+1} < Score(e) <= S_{k+1}, and the expert is GPU-resident or in the top-score set.
- Paper: "importance to an input is reflected in its router's gate score" (Sec. III). Justification for similarity is conceptual: "routing redundancy and noisy gating in sparse MoEs cause tail experts with marginal scores to behave similarly." No measured output similarity.
- Algorithm 1: if |A| >= |B| (A = E_s, B = E_l) pick the top |B| from A; otherwise use all of A and fill the remainder from B (loaded).
- Objective (Eq. 1): maximise |G cap E_a| + min(|E_l \ G|, |E_s|), G = GPU-resident set, E_a = top-k set.
- alpha values: "we set the substitution threshold to 0.35 in S1, 0.3 in S2, and 0.25 in S3" (S1 deepseek-moe-16b-base, S2 XVERSE-MoE-A4.2B-Chat, S3 Qwen2-57B-A14B-Instruct). alpha chosen by Eq. 5: minimise accuracy loss A(alpha) s.t. latency T(alpha) <= R, T fitted by low-degree polynomial, 1-D search.
- Eviction (Eq. 3): evict argmin of mean score over the last n iterations; n not given in paper (code: `window_size`, null means LRU) (verify value).

Code-level rule (utils/expertcache.py, functions `replaceset_between_tokens` and `cache_router`; quoted lines):
- `midscore = sort_scores[token_id][topk]` (0-indexed, so the (k+1)-th score)
- `lscore = midscore+a*midscore` (upper threshold)
- `rscore = midscore-a*midscore` (lower threshold)
- `replaceset_between_tokens`: top-k experts with score >= upper threshold go into `replaceset` (as the name suggests, apparently pooled across tokens in the batch, i.e. experts that will be computed anyway) (verify pooling).
- `cache_router`, per token: (i) top-k experts with score >= upper threshold are kept; (ii) for each other top-k slot, keep the expert if it is in GPU cache or in `replaceset`; (iii) else look for a substitute among experts with score strictly between `rscore` and `midscore` (iterated as a Python set, so not in score order) that is in GPU cache or in `replaceset` and not already assigned to the token; (iv) if none, keep the original and mark for loading.
- Discrepancy: paper says Score(e) <= S_{k+1} for E_s; code summary says strictly below midscore, which would exclude the (k+1)-th expert itself (verify in code). Paper says pick "highest-scoring alternatives"; code iterates a set (verify).

How substituted experts are weighted (code, MoEModule/SMoE_base.py, `run_with_cache`):
- `routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)`
- after substitution: `topk_weight = routing_weights[torch.arange(routing_weights.size(0)).unsqueeze(1), topk_idx]` with `topk_idx` = substituted expert list, so a substitute receives its OWN softmax probability (not the replaced expert's).
- then `if self.get_norm_topk_prob(): topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)`; whether this flag is True per model is set in subclasses (presumably following each model's `norm_topk_prob`; Qwen2-MoE config sets it False by default, DeepSeekMoE 16B does not renormalise) (verify).
- The paper itself does not state the substitute's weight.

Per-token reimplementation sketch (our reading of paper + code):
```
p = softmax(router_logits)                    # over all experts
order = argsort(p, desc); s_ref = p[order[k]] # (k+1)-th probability
hi = (1+alpha)*s_ref; lo = (1-alpha)*s_ref
keep = {e in order[:k] : p[e] >= hi}
avail = GPU_cache  union  replaceset(batch)    # experts resident or being computed anyway
cands = {e in order[k:] : lo < p[e] < s_ref}   # paper: <= s_ref
for e in order[:k] \ keep:
    if e in avail: use e
    else: pick c in cands with c in avail and c not used; if found use c (remove from cands) else use e (load)
weights = p[used]; if norm_topk_prob: weights /= weights.sum()
```

Models and headline numbers:
- Abstract: "a 48% reduction in decoding latency" and "an expert cache hit rate above 60%".
- Sec. VI-B: 48% TPOT reduction on S3 at batch 3 vs best baseline; average TPOT reduction 24% (batch 1) and 35% (batch 3) (verify exact sentence).
- Fig. 13: GPU cache ratio at least 65% improvement over best baseline (relative).
- Table V (Qwen2MoE, alpha = 0.25), substitution vs original: GaoKao 76.1 (73.5), MMLU 71.0 (71.0), HumanEval 53.3 (53.0), MT-Bench 8.37 (8.52).
- Table IV: accuracy roughly flat for alpha up to about 0.35 on most tasks.
- Table VI: lowering gate scores without substitution (ablation).
- Prefetch accuracy about 82% for top-score prediction.

Open questions for reimplementation: exact `norm_topk_prob` per model; window n; pooling semantics of `replaceset`; tie and ordering in substitute choice; whether shared experts are untouched (yes for DeepSeek/Qwen2 shared experts, presumably) (verify).

(a) No. (b) No. (c) No. Substitution is purely by gate-score proximity; "functionally similar" is asserted, not measured.

---

## 2. CCE: Mixture of Cache-Conditional Experts (arXiv 2412.00099)

- Exact title: "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference"
- Authors: Andrii Skliar, Ties van Rozendaal, Romain Lepert, Todor Boinovski, Mart van Baalen, Markus Nagel, Paul Whatmough, Babak Ehteshami Bejnordi (Qualcomm AI Research)
- Dates: v1 27 Nov 2024; v2 24 Jun 2025 (TMLR 06/2025). ID verified.

What it changes: the expert ranking per token, so that already-cached (DRAM-resident) experts are preferred; weights still come from the unmodified router.

Training-free: yes ("flexible, training-free" per abstract).

How it chooses experts (Sec. 3, as rendered in the HTML):
- Eq. 1: y = sum_{i in r[:K]} w_i E_i(x); Eq. 2: r = argsort w; Eq. 3: w = sigma(z) = sigma(G(x)) (softmax).
- Eq. 4: hit_rate = count(r[:K] cap C) / K.
- Eq. 5 (promotion): promote(r^subset; r^all) := r^subset (+) (r^all \ r^subset), order preserved within groups.
- Max Rank (Eq. 6, 7, Alg. 1): r' = promote(r[:M] cap C; r); r' = promote(r[:J]; r'); take r'[:K].
- Cumulative probability threshold (Eq. 8, Alg. 2): M = min i s.t. sum_{j=1}^{i} G(x)[r_j] >= p; then same promotions.
- Cache-Prior (main, Eq. 9, 10): z' = z + lambda * Delta_avg * m~_t, with Delta_avg = E_x E_t [max(z) - min(z)] (running per-layer average; Appendix D also tests calibration-set estimate), m_t in {0,1}^N the cache bitmask after token t-1, m~_t optionally with top-J added. Rank by z', take top K.
- Top-J guarantee: "we always select them, even if they are not in the cache".
- Hyperparameters: lambda in [0,1] (lambda = 0 is original routing; sweeps over 50 points; on device {0.1, 0.3, 0.5, 0.7, 0.9}); J = 1 for Mixtral and Phi-3.5-MoE, J = 2 for Qwen and DeepSeek; p swept over 50 points; cache size default half of experts (Appendix D: quarter).
- Eviction: "All routing strategies in this work use the LRU eviction policy for expert removal"; within one token's parallel top-K, higher-weight experts evicted first (Sec. 4.2) (verify meaning).

How the selected experts are weighted:
- "We still use the unmodified router logits to compute the expert weights"; w = softmax(z) (original), applied to the new set r'[:K].
- Eq. 1 text: "The weights assigned to each expert may be re-normalized after the top-K selection in Equation 1." Whether Cache-Prior renormalises is not stated (verify; probably follows each model's native convention).

Per-token reimplementation (Cache-Prior):
```
z = router_logits(x); w = softmax(z)
m = cache_bitmask (1 if expert in DRAM cache); optionally m[top-J of z] = 1
z' = z + lambda * Delta_avg[layer] * m          # Delta_avg = running mean of max(z)-min(z)
S = topK(z')
y = sum_{i in S} w_i E_i(x)                      # renormalise w over S iff model normally does (verify)
update LRU cache with S
```

Models and headline numbers:
- Models (Table 1): Mixtral-8x7B (8 experts, top-2), Phi-3.5-MoE (16, top-2), DeepSeek-V2-Lite (64+2 shared, top-6+2), Qwen1.5-MoE-A2.7B (60+4 shared, top-4+4).
- Abstract: "reduces cache miss rates by over 50%, with negligible impact on perplexity (0.1%-3%)" (verify exact wording; returned with this content) and ">2x" on-device speedup: "Our approach provides over a 2x speedup on-device" (Sec. 4.5, Fig. 1 right).
- Miss rates at lambda = 0.5 on WikiText (table cited as "Table 10", rendered near Fig. 9): Qwen1.5-MoE (30/60 cached) 35% to 16%; DeepSeek-V2-Lite (32/64) 28% to 7%; Phi-3.5-MoE (8/16) 22% to 9%; Mixtral (4/8) 40% to 21%.
- Sec. 4.7: Qwen halves miss rate at about +0.5% perplexity; Mixtral about +2.9%; DeepSeek about +0.1%.
- On-device: 4-bit Qwen with 30 of 60 experts cached on 12 GB; 8-bit with 45 of 60 on 16 GB (Sec. 4.5). Absolute tokens/s only in figures (Fig. 1 right, Fig. 14) (verify).

Open questions: renormalisation; whether Delta_avg is per-layer running or calibrated in the reported runs; exact M sweep range for Max Rank (reported as {0..K}, which looks odd) (verify).

(a) No. (b) No. (c) No. Uses only router logits, a cache bitmask and the scalar logit range.

---

## 3. MoE Lens (arXiv 2603.05806)

- Exact title: "MoE Lens -- An Expert Is All You Need"
- Authors: Marmik Chaudhari, Idhant Gulati, Nishkal Hundia, Pranav Karra (The Pennsylvania State University), Shivam Raval (Harvard University); a University of Maryland affiliation also appears, unattributed.
- Dates: v1 6 Mar 2026 (only version). Comments: "ICLR 2025 Workshop on Sparsity in LLMs (SLLM)". ID verified.

What it changes: analysis paper (routing statistics plus logit-lens style early decoding); proposes nothing deployable, suggests targeted expert pruning.

Training-free: yes (analysis only).

Exact claim on cosine 0.95:
- Abstract: "with some layers having cosine similarity as high as 0.95" and "perplexity increasing by only 5% when using a single expert across all three domains".
- Model: "DeepSeekMoE ... with 2 shared + 64 routed experts with top-k=6" (the paper does not say "16B" in the text I could read; 27 MoE layers is consistent with DeepSeekMoE 16B, whose first layer is dense) (verify).
- Quantity compared: hidden states, not bare expert outputs. Notation paragraph: "we represent the hidden state of the top-weighted expert (top-k=1) combined with the residual output as" H_t^{l1}, and the top-6 version as H_t^{l6}. Eq. 1: h_t^l = sum_{i=1}^{Top-k(r_t(x))} (r_{i,t} E_i(x)) + u_t^l, u_t^l the post-attention residual stream.
- Layers: Fig. 3 (right) caption: "across all 27 layers shows consistently high alignment". Domains: English, Code, French.
- Numbers: 0.95 and 5% appear only in the abstract; body gives no numeric values (values only in Fig. 3, an image I could not read) (verify from figure). No tables in the paper.
- So our belief "top-1 expert output at cosine about 0.95 with the full MoE output" is NOT what is claimed: the cosine is between residual-stream hidden states (top-1 term plus residual vs top-6 terms plus residual), "as high as" 0.95 in some layers. Because the residual u_t^l is included, this cosine is inflated relative to a comparison of MoE outputs alone. Shared experts: whether included in H is not stated (verify).
- Weighting of the single expert: Eq. 1 uses r_{i,t}; no renormalisation is described (verify; it matters for interpreting the 5% perplexity).

Open questions: exact per-layer values; whether shared experts are included; whether r_{1,t} is renormalised to 1 in the perplexity experiment; dataset sizes.

(a) No expert-expert inner products; only cosine(H^{l1}, H^{l6}). (b) No. (c) No.

---

## 4. Variational Inference, Entropy, and Orthogonality: A Unified Theory of Mixture-of-Experts (arXiv 2601.03577)

- Exact title: "Variational Inference, Entropy, and Orthogonality: A Unified Theory of Mixture-of-Experts"
- Authors: Ye Su, Yong Liu
- Dates: v1 7 Jan 2026 (only version). 27 pages, 3 figures, CC BY 4.0. ID verified.

What it proves / claims:
- Bayesian view: top-k routing and load balancing as sparse posterior approximation and prior regularisation; information-theoretic view: reduce routing ambiguity and maximise channel capacity. Theorem 3.3 holds under an assumption f_j approx p_j (stated as an equivalence) (verify).
- Sec. 5.1 routing as Sparse Subset Selection (SSS): "Our goal is to pick a sparse weighting vector alpha in R^E, which approximates the optimal target representation y." min_alpha ||y - E alpha||_2^2 s.t. ||alpha||_0 <= k, with E = [E_1(x), ..., E_E(x)] in R^{d x E}; "we treat this matrix as a fixed feature basis". Greedy router modelled as scoring s_i approx <y, E_i> and S = Top-k(s).
- Theorem 5.1: "The Sparse Subset Selection problem for MoE routing is NP-hard" (reduction from Exact Cover by 3-Sets, y = 1, E = incidence matrix). The converse direction of the reduction appears loosely argued for real-valued alpha (verify).
- Theorem 5.3 (Coherence Barrier for Greedy Routing): with mu(E) = max_{i != j} |<E_i, E_j>| (unit-norm columns): "The greedy strategy is guaranteed to recover the optimal subset" "if and only if the mutual coherence satisfies the condition" mu(E) < 1/(2k-1); otherwise "there exist target signals y for which the greedy router selects a strictly suboptimal subset". The "if and only if" is stronger than Tropp's result (which is sufficient, with a partial converse via ERC); the paper's own Fig. 2 says "The greedy router succeeds well beyond the conservative theoretical bound". Note 1/(2k-1) is Tropp's m < (1/2)(mu^{-1}+1) rearranged.
- Theorem 6.1: orthogonal columns imply "the greedy solution is identical to the global optimal solution".
- Theorem 6.2: if "the orthogonality loss drives the coherence to satisfy" mu < 1/(2k-1), "the greedy Top-k routing strategy is guaranteed to recover the unique global optimal subset". Assumes the loss reaches the threshold.
- Theorem 6.3 (DPP): F(S) = log det(L_S), L_ij = <E_i, E_j>, claimed monotone submodular with F(S_G) >= (1 - 1/e) F(S*), citing Nemhauser et al. 1978. log det is submodular but not monotone in general (only when eigenvalues of L_S exceed 1, roughly); claim needs checking (verify).
- Theorem 6.4: ambiguity decomposition E_ensemble = mean individual error - Ambiguity (NCL view).
- Orthogonality loss: L_o = sum_{i in S} sum_{j in S, j != i} ( E_i^T E_j / (||E_i|| ||E_j||) )^2.
- In proofs: "the inner minimization problem is a standard Ordinary Least Squares (OLS) regression", alpha_S* = (E_S^T E_S)^{-1} E_S^T y; under orthogonality "the Gram matrix of the subset ... becomes the identity matrix". OMP is not named as such; Tropp 2004 and Tropp and Gilbert 2007 are cited (Appendix F). Das and Kempe 2011 and the submodularity ratio are NOT cited.
- Inference: "zero additional cost during inference, as the router reverts to standard Top-k selection." No reduction of active experts.

Training-free: no; the practical proposal is a training-time orthogonality regulariser.

Experiments: only synthetic. Fig. 2: synthetic SSS with k = 6 (threshold 1/11). Fig. 3: synthetic classification (4,000 samples, 100 features of which 10 informative, 10 classes, class_sep 0.6, seed 4242, stratified 10-fold CV), 16 experts, k = 2, expert hidden dim 32, GELU, batch 128, AdamW lr 1e-3, 30 epochs, aux weight 0.01, regulariser weight 0.1; L_o reaches "the highest final accuracy of 73.6%". "Effective Rank of the expert feature space" is plotted in Fig. 3, definition not found (verify). No LLM experiments.

(a) Yes, in theory: mutual coherence, Gram matrix E_S^T E_S, DPP kernel L_ij = <E_i, E_j>, orthogonality loss on output cosines. (b) Only inside proofs (OLS on a fixed support against an abstract target y); not used to reweight a reduced set at inference, no calibration. (c) No; "Effective Rank" in Fig. 3 is undefined in the text I read and is not tied to a per-token effective number of experts (verify).

Positioning: our work can cite this as the theory side (routing as SSS over expert outputs, coherence) and differentiate by (i) target = dense top-k output sum_{i in T} g_i E_i, so G_ST g replaces E^T y; (ii) refit weights rather than fixed gates; (iii) calibrated second moments rather than per-token outputs; (iv) training-free inference reduction; (v) correct use of ERC / submodularity ratio rather than the "iff" coherence claim.

---

## 5. LightMoE (arXiv 2603.12645)

- Exact title: "LightMoE: Reducing Mixture-of-Experts Redundancy through Expert Replacing"
- Authors: Jiawei Hao, Zhiwei Hao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu, Dan Zeng
- Dates: v1 13 Mar 2026 (only version). ID verified.

What it changes: static compression. Low-importance experts are grouped and replaced by a shared base plus per-expert low-rank adapters, then fine-tuned.

Training-free: no (fine-tuning: AdamW, batch 32, lr 1e-4, 2,000 steps for preservation tasks on MetaMathQA / CodeFeedback / Cleaned Alpaca; 500 steps for adaptation tasks).

How it chooses experts:
- Importance (Eq. 2): G_i = sum_x G(x)_i / sum_i sum_x G(x)_i over a sampled set (about 2^17 tokens).
- Layer threshold (Eq. 3): p_j = clip(p * exp(-alpha (norm_j - 1)), p_min, p_max), norm_j = layer j router output norm relative to mean; alpha = 0.3, p_min = 0.8 p, p_max = 1.2 p; base p per task by binary search (Table 9).
- Sort ascending by G_i and take experts until cumulative score just exceeds the layer threshold.
- Grouping: top-M experts dominant; others join the dominant expert with highest "average sample-wise cosine similarity between the routing logits"; group size 3.

How replaced experts are weighted:
- Shared base (Eq. 4): W_share = sum G_i W_i / sum G_i; each expert becomes W_share + B_n A_n, rank r = 16 (LoRA init).
- Annealed recovery (Eq. 6, 7): W* = beta W_n + (1 - beta) W_share + B A, beta = max(1 - t/(eps T), 0), eps in {0.1..0.5} (0.1 or 0.2 often best).
- Routing weights: router is kept; whether it is frozen during fine-tuning is not stated (verify).
- Compression ratio (Eq. 5): rho = 1 - [(N - N' + M) n m + N' r (n + m)] / (N n m).

Headline numbers: Table 1, 30% compression: LightMoE average 55.3 vs LoRA 55.5 (Math 59.7 vs 59.8). 50%: average 48.1; abstract claims "average performance improvements of 5.6% across five diverse tasks" over existing methods (and 3.8% over direct replacement); I could not see the 50% baseline rows to verify (verify). Models evaluated: OLMoE-1B-7B-SFT (main backbone; 6.9B total, 1.3B active) and DeepSeek-V2-Lite (Appendix A.1, Table 4). Baselines (Table 1, caption "Main performance comparison across methods and tasks at different compression ratios."): Full FT, LoRA, MoBE, HC-SMoE, MC-SMoE, Replace (w/o shared), Replace (w. shared); MC-SMoE* only at 30% and 40%.

Open questions: 50% baseline rows (not visible in HTML extraction); router handling during fine-tuning.

(a) No (cosine of routing logits only). (b) No (gate-weighted average of weights, then gradient descent). (c) No.

---

## 6. BuddyMoE (arXiv 2511.10054)

- Exact title: "BuddyMoE: Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference"
- Authors: Yun Wang, Lingyun Yang, Senhao Yu, Yixiao Wang, Ruixing Li, Zhixiang Wei, James Yen, Zhengwei Qi (Shanghai Jiao Tong University)
- Date: v1 13 Nov 2025. Listed for PPoPP 2026 (verify venue). ID found via search and confirmed on arxiv.org/html/2511.10054.

What it changes: on a prefetch miss, substitutes the missing expert with a GPU-resident "buddy" from an offline lookup table instead of stalling on PCIe or dropping.

Training-free: yes (offline profiling only).

How it chooses experts:
- Offline co-activation counts M(i,j) per layer; q(j|i) = M(i,j) / sum_j' M(i,j'), q(i|i) = 0 (Eq. 4); optional probability-weighted counts (min of the two routing probabilities), Laplace smoothing.
- Buddy list: shortest prefix of experts sorted by q(j|i) with cumulative mass >= alpha (Eq. 5, 6), capped at K_max. Sizes "typically" 2 to 16 (evaluation) vs K_max 5 to 20 (overhead section), not reconciled.
- Intro claims buddies are "identified via an offline analysis of co-activation patterns and output similarity on a calibration dataset", but Sec. 3 defines no output-similarity formula (verify).
- Runtime gates: Token Activating Entropy (TAE): no replacement for a token when TAE <= tau (optional probability-margin rule); distribution gate bypasses replacement when CPU-resident fraction delta >= beta.
- Buddy score: Psi = q(j|i) (1 + eta z^) (1 - kappa hop), defaults eta = kappa = 0 (so ranking by co-activation frequency); avoid reusing a buddy within the same token (atomic compare-and-swap in a CUDA kernel). Algorithm 1 (Sec. 4) is simpler: first GPU-resident buddy within rank H not already in the token's set.
- Fallback: prefetch original or drop.

How the buddy is weighted: not stated (inherit original gate or renormalise unknown) (verify in code if released).

Models and headline numbers (DeepSeek-V2-Lite only, 64 experts top-6, single A100, llama.cpp):
- Abstract: "improves up to 10% throughput" with negligible accuracy loss.
- Table 2 (cache rate 0.75): original 0.735 at 34.23 t/s; BuddyMoE (tau = 0.95, |B| = 16, rho = 3) 0.695 at 36.75 t/s.
- Table 3 (cache rate 0.5): original 28.56 t/s; BuddyMoE rho = 3: 0.635 at 30.21 t/s.
- Table 4 (cache rate 0.375): original 24.78 t/s; BuddyMoE rho = 3: 0.645 at 27.33 t/s.
- Random replacement accuracy 0.55, 0.23, 0.16 at cache rates 0.75, 0.5, 0.375. Fig. 8: about 20% less PCIe read bandwidth.
- Note: 0.735 to 0.695 is a 4-point drop, hard to square with "negligible". Benchmark name not captured (verify).

(a) Only claimed "output similarity" without a definition found; operationally co-activation statistics. (b) No. (c) No.

---

## 7. MoE-Spec (arXiv 2602.16052)

- Exact title: "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding"
- Authors: Bradley McDanel, Steven Li, Sruthikesh Surineni, Harshit Khaitan
- Date: v1 17 Feb 2026 (only version). ID verified.

What it changes: at speculative verification, caps the number of distinct experts per layer across all draft-tree tokens to a budget B.

Training-free: yes.

How it chooses experts:
- Router-based (default), Eq. 5: s_i = sum_{t=1}^{M} g_i(h_t) over the M draft-tree tokens; keep the B experts with highest s_i. Static variant: counts c_i from calibration (Eq. 4).
- Oracle (Eq. 6): i* = argmin_{i not in S} sum_{t=1}^{M} || O*_t - sum_{j in S cup {i}} g_j(h_t) E_j(h_t) ||_2^2, built greedily; "The oracle greedily selects experts to minimize reconstruction error against the unpruned output." "This requires running all experts and is impractical for speedup, but provides a ceiling on achievable quality." Reported: oracle matches router error with about 25% fewer experts (B = 24 oracle vs B = 32 router, both 3.2% error; B = 16 oracle 8.0% vs router B = 24 7.7%).
- Budgets: OLMoE-1B-7B (64 experts, k = 8) B in {8, 16, ..., 56}; Qwen3-30B-A3B (128, k = 8) B in {16, 24, 32, 48, 64}; Mixtral-8x7B (8, k = 2) B in {2..7}; draft tree size 63 (swept 3 to 511).

How remaining experts are weighted:
- Truncation (Eq. 7): MoE_S^trunc(h_t) = sum_{i in T_k(h_t) cap S} g_bar_i(h_t) E_i(h_t), gates as in the original top-k (no renormalisation); a token with no overlap gets zero MoE output.
- Substitution (Eq. 8, default): top-k within S, gates g_bar_i^S "routing probabilities over the constrained selection" (renormalisation implied, formula not given) (verify). The two perform comparably.

Headline numbers: abstract: "yields 10-30% higher throughput than state-of-the-art speculative decoding baselines (EAGLE-3) at comparable quality". Table 1 reports speedup over autoregressive decoding (e.g. Mixtral MBPP T=0: EAGLE 1.7x vs MoE-Spec 2.3x); improves speedup in 27 of 30 configurations. Datasets: GSM8K, MATH500, HumanEval, MBPP, CNN/DailyMail.

(a) Only in the oracle, implicitly (reconstruction error of output sums; no Gram matrix estimated). (b) No: gates stay at router values. (c) No. Appendix D co-activation is a count statistic.

---

## 8. Not All Experts are Equal (arXiv 2402.14800)

- Exact title: "Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models"
- Authors: Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li
- Dates: v1 22 Feb 2024; v2 30 May 2024 (ACL 2024, verify venue). ID verified; our belief about title and authors is correct.
- Code: https://github.com/Lucky-Lance/Expert_Sparsity (`model/modeling_mixtral.py`, class `DynamicSkippingMixtralSparseMoeBlock`).

What it changes: (i) static expert pruning (keep r of 8 experts per layer); (ii) dynamic per-token skipping of the second expert.

Training-free: yes ("It is a post-training pruning method and does not require any parameter update").

Dynamic skipping rule (Sec. 3.3): top-2 experts e0, e1 with w_e1 < w_e0; "if w_e1 < beta w_e0, we do not assign x to expert e1"; "beta is a hyper-parameter separately set for each MoE layer"; "we forward the model over the sampled calibration data and set beta as the median value of" w_e1 / w_e0 per layer. Appendix A.7: median chosen so "the skipping will happen with around 50% possibility". (The median, not mean.)

Per-layer beta values (Appendix A.7, Mixtral 8x7B, 32 layers):
- C4: 0.402, 0.494, 0.463, 0.484, 0.478, 0.491, 0.523, 0.521, 0.544, 0.570, 0.574, 0.489, 0.503, 0.618, 0.568, 0.535, 0.559, 0.519, 0.537, 0.487, 0.469, 0.461, 0.461, 0.469, 0.458, 0.418, 0.433, 0.418, 0.406, 0.433, 0.447, 0.535
- MATH: 0.503, 0.586, 0.505, 0.531, 0.509, 0.422, 0.511, 0.461, 0.447, 0.478, 0.529, 0.454, 0.472, 0.531, 0.499, 0.486, 0.503, 0.491, 0.430, 0.440, 0.402, 0.423, 0.386, 0.407, 0.395, 0.354, 0.340, 0.351, 0.334, 0.368, 0.365, 0.346

General top-k version (Appendix A.2, theory only, no experiments): keep top-i* experts with i* = min i s.t. sum_{m=i+1}^{k} w_m <= beta * sum_{m=1}^{k} w_m; for top-2 they map this to w_2 <= beta w_1 via a reparameterisation beta = H/(D-H) (verify notation).

Weighting (code, verbatim lines):
- `routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)`
- `routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)`
- `onlytop1_mask = routing_weights[:, 1] < self.beta * routing_weights[:, 0]`
- `routing_weights[onlytop1_mask, 1] = 0`
- `routing_weights /= routing_weights.sum(dim=-1, keepdim=True)`
- `expert_mask[onlytop1_mask, 1, :] = 0`
So a skipped token's output is E_e0(x) with weight exactly 1 (renormalised). The ratio test is on softmax probabilities, which is the same ratio as on renormalised top-2 weights.

Per-token reimplementation:
```
p = softmax(router_logits); (w0, w1), (e0, e1) = top2(p)
if w1 < beta[layer] * w0:  y = E_e0(x)
else: y = (w0 E_e0(x) + w1 E_e1(x)) / (w0 + w1)
beta[layer] = median over calibration tokens of w1/w0 (128 x 2048-token C4 sequences in the paper's default calibration)
```
Prefill vs decode: not specified; speed measured on token generation (verify).

Expert pruning (Sec. 3.2, Eq. 3): min_C || F'(x, C) - F(x) ||_F s.t. C subset of experts, |C| = r, enumerating all subsets per layer on cached layer inputs/outputs; calibration 128 sequences x 2048 tokens from C4 (MATH train for maths); App. A.3: 64 to 128 sequences give about 64.2 to 64.3, one sequence 62.6.

Headline numbers (Table 5, LM-eval average of 8 tasks, speedup vs 8 experts):
- Mixtral 8x7B: baseline 67.58 (1.00x); skipping only 66.37 (1.08x); prune r=6 64.22 (1.19x); r=6 + skip 62.91 (1.23x); r=4 + skip 57.91 (1.31x).
- Mixtral 8x7B Instruct: baseline 69.98; skipping only 69.03 (1.08x); r=6 + skip 66.04 (1.27x); r=4 + skip 62.33 (1.33x).
- Table 8 (GSM8K): skipping only 54.28 at 1.08x (Mixtral), 61.94 at 1.05x (Instruct).
- Text: "we observe a 1.2x to 1.3x inference speedup with r=6".
- Average activated experts per token is not reported (the ~1.7 figure is not in the paper as far as I could see; with median beta about half of tokens skip, implying roughly 1.5 per token, which is our inference, not theirs) (verify).
- Column alignment of the skipping tick in Table 5 was ambiguous in HTML (verify).

(a) No. (b) No (no refit; pruning enumerates subsets). (c) No.

---

## 9. Harder Tasks Need More Experts (arXiv 2403.07652)

- Exact title: "Harder Tasks Need More Experts: Dynamic Routing in MoE Models"
- Authors: Quzhe Huang, Zhenwei An, Nan Zhuang, Mingxu Tao, Chen Zhang, Yang Jin, Kun Xu, Kun Xu (listed twice on arXiv), Liwei Chen, Songfang Huang, Yansong Feng
- Date: v1 12 Mar 2024 (only version). ID verified; our belief is correct. Code: github.com/ZhenweiAn/Dynamic_MoE.

What it changes: top-p routing; number of experts varies per token.

Training-free: no. "We train several variants of our architecture from scratch" (RedPajama, 100B tokens; MoE 3.5B total, 16 experts, 24 layers, hidden 1024; dense baselines 374M and 570M).

Rule (Sec. 2.2, Eq. 4): "find the smallest set of experts whose cumulative probability exceeds the threshold p"; "We add new experts in descending order of their probabilities." p = 0.4 ("The threshold p in our routing mechanism is 0.4"); Sec. 3.3 sweeps p from 0.1 to 0.7 at inference.

Weighting (Eq. 5): g_i(x) = P_i for selected experts, 0 otherwise; no renormalisation described for the dynamic model (the Top-K baseline does renormalise).

Losses: dynamic loss Loss_d = -sum P_i log P_i (Eq. 7); total Loss = Loss_lm + alpha Loss_b + beta Loss_d (Eq. 11), alpha = 1e-2, beta = 1e-4.

Headline numbers: abstract: "achieving an average improvement of 0.7% with less than 90% activated parameters"; Table 1: MoE-Dynamic 42.3 vs MoE-Top2 41.6; BBH 25.6 vs 23.3; Table 3: 1.76 activated experts per token on average, 1.87 on BBH; Table 2: about 1.82 across training data sources. Sec. 6, Fig. 4: "The lowest layer activates the most experts, up to 4 experts per token"; "At the topmost layer, the number of activated experts per token is reduced to even one."

(a) No. (b) No. (c) No (effective number of experts is a count from routing probabilities).

---

## 10a. Tropp 2004, "Greed is good"

- Exact title: "Greed is Good: Algorithmic Results for Sparse Approximation"; Joel A. Tropp; IEEE Transactions on Information Theory 50(10), October 2004, DOI 10.1109/TIT.2004.834793. Read from the author's preprint PDF (tropp.caltech.edu/papers/Tro04-Greed-Good-preprint.pdf); theorem numbers below are from the preprint and should be checked against the published version (verify numbering).

Definitions (Sec. II): coherence mu = max_{j != k} |<phi_j, phi_k>| (unit-norm atoms); cumulative coherence mu_1(m) = max_{|Lambda| = m} max_{psi} sum_{lambda in Lambda} |<psi, phi_lambda>|, psi ranging over atoms not in Lambda, mu_1(0) = 0.

Exact recovery condition (Theorem 3.1, Sec. III-A): s = sum_{lambda in Lambda_opt} b_lambda phi_lambda with |Lambda_opt| = m, atoms linearly independent, coefficients nonzero; Phi_opt the d x m matrix of these atoms, Psi_opt the remaining atoms; Phi_opt^+ = (Phi_opt^* Phi_opt)^{-1} Phi_opt^*. "A sufficient condition for Orthogonal Matching Pursuit to recover the sparsest representation of the input signal is that max_psi || Phi_opt^+ psi ||_1 < 1, (ERC) where psi ranges over the columns of Psi_opt." "A fortiori, Orthogonal Matching Pursuit is a correct algorithm for (D, m)-EXACT-SPARSE so long as the condition (ERC) holds for every signal with an m-term representation." Theorem A (intro) adds that the sparsest representation is then unique and BP also recovers it.

Partial converse (Theorem 3.10): "Assume that all m-term representations are unique but that the Exact Recovery Condition fails for a signal with optimal synthesis matrix Phi_opt. Then there are signals in the column span of Phi_opt that Orthogonal Matching Pursuit cannot recover."

Coherence-based sufficient conditions: Theorem 3.5: ERC holds whenever mu_1(m-1) + mu_1(m) < 1 (via max_psi ||Phi_opt^+ psi||_1 <= mu_1(m) / (1 - mu_1(m-1))). Corollary 3.6: OMP and BP recover every m-atom superposition if m < (1/2)(mu^{-1} + 1), or mu_1(m) < 1/2. ("The incoherence condition is the best possible.")

Approximation (Sec. IV): Theorem 4.2 (General Recovery), Corollary 4.3: if mu_1(m) < 1/2, OMP gives ||s - a_m||_2 <= sqrt(1 + C(D, m)) ||s - a_opt||_2 with C(D, m) <= m (1 - mu_1(m)) / (1 - 2 mu_1(m))^2. Corollary 4.4 (= Theorem C): if m <= (1/3) mu^{-1} or mu_1(m) <= 1/3, ||s - a_m||_2 <= sqrt(1 + 6m) ||s - a_opt||_2.

Relevance: ERC is the right per-token statement for our OMP on expert outputs, with Phi_opt^+ computable from G (Phi_opt^+ psi = G_SS^{-1} G_{S,psi}), so ERC = max_{j not in S} || G_SS^{-1} G_{S j} ||_1 < 1 is checkable from the calibrated Gram (requires unit-normalised columns for the coherence corollaries). Note ERC concerns exact recovery of a target that lies in the span of m atoms; our target sum_{i in T} g_i E_i lies in span(T), |T| = k, so ERC answers "does OMP pick T", while reducing to |S| < k is the SPARSE (approximation) regime of Sec. IV.

(a) Yes (theory over dictionary inner products). (b) Yes (OMP coefficients by least squares). (c) No.

## 10b. Das and Kempe 2011, "Submodular meets Spectral"

- Exact title: "Submodular meets Spectral: Greedy Algorithms for Subset Selection, Sparse Approximation and Dictionary Selection"; Abhimanyu Das, David Kempe (USC); ICML 2011; arXiv 1102.3975 (v1 19 Feb 2011, v2 25 Feb 2011). Read from arXiv PDF; numbering may differ in the ICML proceedings version (verify).

Setting (Sec. 2): variables normalised to mean 0, variance 1; C with c_ij = Cov(X_i, X_j); b_i = Cov(Z, X_i). Definition 2.1: R^2_{Z,S} = (Var(Z) - E[(Z - Z')^2]) / Var(Z), optimal coefficients a_S = C_S^{-1} b_S, so R^2_{Z,S} = b_S^T C_S^{-1} b_S. lambda_min(C, k) = min_{|S| = k} lambda_min(C_S) ("the smallest k-sparse eigenvalue"; the paper relates it to RIP; it does not use the term "restricted eigenvalue").

Definition 2.3 (Submodularity Ratio): gamma_{U,k}(f) = min_{L subset U, S: |S| <= k, S cap L = empty} [ sum_{x in S} (f(L cup {x}) - f(L)) ] / [ f(L cup S) - f(L) ]; for R^2, gamma_{U,k} = min (b_S^L)^T b_S^L / ((b_S^L)^T (C_S^L)^{-1} b_S^L), with C^L, b^L the normalised covariances of the residuals Res(X_i, L). "f is submodular if and only if gamma_{U,k} >= 1, for all U and k."

Lemma 2.4: gamma_{U,k} >= lambda_min(C, k + |U|) >= lambda_min(C). "For all our analysis in this paper, we will use |U| = k, and hence gamma_{U,k} >= lambda_min(C, 2k)."

Theorem 3.2 (Forward Regression, Definition 3.1: add X_m maximising R^2_{Z, S_i cup {X_m}}, i.e. OLS / orthogonal least squares):
R^2_{Z,S^FR} >= (1 - e^{-gamma_{S^FR,k}}) OPT >= (1 - e^{-lambda_min(C,2k)}) OPT >= (1 - e^{-lambda_min(C,k)}) Theta((1/2)^{1/lambda_min(C,k)}) OPT, with OPT = max_{|S| = k} R^2_{Z,S}.

Theorem 3.7 (OMP, Definition 3.5: add X_m maximising |Cov(Res(Z, S_i), X_m)|):
R^2_{Z,S^OMP} >= (1 - e^{-(gamma_{S^OMP,k} lambda_min(C,2k))}) OPT >= (1 - e^{-lambda_min(C,2k)^2}) OPT >= (1 - e^{-lambda_min(C,k)^2}) Theta((1/2)^{1/lambda_min(C,k)}) OPT.

Theorem 3.9 (Oblivious: take the k variables with largest b_i): R^2 >= gamma_{empty,k} / lambda_max(C,k) OPT >= lambda_min(C,k) / lambda_max(C,k) OPT. Footnote: "Forward Regression is sometimes also referred to as Orthogonal Matching Pursuit (OMP)"; they follow the naming where FR = our OLS and OMP = residual-correlation selection.

Relevance: gives the multiplicative guarantee for our OLS (= their Forward Regression) and OMP variants with gamma lower-bounded by the smallest 2k-sparse eigenvalue of the normalised Gram; the Oblivious bound is the analogue of "keep the largest individual contributions", useful as a baseline argument. Needs column normalisation and R^2 defined relative to ||target||^2.

(a) Yes (covariance / Gram theory). (b) Yes (C_S^{-1} b_S). (c) No.

---

## Not accessed or not verified

- MoE Lens Fig. 3 per-layer cosine values (image only).
- SMoE: exact `norm_topk_prob` flags in subclasses; pooling semantics of `replaceset`; strict vs non-strict bound for substitutes.
- CCE: renormalisation under Cache-Prior; absolute on-device tokens/s.
- LightMoE: 50% baseline rows (so the "+5.6%" claim is unverified).
- BuddyMoE: output-similarity definition; buddy gate weighting; benchmark name; code availability.
- 2601.03577: Appendix K and parts of Appendix F to J beyond the fetched window; definition of "Effective Rank".
- Tropp: published-version theorem numbering (preprint read). Das and Kempe: ICML-version numbering (arXiv read).


---

# Part 3 (full reading notes)

# Literature check, part 3: PRIME, ConMoE, COMPEL

Our method, for reference: per token, router top-k set T with gates g; keep a subset S; kept weights c = G_SS^{-1} G_ST g, where G is the Gram matrix of expert OUTPUTS from calibration data (projection of the full top-k output onto span of kept experts); S chosen by greedy OLS/OMP on G.

Sources read on 10 Nov 2026: arXiv abstract and HTML pages for PRIME and ConMoE (read through a summarising fetch tool, so quotes are as returned by that tool; the last ~5k characters of PRIME and ~11k of ConMoE, i.e. late appendices, were not read). COMPEL full text read from the ACL Anthology PDF (text extraction).

---

## 1. ConMoE (closest overlap of the three, but still not our method)

- Title (abs page, v2): "ConMoE: MoE Compression as Expert Selection and Functional Assignment". A third-party aggregator lists the title "ConMoE: Expert-Pool Consolidation via Prototype Reassignment for MoE Compression", possibly the v1 title (verify).
- Authors: Yilun Yao, Jiaming Pan, Elsie Dai, Peizhuang Cong, Chao-Hsuan Liu, Yaoming Li, Tong Yang.
- arXiv: 2605.29350 (cs.AI). v1 28 May 2026; v2 12 Oct 2026.
- What it changes: static expert-pool reduction (memory compression). Removed experts are hard-mapped to one retained "prototype" expert; router and expert weights are frozen.
- Training-free: yes. "ConMoE uses unlabeled calibration inputs and keeps the router and expert weights frozen." Also "The procedure uses only forward computation on unlabeled inputs ..." and "with no gradients or fine-tuning" (Sec. 5.1).
- Expert choice (Sec. 4.1, "Anchor Coverage"): demand score w_{l,e} = sum over tokens t where e is in top-k T_l(h_t) of g_{l,e}(h_t)^2 ||E_{l,e}(h_t)||^2 (Eq. 6). Coverage distance d_l(e,p) = normalised Frobenius distance of WEIGHTS averaged over gate/up/down projections (Eq. 7). Pool objective: min sum_e w_{l,e} min_{p in C_l} d_l(e,p), |C_l| = K_l (Eq. 8). Anchors = shortest demand-sorted prefix reaching rho = 0.8, then greedy fill of the budget by Eq. 8.
- Assignment (Sec. 3.2 / 4.2): functional cost F_l(e,p) = E_{h ~ D_{l,e}}[ g_{l,e}(h)^2 ||E_{l,e}(h) - E_{l,p}(h)||^2 ] (Eq. 4); pi_l(e) in argmin_{p in C_l} F_l(e,p) (Eq. 5). Hard one-to-one map, no soft weights.
- Weighting of remaining experts (Sec. 4.3): gates are simply summed. If pi(e1) = pi(e2) = p, "their contributions sum to (g_{l,e1}(h) + g_{l,e2}(h)) E_{l,p}(h)". No refit of coefficients; normalisation not described (verify).
- (a) Partly. It uses squared distances between expert OUTPUTS on calibration tokens (Eq. 4, which expands into inner products), but no Gram matrix, correlation matrix or cosine between outputs was reported. Selection distance is weight space (Eq. 7).
- (b) No. No least squares, projection or closed-form coefficients; the coefficient on a prototype is the sum of the original gates (coefficient 1 per redirected expert), static map.
- (c) No. Only "k is the maximum number of experts selected per input" (Eq. 3); nothing on rank or span of expert outputs per token.
- Relation to us: ConMoE's F(e,p) is the 1-term, coefficient-fixed-at-1 special case of our residual; we instead solve for optimal coefficients on a multi-expert span per token. Worth citing as closest on "replace removed expert's output by a kept expert's output".

## 2. PRIME

- Title: "PRIME: Frequency-Aware Unified Pruning and Merging for MoE Compression".
- Authors: Shanfeng Huang, Rongqian Zhang, Zhou Fang, Song Xiao, Hai Du.
- arXiv: 2610.13982 (cs.LG, stat.ML). v1 23 Sep 2026 (date as shown on abs page; the ID prefix 2610 suggests October, verify).
- What it changes: static compression; joint prune and merge of experts plus a refit of the router weight matrix.
- Training-free: yes, one-shot post-training. Sec. 3.1: compression "without retraining"; Limitations: "a one-shot post-training method without fine-tuning recovery". Calibration: "we sample 32 sequences of 2,048 tokens each from a mixed corpus" (C4, Evol-CodeAlpaca, NuminaMath), Sec. 4.1.
- Expert choice (Sec. 3.2, Eqs. 1 to 6): global ILP, min sum_{ij} f_i D_{ij} x_{ij} + lambda sum_i f_i y_i with assignment, centroid budget K and self-assignment constraints. D_{ij} is "the normalized Euclidean distance between the mean outputs of E_i and E_j on the calibration set", "an empirical output-space proxy", following HC-SMoE.
- Weighting of remaining experts: merge W'_j = sum_{i in C_j} f_i W_i / sum_{i in C_j} f_i (Eq. 7, "We adopt frequency-weighted averaging"), i.e. parameter averaging. Router: pruned experts removed from the softmax, group probabilities summed (Eqs. 8 to 9), "We first remove pruned experts from the normalization and then aggregate probability mass within each group"; grouped logits are log-sum-exp of member logits.
- The "closed-form router refit" (Sec. 3.4, "Offline linear reconstruction"): "The compressed router is obtained by ridge regression:" (Eq. 10), closed form W'_r = Z~^T H (H^T H + lambda_r I_d)^{-1} (Eq. 11), H = calibration HIDDEN STATES (router inputs), Z~ = grouped logits. "This calibration-time step uses no backpropagation or iterative optimizer and leaves the aggregated expert weights unchanged." The refit "does not guarantee identical sparse routing decisions." Top-k unchanged at serving.
- (a) Only weakly: distances between MEAN expert outputs (one vector per expert), not inner products / Gram of per-token outputs. Its Gram matrix H^T H is of hidden states, not expert outputs.
- (b) No. It does have a closed-form least squares, but it regresses router LOGITS on hidden states to imitate log-sum-exp grouping; it does not solve for combination weights of kept experts to reconstruct the MoE output.
- (c) No. Nothing on effective number of experts, rank or span (late appendices from Appendix I onward not read, verify).
- Relation to us: the phrase "closed-form router refit" is a surface match only; the object regressed (router logits from inputs) and the Gram (inputs) differ from ours (output-space projection, Gram of expert outputs).

## 3. COMPEL

- Title: "COMPEL: Compensated Mixture-of-Experts Pruning with Expert-Layer distribution".
- Authors: Seohee Yoon, Yong Suk Choi (Hanyang University).
- Venue: Findings of ACL 2026, pp. 30408 to 30423, July 2026, DOI 10.18653/v1/2026.findings-acl.1521. arXiv ID: none found (searched arXiv and web; verify). Code: github.com/seohee0925/COMPEL.
- What it changes: static, layer-adaptive expert pruning plus additive WEIGHT compensation of surviving experts.
- Training-free: no in the strict sense. Fisher scores need gradients: F_{l,i} = E_{(x,y)~D_cal}[ ||grad_{theta_{l,i}} log p(y|x; theta)||_2^2 ] (Eq. 5); learnable importance parameters alpha, beta fitted by KL with "the Adam optimizer ... over 3 epochs with a batch size of 4 and a learning rate of 5e-3" (Sec. 4.2). No fine-tuning of model weights. Calibration: C4, 128 sequences of 1024 tokens.
- Expert choice: score = alpha_i^(l) times beta^(l); alpha fitted to normalised Fisher pi_F = F_{l,i} / sum_j F_{l,j} (Eqs. 6, 7); beta fitted to layer outlier ratio D_l = (1/E_l) sum_i 1(A_{l,i} > M * mean A_l), M = 5.0 (Eqs. 8 to 10).
- Weighting of remaining experts (Eq. 11): theta_{l,j} <- theta_{l,j} + gamma sum_{i in R(l)} ( F_{l,j} / (sum_{k in S(l)} F_{l,k} + eps) ) theta_{l,i}, for all j in S(l), gamma = 0.1. A parameter-space heuristic; router gates unchanged. The paper explicitly sets aside OBS closed form: "Under a local second-order approximation of the loss, OBS-style compensation admits a closed-form update in terms of the inverse Hessian H^{-1}. However, such a derivation is difficult to obtain for MoE".
- (a) No for the method. Figure 1 shows "Linear CKA Similarity" between experts (motivation only, following Lo et al. 2025; whether computed on outputs, verify), and Appendix C computes "cosine similarity between the weight matrices of experts". Neither enters the algorithm.
- (b) No. Compensation is a fixed Fisher-proportional parameter addition, not least squares or projection; static, not per token.
- (c) No. Nothing on effective number of experts, rank or span of outputs.

## Verdict (closest first)

1. ConMoE: (a) partial, output-difference distances E[g^2 ||E_e(h) - E_p(h)||^2] but no output Gram; (b) no, hard map with summed gates; (c) no.
2. PRIME: (a) weak, distance between mean expert outputs only; (b) no, its closed-form ridge refit is of router logits on hidden states (Gram of inputs), not of kept-expert combination weights; (c) no.
3. COMPEL: (a) no (CKA/cosine only in motivation figures); (b) no, Fisher-proportional weight-space addition; (c) no. Also not gradient-free.

None of the three estimates a Gram of expert outputs and solves c = G_SS^{-1} G_ST g, and none measures per-token effective experts via output span.


---
