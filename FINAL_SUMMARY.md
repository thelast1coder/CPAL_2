# MoE_CPAL: final summary (15 Nov 2026)

The kill test is NO-GO: in both models tested, a token's output is spread across most of its 8 experts, so
dropping experts and reweighting the rest cannot recover the loss. Even a method with perfect per-token
information gains only 1.1x to 1.2x over the best simple rule, against the 2x the project needed.
This direction is closed; the plan's next phases (P2 to P7) will not run.

## 1. What was tested
Projection routing: at inference, keep only m of the k = 8 experts a mixture-of-experts layer selects for
each token, choose them by pursuit, and set their weights by least squares from expert-output Gram matrices
estimated once on calibration text. Training-free. The paper needed three things to be true, each checked
cheaply in a pre-registered kill test (thresholds fixed at commit 1d7c06c before any real-model run):

| Check | Question | GO bar |
|---|---|---|
| K0 | Does our code reproduce the original models exactly? | Exactness checks |
| K1 | Do a few experts carry almost all of each token's output? | 95% of the output from 3 or fewer of 8 experts, in at least 60% of layers |
| K2 | Does the calibrated method clearly beat the best simple rule? | At least 2x lower layer error at 1 and 2 experts |
| K3 | Does it hold up on a real task (GSM8K)? | Recover at least half the accuracy plain top-m loses |

## 2. Setup
- Models: OLMoE-1B-7B-0924 (base and Instruct) and Qwen3-30B-A3B (bf16, decision D7), revisions pinned.
- Data: 1M calibration tokens, 20k validation, 50k (OLMoE) and 30k (Qwen3) analysis tokens, disjoint and
  checked against GSM8K and MATH-500 test questions. Seed 0 throughout.
- Hardware: Jarvis Labs, one RTX PRO 6000 Blackwell (96 GB), 14 Nov 2026.

## 3. Results

**K0 (correctness): PASS on both models** (after three fixes; see section 5).

| Check | OLMoE | Qwen3 |
|---|---|---|
| Wrapper vs original, fp32 logit difference | 0.0 | (OLMoE only, by design) |
| Identical greedy answers, 20 GSM8K prompts (bf16) | 20/20 | 20/20 |
| Perplexity, original vs wrapped | identical | identical |
| Native GSM8K accuracy, fixed 250 questions | 48.8% (card: 45.5) | 93.2% |

**K1 and K2: NO-GO on both models.** K3 was therefore not needed and not run.

| | OLMoE (16 layers) | Qwen3 (48 layers) | GO bar |
|---|---|---|---|
| K1: experts needed for 95% of output (median over layers) | 5 of 8 | 6 of 8 | 3 or fewer |
| K1: share of layers at 3 or fewer | 0% | 2% | 60% or more |
| K2: our method vs best simple rule, m = 1 / 2 | 1.03x / 1.05x | 1.03x / 1.04x | 2x |
| K2: ceiling with perfect information (U1) | 1.17x / 1.23x | 1.11x / 1.14x | |
| K2: pre-registered rescue (sketch Gram) | 0.99x / 0.99x | 0.96x / 0.95x | 2x |

Overall verdict by the pre-registered rule: **NO-GO** (K1 NO-GO on both models; K2 NO-GO on both after the
rescue).

## 4. What it means
- **Experts do not overlap.** The outputs of a token's 8 experts are close to orthogonal (median cosine
  0.03 OLMoE, 0.02 Qwen3). A dropped expert's contribution lies in a direction the kept experts cannot
  reach, so no reweighting can stand in for it. This is why even the perfect-information ceiling is low.
- **The output is spread.** About 3.5 of 8 experts are effectively active per token (participation ratio); 95% of the output
  needs 5 to 6 experts and 99% needs 7 to 8.
- **Keep the gates; do not renormalise.** Simply keeping the router's weights is the best simple rule.
  Renormalising them when experts are dropped makes the error about 5 times larger at 2 experts and about 8 to 9 times larger at 1 expert.
- **The maths was right; the premise was not.** Our closed-form error prediction matches the measurements
  to about 1 to 2%, and pursuit picks near-optimal subsets. The method works as designed; the redundancy
  it needs is not in these models.

## 5. Issues found and how they were handled
| Issue | Effect | Handling |
|---|---|---|
| Expert outputs summed in float32, the model sums in bf16 | K0.2 failed first (7/20 identical answers) | Wrapper now combines exactly as the model; rerun passed 20/20 (logged deviation, no threshold changed) |
| K0 loaded Qwen3 in fp32 (about 122 GB) | Out of memory | fp32 check runs on OLMoE only, as the spec says |
| Native accuracy (K0.6) only measured inside K3 | K0 incomplete | Added a native-only run |
| Two method variants numerically unstable on real data (ours-clu on OLMoE, ours-avg on Qwen3) | Huge errors for those variants | Cannot change the verdict (the ceiling itself fails); must be fixed before any reuse of the code |
| First VM (RTX 5090, 32 GB) too small for Qwen3 | Delay | Moved to a 96 GB GPU |

All deviations are in `results/STATUS.md` and `docs/decisions.md`. No claim, model, threshold, task or
metric was changed after results were seen.

## 6. Cost and time
- GPU: about 5 h 15 min, about Rs 1,100 (P1 budget: 6 to 8 GPU-hours). About 45 minutes of that was idle
  time after the run (slow results copy, one failed destroy call).
- The instance has been destroyed; nothing is billing.

## 7. Where everything is
- Post-mortem with full numbers: `results/kill_test/POSTMORTEM.md`
- Raw results: `results/kill_test/<model>/` (`k0.json`, `k0_native.json`, `layer_eval.json`, `layer_eval.csv`)
- Status and deviations log: `results/STATUS.md`; implementation decisions: `docs/decisions.md`
- Code and configs: GitHub `thelast1coder/CPAL_2`
- Not kept: the 4.1 GB of per-token analysis records (only 207 MB copied before the instance was destroyed).
  Regenerating them takes about 1.5 GPU-hours.

## 8. Recommendation
Close this direction. No main-conference paper is possible from it: the core claim is false, and no
engineering change can fix that. The geometry findings (near-orthogonal experts, a low ceiling for
reweighting, keep-gates beats renormalisation) might make a short workshop note, but only after a
literature check for novelty (free, on the laptop) and two or three more models (about 3 to 5 GPU-hours).
My advice is to move the GPU budget to the next project.
