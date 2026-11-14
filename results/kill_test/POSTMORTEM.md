# Kill test post-mortem: NO-GO (15 Nov 2026)

The kill test is NO-GO: the project's first premise, that a few experts carry almost all of each token's
output, is false on both models. The calibrated method cannot help, because even a method with perfect
per-token information gains only 1.1x to 1.2x over the best simple rule, against a 2x bar. K3 was not run.

## Numbers
Source: `results/kill_test/<model>/layer_eval.json` and `layer_eval.csv` (seed 0; analysis tokens 50k for
OLMoE, 30k for Qwen3; primary variant chosen on validation tokens as pre-registered). Prereg commit 1d7c06c.

| | OLMoE-1B-7B (16 MoE layers) | Qwen3-30B-A3B (48 MoE layers) | GO bar |
|---|---|---|---|
| K1: median m95 per layer (min / median / max) | 5 / 5 / 6 of 8 | 3 / 6 / 6 of 8 | <= 3 |
| K1: share of layers with median m95 <= 3 | 0% | 2% (1 layer) | >= 60% |
| K1: share of layers with median m95 >= 5 (NO-GO bar) | 100% | 92% | NO-GO if >= 60% |
| K2: primary variant | ours-avg | ours-cos | |
| K2: median ratio reference / ours, m = 1 / m = 2 | 1.03 / 1.05 | 1.03 / 1.04 | >= 2.0 |
| K2: U1 ceiling (exhaustive subset, oracle weights) | 1.17 / 1.23 | 1.11 / 1.14 | |
| K2: U2 (gate order, oracle weights) | 1.01 / 1.02 | 1.01 / 1.01 | |
| K2 rescue (ours-sk, sketch 5%) | 0.99 / 0.99 | 0.96 / 0.95 | >= 2.0 |
| Verdict per model | K1 NO-GO, K2 NO-GO | K1 NO-GO, K2 NO-GO | |

Overall (prereg rule): NO-GO, since K1 is NO-GO on both models and K2 is NO-GO on both after the rescue.

Mean relative error energy over layers (lower is better):

| Method | OLMoE m=1 | m=2 | m=4 | Qwen3 m=1 | m=2 | m=4 |
|---|---|---|---|---|---|---|
| B1 keep gates | 0.391 | 0.234 | 0.081 | 0.458 | 0.268 | 0.098 |
| B2 renormalise | 3.350 | 1.092 | 0.212 | 3.688 | 1.231 | 0.252 |
| primary ours | 0.381 | 0.228 | 0.095 | 0.482 | 0.296 | 0.120 |
| U1 ceiling | 0.321 | 0.182 | 0.062 | 0.396 | 0.224 | 0.081 |

## What the numbers say about MoE geometry
1. **Expert outputs are close to orthogonal.** The median cosine between the outputs of a token's 8
   experts is 0.03 (OLMoE) and 0.02 (Qwen3), and the smallest eigenvalue of the normalised per-token Gram
   is about 0.85 in a typical layer. A dropped expert's contribution lies in a direction the kept experts
   cannot reach, so reweighting the kept experts recovers almost nothing (U2 and W-avg are within 2% of B1).
2. **The output energy is spread.** The effective number of experts (participation ratio) is about 3.4 of
   8 in a typical layer; 95% of the energy needs 5 to 6 experts and 99% needs 7 to 8.
3. **Keeping the gates is the right simple rule; renormalising is harmful.** B2 overshoots badly at small m
   (error 3.4 to 3.7 at m = 1), because scaling up near-orthogonal parts adds error. On Qwen3 the literal
   k2 fix (B3) equals B1, as expected for a router trained with renormalisation.
4. **The theory is right; the premise is not.** The closed-form prediction of PROJECT.md 5.2 matches the
   measured layer errors to a median of 0.7% (OLMoE) and 1.8% (Qwen3). Pursuit is near-optimal for its
   budget (greedy-to-oracle gap 0 at m = 1 and at most 0.0075 at m = 2, over all layers of both models). The method works as designed; the redundancy it
   needs is not there.
5. **A few layers differ.** Some early layers show higher cosines (up to 0.22 OLMoE, 0.55 Qwen3) and one
   Qwen3 layer has median m95 = 3. Too few to matter for whole-model compute.

## Problems found during the run (none changes the verdict)
- K0 needed three fixes before passing (expert combine precision, fp32 load for Qwen3, K0.6 timing); all
  logged in `docs/decisions.md` and the STATUS deviations log. No threshold, model, task or metric changed.
- Two Gram variants are numerically unstable on real data: ours-clu on OLMoE (error about 22 to 28 at
  m >= 2) and ours-avg on Qwen3 (error up to about 3,300 at m = 2 and 3). The least-squares solve blows up,
  most likely when a calibrated Gram is ill-conditioned for some token subsets. A working version of the
  broken variant might have been chosen as primary instead (the chosen variants are stable), but this
  cannot affect the verdict, because the U1 ceiling, which no calibrated variant can beat, is itself far
  below 2x. It must be fixed (stronger ridge or conditioning check)
  before any reuse of the code.
- The per-token analysis records (4.1 GB) were only partly copied off the VM (207 MB) before the instance
  was destroyed; the summary results above are complete. Further geometry analysis would need a re-run of
  calibration and recording (about 1.5 GPU-hours).

## Cost
Jarvis Labs RTX PRO 6000 (96 GB), about 5 h 15 min and about Rs 1,100, against a P1 budget of 6 to 8
GPU-hours. About 45 minutes of that was idle time after the run (slow copy, one failed destroy call).

## Recommendation
Stop this line (projection routing as a compute saving at fixed quality). The plan's special case
(K1 GO, K2 and K3 NO-GO) does not apply, since K1 itself failed; the measurement could still make a short
workshop note on MoE output geometry (near-orthogonal experts, spread energy, keep-gates beats
renormalisation), if Dawar wants it. Otherwise move the GPU time to the next project.
