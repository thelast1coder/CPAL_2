# TINY RUN, NUMBERS ARE MEANINGLESS

Tiny random models and synthetic tokens; this report only exercises the pipeline and the decision logic.

# Kill test report

Verdict: **NO-GO**.
Plain summary (three lines): pipeline check only.

## Setup

- Prereg version 1; git 62dfda6223f3436246695d687ff823169cb405d3 (dirty: True).
- Models: olmoe_1b_7b, qwen3_30b_a3b. Precision and revisions: see results/STATUS.md (decision D7).

## K0 sanity

- olmoe_1b_7b: K0.1 pass, K0.3 pass, K0.4 pass, K0.5 pass, K0.2 pass, K0.3_task pass, K0.4_task pass, K0.5_task pass
- qwen3_30b_a3b: K0.1 pass, K0.3 pass, K0.4 pass, K0.5 pass, K0.2 pass, K0.3_task pass, K0.4_task pass, K0.5_task pass

## K1: redundancy

| model | share of layers with median m95 <= 3 | share >= 5 | cell |
|---|---|---|---|
| olmoe_1b_7b | 1.000 | 0.000 | GO |
| qwen3_30b_a3b | 1.000 | 0.000 | GO |

Per-layer median m95 and PR:

- olmoe_1b_7b: L0: 3 / 2.11, L1: 3 / 1.94, L2: 3 / 2.01, L3: 3 / 1.95
- qwen3_30b_a3b: L0: 3 / 2.07, L1: 3 / 2.11

## K2: exploitable with calibration

| model | primary | ratio m=1 | ratio m=2 | U1 m=1 / m=2 | U2 m=1 / m=2 | W-avg m=1 / m=2 | cell |
|---|---|---|---|---|---|---|---|
| olmoe_1b_7b | ours-cos | 1.005 | 1.006 | 1.138 / 1.309 | 1.033 / 1.068 | 1.000 / 1.000 | NO-GO |
  - validation error at m=2 per candidate: {'ours-avg': 0.13151390952344055, 'ours-cos': 0.1314953367659481, 'ours-clu': 0.1349648361605964}
  - rescue (ours-sk): {'1': 0.7978830699188341, '2': 0.7048656373005897}, counts: False
  - U1 (oracle) is below the 2x bar: no calibrated method can pass K2 on this model.
| qwen3_30b_a3b | ours-cos | 1.001 | 1.012 | 1.149 / 1.287 | 1.031 / 1.070 | 1.000 / 1.001 | NO-GO |
  - validation error at m=2 per candidate: {'ours-avg': 0.12703750373832096, 'ours-cos': 0.12696683311303655, 'ours-clu': 0.12818366944535228}
  - rescue (ours-sk): {'1': 0.7871142987135532, '2': 0.685199212740834}, counts: False
  - U1 (oracle) is below the 2x bar: no calibrated method can pass K2 on this model.

## K3: survives the network

| model | task | m* | B0 | B2 | B2+P | B3 | B3+P | ours-fixed | ours-adaptive | recovery [95% CI] | gain vs B2 (points) [CI] | ours - B3 (points) [CI] | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmoe_1b_7b | tiny_agreement | 1 | 100.0 | 23.6 | 44.4 | 42.8 | 44.4 | 44.0 | 56.8 | 0.267 [0.185, 0.346] | 20.4 [13.6, 27.2] | 1.2 [-3.2, 5.6] | AMBER |
  - n = 250; realised mean active experts: B0 4.00, B2_m1 1.00, B2_m2 2.00, B2_m3 3.00, B2+P 1.00, B3 1.00, B3+P 1.00, ours-fixed 1.00, ours-adaptive 1.52
  - perplexity: B0 244.613, B2_m1 273.248, ours-fixed_m1 180.538, B2_m2 256.105, ours-fixed_m2 202.999, B2_m3 244.963, ours-fixed_m3 236.762
| qwen3_30b_a3b | tiny_agreement | 1 | 100.0 | 31.2 | 49.2 | 49.6 | 49.2 | 50.4 | 58.0 | 0.279 [0.185, 0.366] | 19.2 [12.4, 26.0] | 0.8 [-2.0, 3.6] | AMBER |
  - n = 250; realised mean active experts: B0 4.00, B2_m1 1.00, B2_m2 2.00, B2_m3 3.00, B2+P 1.00, B3 1.00, B3+P 1.00, ours-fixed 1.00, ours-adaptive 1.49
  - perplexity: B0 231.917, B2_m1 191.149, ours-fixed_m1 170.219, B2_m2 273.383, ours-fixed_m2 216.778, B2_m3 240.814, ours-fixed_m3 232.437

## Decision table

| model | K1 | K2 | K3 |
|---|---|---|---|
| olmoe_1b_7b | GO | NO-GO | AMBER |
| qwen3_30b_a3b | GO | NO-GO | AMBER |

Overall: **NO-GO** (rule: K1 GO on both models; K2 GO on at least one and not NO-GO on the other; K3 GO on at least one and not NO-GO on the other => GO; K1 NO-GO on both, or K2 NO-GO on both after rescue, or K3 NO-GO on both => NO-GO; otherwise AMBER).

## What the numbers mean

(to be written after reading the numbers)
