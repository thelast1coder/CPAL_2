# PREREG_KILL_TEST.md: pre-registration of the kill test

Status: DRAFT until committed. Commit this file and `configs/prereg_kill_test.yaml` in their own
commit before the first real-model run, and write the hash into `results/STATUS.md`. After that,
nothing in Sections 1 to 6 changes. Anything done differently goes in Section 7 with a reason.

## 1. Hypotheses
- **H-K1 (redundancy):** in fine-grained MoE LLMs, a few experts reproduce at least 95% of the
  layer output's energy for most tokens (median m95 of 3 or fewer of 8 in most MoE layers).
- **H-K2 (exploitable with calibration):** pursuit selection plus least-squares weights from a Gram
  matrix estimated once on calibration text gives at least 2x lower layer error than the better of
  keeping the gates and renormalising them, at 1 and 2 kept experts.
- **H-K3 (survives the network):** with the reduction in every MoE layer, the method recovers at
  least half of the accuracy that renormalised top-m loses at the pre-registered budget m*.

## 2. Models, data and splits
OLMoE-1B-7B-0924 (base for K1 and K2 statistics; Instruct for K3) and Qwen3-30B-A3B (non-thinking
mode for K3; bf16 or FP8 by decision D7, fixed before K0). Calibration mix v1 (60% FineWeb-Edu,
40% MetaMathQA, chat-formatted), sequence length 1024, seed 0, split by document into calibration
(1M tokens), validation (20k) and analysis (50k OLMoE, 30k Qwen3), all disjoint and free of GSM8K
and MATH test questions.

## 3. Measurements
- m95(x): smallest m whose best subset of T with oracle weights leaves relative error <= 0.05,
  computed from the true per-token Gram.
- Relative error energy of a method in a layer: sum over analysis tokens of e(x) divided by sum of
  ||y(x)||^2, with e from the quadratic form (PROJECT.md 5.1).
- K2 reference per layer: the lower error energy of B1 (keep gates) and B2 (renormalise).
- Primary "ours": OLS pursuit plus least squares with the Gram variant (G-avg, G-cos or G-clu with
  C = 16) that has the lowest validation error energy at m = 2, averaged over layers. Chosen on
  validation tokens only.
- K3 task: GSM8K fixed-250 subset (seeded indices committed in configs/), zero-shot chat prompt,
  greedy, max_new_tokens 512, extraction as in PHASE0_SETUP.md 2.8. For OLMoE-Instruct, if native
  accuracy on this subset is below 25%, the task is ARC-Challenge instead (decided from the native
  number only).
- m*: the smallest m in {1, 2, 3} at which B2 loses at least 5.0 points against native on the K3
  task; if none, m* = 1.
- Recovery: (acc(ours-fixed) - acc(B2)) / (acc(B0) - acc(B2)) at m*. Gain interval: paired bootstrap,
  10,000 resamples, seed 0, 95%.

## 4. Thresholds (machine-readable copy; must equal configs/prereg_kill_test.yaml, test T14)
```yaml
prereg_version: 1
models: [olmoe_1b_7b, qwen3_30b_a3b]
splits:
  calibration_tokens: 1000000
  validation_tokens: 20000
  analysis_tokens: {olmoe_1b_7b: 50000, qwen3_30b_a3b: 30000}
  seq_len: 1024
  seed: 0
k1:
  statistic: share_of_moe_layers_with_median_m95_at_most
  m95_level: 0.95
  go: {median_m95_at_most: 3, layer_share_at_least: 0.60}
  nogo: {median_m95_at_least: 5, layer_share_at_least: 0.60}
k2:
  budgets: [1, 2]
  metric: relative_error_energy
  reference: better_of_B1_B2_per_layer
  candidates: [ours_avg, ours_cos, ours_clu16]
  choose_on: validation_mean_over_layers_at_m2
  aggregate: median_over_moe_layers_of_reference_over_ours_ratio
  go_ratio_at_least: 2.0
  amber_ratio_at_least: 1.3
  rule: "GO if ratio >= 2.0 at both budgets; AMBER if >= 1.3 at both but not GO; NO-GO otherwise"
  rescue:
    variant: ours_sk
    sketch_fraction: 0.05
    allowed_only_if_primary_not_go: true
    extra_condition: "rescue error <= reference error at m + 1"
k3:
  task: gsm8k_fixed250
  olmoe_task_switch: {to: arc_challenge, if_native_accuracy_below: 0.25}
  m_star_candidates: [1, 2, 3]
  m_star_min_b2_loss_points: 5.0
  primary: recovery_of_ours_fixed_over_B2
  go: {recovery_at_least: 0.50, gain_ci_excludes_zero: true, ours_minus_b3_ci_lower_at_least_points: -1.0}
  amber: {recovery_at_least: 0.25, gain_ci_excludes_zero: true}
  nogo: "recovery below 0.25 or gain interval includes zero"
  note_b3: "the B3 condition applies only if B3 is implementable; if GO holds except the B3 condition, the cell is AMBER"
  bootstrap: {resamples: 10000, level: 0.95, seed: 0, paired: true}
  amber_extension: {allowed_once: true, extra_questions: 250, subset: gsm8k_fixed500}
decision:
  go: "K1 GO on both models; K2 GO on at least one and not NO-GO on the other; K3 GO on at least one and not NO-GO on the other"
  nogo: "K1 NO-GO on both, or K2 NO-GO on both after rescue, or K3 NO-GO on both"
  amber: "anything else"
```

## 5. Secondary analyses (reported, not used for the decision)
PR and mean cosine by layer; U1 and U2 headroom; W-avg (weights alone); m = 3 and 4 layer errors;
B2+P and B3+P recovery; ours-fixed minus B3; WikiText-2 perplexity at m in {1, 2, 3}; realised
active experts of ours-adaptive on the analysis tokens; the closed-form prediction of PROJECT.md 5.2
against the measured errors.

## 6. What may not happen
No tuning of any kind on benchmark items. No change of task, budget, threshold, model or Gram
candidate list after seeing reduced-method results. No dropping of layers, questions or tokens after
seeing results (padding is excluded by design, before any run). The only pre-registered adaptations
are the OLMoE task switch (native accuracy only), the sketch rescue, the K3 extension, and the
AMBER fix window, which needs Dawar's approval and is reported as exploratory.

## 7. Deviations log
| Date | What changed | Why | Effect on the decision |
|---|---|---|---|
| | | | |
