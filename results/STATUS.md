# STATUS

## Now
**Kill test: NO-GO (15 Nov 2026).** K1 and K2 are NO-GO on both models (prereg rule); K3 not needed and
not run. Post-mortem: results/kill_test/POSTMORTEM.md. VM 2 destroyed (Jarvis instance 574523, about 5 h 15 min,
about Rs 1,100). Waiting for Dawar's decision on what comes next.

## K1 and K2 (14 Nov 2026, 18:27 to 19:06 UTC): NO-GO on both models
| | OLMoE | Qwen3-30B-A3B | GO bar |
|---|---|---|---|
| K1 share of layers with median m95 <= 3 | 0% (median m95 5 of 8) | 2% (median m95 6 of 8) | >= 60% |
| K2 ratio at m = 1 / 2 (primary) | 1.03 / 1.05 (ours-avg) | 1.03 / 1.04 (ours-cos) | >= 2.0 |
| U1 ceiling | 1.17 / 1.23 | 1.11 / 1.14 | |
| Rescue ours-sk | 0.99 / 0.99 | 0.96 / 0.95 | >= 2.0 |
Files: results/kill_test/<model>/layer_eval.json and .csv.

## K0 (14 Nov 2026): PASS on both models. Stopped at the gate; waiting for Dawar's go-ahead for K1 to K3.
| Check | OLMoE (base fp32 / Instruct bf16) | Qwen3-30B-A3B (bf16) | Bar |
|---|---|---|---|
| K0.1 wrapper exact, fp32 logits | 0.0 | not run (OLMoE only, spec) | <= 1e-4 |
| K0.2 identical greedy outputs (20 GSM8K prompts) | 20/20 | 20/20 | >= 19/20 |
| K0.2 WikiText-2 perplexity, native vs wrapped | 10.1924 vs 10.1924 (diff 0.0) | 8.7073 vs 8.7073 (diff 0.0) | within 0.2% |
| K0.3 Gram identity, max rel diff | 5.8e-7 (fp32), 3.7e-7 (bf16) | 3.6e-7 | <= 1e-3 |
| K0.4 offline vs online, rel gap (middle layer) | 1.9e-7 (fp32), 2.1e-3 (bf16) | 9.5e-4 | <= 1% |
| K0.5 full set reproduces native, rel error | 0.0 / 0.0 | 0.0 | <= 1e-4 / 1e-3 |
| K0.6 native GSM8K fixed-250 accuracy | 48.8% (OLMoE-Instruct) | 93.2% (non-thinking) | recorded |
- K0.6 vs model cards: OLMoE-Instruct card GSM8K 45.5 (8-shot CoT, +DPO row); ours 48.8 zero-shot on the
  fixed 250: consistent. OLMoE stays on GSM8K for K3 (rule: switch to ARC only below 25%). The Qwen3 card has
  no GSM8K figure (points to the blog and technical report) (verify there).
- Results pulled to the laptop: vm_pull/ (k0.json, k0_native.json, k3/B0.json(l), logs). Failed first OLMoE
  record kept as k0_run1_failed.json.
- Record note: Qwen3 k0.json shows git cb1e612 because the repo was updated during its run; the K0 code was
  identical at 04b08e7 and cb1e612 (only run_downstream.py changed).

## Decision D7 (fixed 14 Nov 2026, before K0)
- Qwen3-30B-A3B runs in **bf16** for every number. Reason: the prereg rule is bf16 if at least 70 GB of GPU
  memory can be reserved; VM 2 has 96 GB (97,887 MiB) with 0 MiB in use and no other processes at start.
- Memory caps: OLMoE 40 GB, Qwen3 75 GB; every GPU step waits for free memory first.

## Prereg
- PREREG_KILL_TEST.md + configs/prereg_kill_test.yaml commit: `1d7c06c62b7336aeed70ac4285b70908c6151b7a`
  (10 Nov 2026, before any real-model run). Thresholds fixed from this commit.

## Done (P0, laptop, 10 Nov 2026)
- Repository, environment (.venv, CPU torch 2.15.0, transformers 5.22.2 pinned).
- Literature notes and novelty gate: NOT TRIPPED (docs/related_work_notes.md).
- All modules, scripts and configs; CPU tests; tiny end-to-end run (results/tiny/KILL_REPORT.md, TINY).

## Next (P1, on an 80 GB+ GPU)
1. Decide D7 with Dawar from the memory that can be reserved: bf16 (needs about 75 GB) or FP8 (about 45 GB).
   Record it here with the free GPU memory.
2. On the VM: `git clone https://github.com/thelast1coder/CPAL_2 MoE_CPAL && cd MoE_CPAL`, then
   `mkdir -p logs && nohup bash scripts/p1_start.sh --qwen <bf16|fp8> > logs/p1_start.log 2>&1 &`.
   This runs setup and CPU tests, the downloads (pinned revisions; no HF token needed), data, and K0 for
   both models, then stops. Pull results with `bash scripts/pull_results.sh <port> root@<host>`.
3. Report K0. After Dawar's go-ahead: `bash scripts/run_kill_test.sh --model olmoe --mem-gb 40`, then
   `--model qwen3`, then the kill report.

## Ready before the VM (11 Nov 2026)
- Model IDs, routing fields, dataset IDs and splits, GSM8K test size (1319) and the Qwen3
  `enable_thinking` flag checked against Hugging Face; revisions pinned (configs/models/*.yaml).
- GPU queue: every GPU step waits for free memory before starting. K0 stop: `--stop-after k0`.

## Decisions from Dawar (15 Nov 2026, before any K3 result)
- B3 on OLMoE: the B3 of record is the literal k2 = k (as published); B1 (= the k2 = N reading) is also
  run and reported at m* on OLMoE's K3 task (configs/experiments/kill_test.yaml: b1_at_mstar_models).
- Scope: run K1 and K2 now (`--stop-after k2`), the minimum needed; K3 only after the K2 report and go-ahead.
- Instance lifecycle (Jarvis Labs CLI `jl`, laptop venv ~/.jlcli): when K1 and K2 finish, copy results to
  the laptop; if the verdict makes K3 unnecessary (NO-GO), destroy the instance at once; if GO or AMBER,
  pause it (storage still billed) and destroy it after K3. Needs `jl setup` with Dawar's API key.

## Open questions for Dawar
- None.

## Blockers
- None. Next phase needs Dawar's decision (the plan says stop this line on NO-GO).

## VM notes
- VM 1 (11 Nov 2026): vast.ai instance 60187060, 1x RTX 5090 32 GB, 30 GB RAM, 8 CPUs, 350 GB disk,
  driver 580.178.04 (CUDA 13.0). No persistent volume: everything on it is lost on recycle or destroy.
  Repo cloned to /workspace/MoE_CPAL from a git bundle. Shared with other projects.
- VM 2 (14 Nov 2026): Jarvis Labs VM (ubuntu@151.185.34.98), 1x RTX PRO 6000 Blackwell Server Edition
  96 GB, driver 595.91.07 (CUDA 13.2), 157 GB RAM, 28 CPUs, 242 GB disk (228 GB free), Ubuntu 24.04.
  Repo at ~/MoE_CPAL (cloned from GitHub at c2fb245), HF_HOME=~/hf_home. Installed python3.12-venv (apt)
  and uv 0.13.8. Env: torch 2.15.0+cu132, transformers 5.22.2. CPU tests on the VM: 38 passed (59.5 s).
  No HF token (all models and datasets public).
- Repository: https://github.com/thelast1coder/CPAL_2 (origin; P0 pushed at 62dfda6).

## Deviations log
| Date | What changed | Why | Effect on the decision |
|---|---|---|---|
| 2026-11-14 | Expert combine in the wrapper now in the model's dtype, mirroring transformers' eager loop (was float32; PHASE0_SETUP.md 2.2) | First K0 run on OLMoE: K0.2 failed (7 of 20 identical bf16 greedy outputs; perplexity within 0.07%) because native accumulates in bf16. Implementation fix, logged in docs/decisions.md | None: no threshold, model, task or metric changed; K0 rerun from scratch after the fix |

## Log
- 2026-11-10: P0 complete. pytest: 32 passed (CPU, clean state, 8 min 53 s including the tiny end-to-end run). Tiny report: results/tiny/KILL_REPORT.md (TINY, meaningless).
- 2026-11-11: VM 1 (RTX 5090 32 GB) received. Qwen3-30B-A3B cannot fit; P1 paused until an 80 GB+ GPU.
  Environment setup and CPU tests run on VM 1 only; no model downloads there.
- 2026-11-11: Pre-VM preparation: IDs checked and revisions pinned, GPU queue, `--stop-after k0`,
  `p1_start.sh`, `pull_results.sh`. pytest: 38 passed (CPU, 4 min 42 s). Tiny run with `--stop-after k0`: all K0 checks pass.
- 2026-11-14: VM 2 (RTX PRO 6000 Blackwell 96 GB, Jarvis Labs). D7 = bf16. P1 started to K0.
- 2026-11-14 14:45 UTC: first K0 on OLMoE: K0.1, K0.3, K0.4, K0.5 pass (K0.1 logit difference 0.0 in fp32); K0.2 FAIL (7/20 identical, ppl rel diff 0.0007). Cause: float32 vs bf16 expert combine. Fixed; K0 rerun. Failed record kept as results/kill_test/olmoe_1b_7b/k0_run1_failed.json on the VM.
- 2026-11-14: Qwen3 K0 first attempt ran out of memory (fp32 load for K0.1); fixed (04b08e7). K0.6 added as `--native-only` (cb1e612). K0 PASS on both models; native GSM8K 48.8% (OLMoE-Instruct), 93.2% (Qwen3).
- 2026-11-14 19:06 UTC: K1 and K2 done on both models: NO-GO. Results copied (summary files complete; per-token records only 207 MB of 4.1 GB). Automatic destroy failed on a network error; instance destroyed by hand at about 19:50 UTC.
- 2026-11-15: Post-mortem written (results/kill_test/POSTMORTEM.md). Found numerically unstable Gram variants on real data (ours-clu OLMoE, ours-avg Qwen3); does not affect the verdict.
