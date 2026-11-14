# STATUS

## Now
P1 running on VM 2 (Jarvis Labs, RTX PRO 6000 Blackwell 96 GB), started 14 Nov 2026 14:37 UTC:
`scripts/p1_start.sh --qwen bf16` (setup, downloads, data, K0 for both models, then stop for the K0 report).

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

## Open questions for Dawar
- B3 (k2 fix) on OLMoE: the router does not renormalise, so the paper's "trained value" corresponds to
  k2 = N (= keep gates). Default in the code is the literal k2 = k. Which do you want before K3?

## Blockers
- None.

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
