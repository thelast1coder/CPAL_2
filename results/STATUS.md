# STATUS

## Now
P0 done. P1 paused: the first VM (RTX 5090, 32 GB) is too small for Qwen3-30B-A3B. Dawar chose (11 Nov
2026) to move the whole P1 run to an 80 GB+ GPU. The 5090 is used only for environment setup and CPU tests.

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
- An 80 GB+ GPU (H100 class) for P1. Qwen3-30B-A3B needs about 61 GB in bf16 and about 31 GB in FP8
  for the weights alone, so it does not fit on the 32 GB RTX 5090.

## VM notes
- VM 1 (11 Nov 2026): vast.ai instance 60187060, 1x RTX 5090 32 GB, 30 GB RAM, 8 CPUs, 350 GB disk,
  driver 580.178.04 (CUDA 13.0). No persistent volume: everything on it is lost on recycle or destroy.
  Repo cloned to /workspace/MoE_CPAL from a git bundle. Shared with other projects.
- Repository: https://github.com/thelast1coder/CPAL_2 (origin; P0 pushed at 62dfda6).

## Deviations log
(none)

## Log
- 2026-11-10: P0 complete. pytest: 32 passed (CPU, clean state, 8 min 53 s including the tiny end-to-end run). Tiny report: results/tiny/KILL_REPORT.md (TINY, meaningless).
- 2026-11-11: VM 1 (RTX 5090 32 GB) received. Qwen3-30B-A3B cannot fit; P1 paused until an 80 GB+ GPU.
  Environment setup and CPU tests run on VM 1 only; no model downloads there.
- 2026-11-11: Pre-VM preparation: IDs checked and revisions pinned, GPU queue, `--stop-after k0`,
  `p1_start.sh`, `pull_results.sh`. pytest: 38 passed (CPU, 4 min 42 s). Tiny run with `--stop-after k0`: all K0 checks pass.
