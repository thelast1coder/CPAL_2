# STATUS

## Now
P0 done, waiting for VM.

## Prereg
- PREREG_KILL_TEST.md + configs/prereg_kill_test.yaml commit: `1d7c06c62b7336aeed70ac4285b70908c6151b7a`
  (10 Nov 2026, before any real-model run). Thresholds fixed from this commit.

## Done (P0, laptop, 10 Nov 2026)
- Repository, environment (.venv, CPU torch 2.15.0, transformers 5.22.2 pinned).
- Literature notes and novelty gate: NOT TRIPPED (docs/related_work_notes.md).
- All modules, scripts and configs; CPU tests; tiny end-to-end run (results/tiny/KILL_REPORT.md, TINY).

## Next (P1, VM)
1. `bash scripts/vm_setup.sh` (all CPU tests must pass), `hf auth login` with a fresh token.
2. Record here: PREREG commit hash (above), free GPU memory, decision D7 (bf16 or FP8 for Qwen3).
3. Downloads, fill configs/models/*.yaml (`--fill`), then `bash scripts/run_kill_test.sh --model olmoe`
   while Qwen3 downloads; then `--model qwen3`.

## Open questions for Dawar
- B3 (k2 fix) on OLMoE: the router does not renormalise, so the paper's "trained value" corresponds to
  k2 = N (= keep gates). Default in the code is the literal k2 = k. Which do you want before K3?
- GitHub repository URL (not yet given; nothing pushed).

## Blockers
- None on the laptop. The VM is needed for everything else.

## Deviations log
(none)

## Log
- 2026-11-10: P0 complete. pytest: 32 passed (CPU, clean state, 8 min 53 s including the tiny end-to-end run). Tiny report: results/tiny/KILL_REPORT.md (TINY, meaningless).
