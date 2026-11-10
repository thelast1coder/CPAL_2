# KICKOFF.md: what to paste into Claude Code

Put all files from this zip in an empty folder called `MoE_CPAL` on the laptop, open Claude Code in
that folder, and paste the prompt below.

## Prompt for P0 (laptop, now)
```
Read CLAUDE.md, then PROJECT.md, EXPERIMENTS.md, PHASE0_SETUP.md, PHASE1_KILL_TEST.md and
PREREG_KILL_TEST.md in full before writing anything.

We are in Phase P0 on my laptop: no GPU and only about 0.5 GB of free disk. Do not download any
model, dataset or weights here. Do all of P0 as PHASE0_SETUP.md describes:
1. Set up the repository and the small environment (reuse the system PyTorch).
2. Write docs/related_work_notes.md and apply the novelty gate. If the gate trips, stop and tell me.
3. Write every module, script and config in Section 2, and every test in Section 4.
4. Run the tests on CPU and the tiny end-to-end run (scripts/run_kill_test.sh --tiny).
5. Commit PREREG_KILL_TEST.md with configs/prereg_kill_test.yaml in their own commit and record the hash.
Ask me for the private GitHub repo URL when you need it. Use a task list. When P0 is complete, stop
and report: what is ready, test results, the novelty-gate outcome, anything marked (verify), and the
exact first commands for the VM.
```

## Prompt for P1 (when the VM is ready)
```
The H100 VM is ready: <how to reach it, the path for HF_HOME>. The GPU is shared, so check
nvidia-smi first. Start Phase P1 exactly as PHASE1_KILL_TEST.md describes, with the thresholds in
PREREG_KILL_TEST.md. Run K0 first and stop if any K0 check fails. Keep results/STATUS.md current.
When the kill test finishes, write results/KILL_REPORT.md, apply the decision table, and stop and
report to me. Do not start P2 without my go-ahead.
```
