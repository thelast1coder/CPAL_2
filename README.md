# MoE_CPAL: projection routing for Mixture-of-Experts LLMs

Training-free reduction of active routed experts: choose kept experts by greedy pursuit on a calibrated
Gram matrix of expert outputs and give them least-squares (projection) weights. CPAL 2027, paper 2.
Specification: CLAUDE.md, PROJECT.md, EXPERIMENTS.md, PHASE0_SETUP.md, PHASE1_KILL_TEST.md, PREREG_KILL_TEST.md.

## Layout
```
src/prouting/
  adapters/     one adapter per MoE family (transformers==5.22.2 fused-expert layout), registry
  routing/      linalg.py (batched Gram algebra), policies.py (ours, B0 to B2), baselines.py (B3 to B9),
                block.py (ReducedMoEBlock), patch.py (patch_model / restore)
  calibrate/    stats.py (GramAccumulator, GramState), clusters.py, sketch.py, data.py, driver.py
  analysis/     recorder.py, layer_eval.py (offline methods), law.py (m95, PR), theory.py
  eval/         gsm8k.py, math500.py, ppl.py, lm_eval_bridge.py, stats.py (bootstrap), prompts.py
  vllm_plugin/  routing_fn.py (pure torch, tested on CPU), register.py (VM only)
  utils/        gpu.py, logging.py (run records), seed.py, io.py, status.py, tiny.py, cli.py
configs/        models/*.yaml, experiments/kill_test.yaml, prereg_kill_test.yaml, gsm8k_fixed{250,500}.json
scripts/        run_kill_test.sh and its steps, vm_setup.sh, download_models.py, check_vllm_equivalence.py
tests/          CPU tests T1 to T16
docs/           related_work_notes.md (novelty gate), decisions.md
results/        STATUS.md, small result files (committed); artifacts/ and logs/ are gitignored
```

## Laptop (CPU tests, tiny run)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows path
.venv/Scripts/python -m pip install -r requirements-cpu.txt
.venv/Scripts/python -m pytest -q                       # T1 to T16 (T16 runs the tiny pipeline, about 6 minutes)
bash scripts/run_kill_test.sh --tiny                     # writes results/tiny/KILL_REPORT.md (TINY, meaningless)
```

## VM: first commands (PHASE1_KILL_TEST.md Sections 1 and 2)
```bash
git clone <private repo URL> MoE_CPAL && cd MoE_CPAL
export HF_HOME=/path/on/vm/big_disk/hf        # all downloads go here
df -h "$HF_HOME"                               # need about 150 GB free
nvidia-smi                                     # see what Mirror_CPAL and the CVPR job are using
bash scripts/vm_setup.sh                       # venvs, installs, versions, pytest -q (must all pass)
hf auth login                                  # fresh token (verify the command); the old one is revoked
# record in results/STATUS.md: PREREG commit hash, free GPU memory, decision D7

source .venv/bin/activate
python scripts/download_models.py --model qwen3_30b_a3b --fill &   # start the big one first (or --fp8, D7)
python scripts/download_models.py --model olmoe_1b_7b --fill
python scripts/download_models.py --model olmoe_1b_7b_instruct --fill
git diff configs/models                         # check every (verify) field against config.json

bash scripts/run_kill_test.sh --model olmoe --mem-gb 40   # K0 first; stops if any K0 check fails
bash scripts/run_kill_test.sh --model qwen3 --mem-gb 75   # after the download and D7
```
Each step skips finished outputs, so rerunning the same command resumes. `--force` recomputes.

## Where results appear
- `results/kill_test/<model>/`: k0.json, calibration and recording summaries, layer_eval.json and .csv (K1, K2),
  tau_task.json, k3/ (every generation and per-config result), k3.json.
- `results/kill_test/KILL_REPORT.md` and `kill_test.json`: decision table applied from the prereg YAML.
- `plots/`: k1_m95_by_layer.png, k2_error_vs_m_<model>.png (needs matplotlib).
- `artifacts/<model>/`: calibration statistics and per-layer analysis shards (gitignored, 1 to 2 GB per model).
