# Implementation decisions (dated)

Project-level decisions D1 to D13 are in PROJECT.md Section 13. These are implementation details
(CLAUDE.md: "Implementation details are yours to decide; log them in docs/decisions.md").

| Date | Decision | Reason |
|---|---|---|
| 2026-11-10 | Repository root is the folder the zip was unpacked into (`CPAL_2`), not a new `MoE_CPAL` folder | Spec files were already there; same layout |
| 2026-11-10 | Laptop has 41 GB free (not 0.5 GB) and no system PyTorch: created `.venv` with CPU torch 2.15.0+cpu from the PyTorch CPU index | The "reuse system PyTorch" path was impossible; no model or dataset downloaded |
| 2026-11-10 | Git for Windows 2.56 installed with winget (with Dawar's approval) for git and bash | Needed for commits and `run_kill_test.sh` |
| 2026-11-10 | Pin `transformers==5.22.2` (latest stable on 10 Nov 2026) | Natively supports OLMoE, Qwen3-MoE, Qwen2-MoE and DeepSeek-V2. All four store experts as fused 3D tensors (`gate_up_proj` [N, 2I, d] concatenated gate then up; `down_proj` [N, d, I]) with a separate router module `block.gate`, and blocks return a plain tensor. Adapters are written against this code and raise `UnknownLayoutError` on any other layout (ModuleList experts, transposed or interleaved weights, biases) |
| 2026-11-10 | Reference experts kernel is `eager` (`config._experts_implementation = "eager"`); the default in 5.22.2 is `grouped_mm` | With eager, the patched native model is bit-identical to the original (difference 0.0); against grouped_mm the difference is about 1.4e-6 in fp32 (summation order), above T1's 1e-6 bar. Our wrapper loops over experts like the eager kernel, in the same (slot, token) order. Correctness first (D5) |
| 2026-11-10 | Gates follow each router exactly: OLMoE/Qwen softmax in fp32, top-k, optional renormalisation, cast to the logits dtype; DeepSeek-V2 fp32 logits, greedy or group-limited top-k (unsorted), times routed_scaling_factor, kept in fp32 | Read from the installed modelling code; T2 checks equality |
| 2026-11-10 | All expert combination in the wrapper is accumulated in float32, then cast | PHASE0_SETUP.md 2 |
| 2026-11-10 | When a policy keeps the whole set T, the weights are the native gates exactly (not the ridge solve) | Exact optimum whatever the ridge; makes K0.5 exact |
| 2026-11-10 | G-cos uses RMS norms (sqrt of the G-avg diagonal) as the calibrated mean norms of dropped experts | Consistent with G-avg: n_i^2 = G-avg_ii |
| 2026-11-10 | Experts never seen in T during calibration get the layer's mean squared norm on the diagonal | Back-off for the diagonal is not specified in PROJECT.md 5.4 |
| 2026-11-10 | Cluster-conditioned Grams back off entry-wise to the layer G-avg when the pair count in the cluster is below n_min | PROJECT.md 5.4 ("backing off to the layer's G-avg when counts are low") |
| 2026-11-10 | k-means: seeded spherical mini-batch k-means (k-means++ init on up to 20k points, 100 mini-batch steps of 4,096) fitted on a seeded reservoir sample of at most 32,768 pass-1 inputs per layer (fp16 on CPU) | Bounded memory for 48 layers; deterministic |
| 2026-11-10 | Greedy OLS is implemented by rank-one updates of the projected Gram (O(k^2) per step); OMP uses the same state with the score bp_j^2 / G_jj | PHASE0_SETUP.md 2.3; T6 checks against brute force |
| 2026-11-10 | Adaptive stop compares the relative residual with max(tau, 100 eps) | tau = 0 must stop at numerical zero for rank-deficient Grams (T7) |
| 2026-11-10 | `AddProjection` (B2+P, B3+P) uses G-avg for selections inside T and the full-Gram calibration for selections outside T | The primary variant may be G-cos or G-clu; using G-avg for the add-on keeps it one-pass. Revisit in P3 |
| 2026-11-10 | K0 builds a throwaway G-avg from its own 4 x 256 tokens for K0.5, so K0 runs before calibration as in the spec order | K0.5 only needs a Gram object; with m = k the weights are the native gates |
| 2026-11-10 | The K3 task model (OLMoE-Instruct) gets its own calibration and validation records (`--target task`); the primary Gram variant is chosen on the stats model's validation records (prereg) and reused for K3 | The Instruct weights differ from the base model's, so the base Gram does not apply |
| 2026-11-10 | Data split by documents: one 1024-token chunk per text document, one example per maths sequence; token quotas by the stats model's tokenizer; documents shared across families | PHASE0_SETUP.md 2.7; disjointness re-checked by document hash |
| 2026-11-10 | Leakage check: drop maths documents whose normalised question equals a GSM8K or MATH-500 test question, and any document containing a test question (normalised, length > 40 characters) as a substring | "exact-match check on normalised question text" |
| 2026-11-10 | GSM8K fixed subsets: first 250 and 500 of `torch.randperm(1319, generator=manual_seed(0))`, sorted; fixed500 contains fixed250 (`configs/gsm8k_fixed250.json`, `gsm8k_fixed500.json`) | The K3 extension adds the next 250 questions |
| 2026-11-10 | Tiny run: K3 stand-in task is agreement with the native model's greedy next token at the last position of 250 synthetic sequences | A random tiny model has no task skill; agreement exercises recovery and the bootstrap |
| 2026-11-10 | Realised budget counts computed (token, expert) pairs with non-zero weight on valid tokens, per layer, then averaged over layers | PROJECT.md 9 |
| 2026-11-11 | Model IDs, routing fields and dataset IDs checked against the Hugging Face API and config.json files (metadata only, nothing downloaded); revisions pinned to HF main as of 11 Nov 2026, with a separate `revision_fp8` for the Qwen3 FP8 checkpoint | Removes the (verify) marks the VM step would have resolved; the same weights are used whenever P1 starts. `download_models.py` now downloads the pinned revision and reports if HF main has moved |
| 2026-11-11 | All kill-test models and datasets are public and not gated: no Hugging Face token is needed for P1 | Checked with the HF API (`gated: false`); avoids handling a token on a shared machine |
| 2026-11-11 | Every GPU script with `--mem-gb` first waits until that much memory (+1 GB) is free, polling each minute (limit `PROUTING_GPU_WAIT_MAX_MIN`, default 720) | The GPU is shared; queueing instead of starting avoids out-of-memory failures for us and for the other projects |
| 2026-11-11 | `run_kill_test.sh --stop-after k0` runs data preparation and K0 only | CLAUDE.md: stop and report at K0 |
| 2026-11-11 | `vm_setup.sh` uses uv when available and builds the vLLM venv only with `WITH_VLLM=1` | vLLM is needed only from V1 in P2; saves time and disk on the VM |
| 2026-11-11 | `scripts/p1_start.sh` drives a fresh VM to the K0 gate under nohup; `scripts/pull_results.sh` copies results to the laptop | vast.ai instances may have no persistent volume (the first one did not) |
