# PHASE0_SETUP.md: repository, literature check, all code and CPU tests (no GPU)

Goal: when Dawar switches on the VM, the kill test runs with three commands and no new code.
Everything below runs on the laptop (no GPU, about 0.5 GB free disk) unless marked VM.

**Exit criteria for P0**
1. Repository created and pushed, with the layout in 1.2.
2. `docs/related_work_notes.md` written and the novelty gate (Section 3) passed.
3. All modules in Section 2 written; all tests in Section 4 pass on CPU (if the laptop cannot install
   the packages, the tests are written and run as the first step on the VM).
4. `bash scripts/run_kill_test.sh --tiny` runs end to end on a tiny random model and writes
   `results/tiny/KILL_REPORT.md` (clearly labelled TINY, numbers meaningless).
5. `PREREG_KILL_TEST.md` and `configs/prereg_kill_test.yaml` committed; the test that checks they
   agree passes.
6. README has the exact VM commands. `results/STATUS.md` says "P0 done, waiting for VM".
Then stop and report to Dawar.

---

## 1. Repository setup

### 1.1 Git
- Directory `MoE_CPAL`, `git init`, main branch `main`. Ask Dawar for the private GitHub repository URL
  and add it as `origin`. If he has not given one yet, keep working locally and ask at the end.
- Keep the specification files from the zip (CLAUDE.md, PROJECT.md, EXPERIMENTS.md, PHASE0_SETUP.md,
  PHASE1_KILL_TEST.md, PREREG_KILL_TEST.md, KICKOFF.md) in the repository root, unchanged.
- Commit messages: short, imperative, one topic each.

### 1.2 Layout
```
MoE_CPAL/
  CLAUDE.md PROJECT.md EXPERIMENTS.md PHASE0_SETUP.md PHASE1_KILL_TEST.md PREREG_KILL_TEST.md
  README.md pyproject.toml requirements.txt requirements-vllm.txt requirements-cpu.txt .gitignore
  src/prouting/
    __init__.py
    adapters/      base.py olmoe.py qwen3_moe.py qwen2_moe.py deepseek_v2.py registry.py (gpt_oss.py stub)
    routing/       block.py policies.py linalg.py baselines.py patch.py
    calibrate/     stats.py clusters.py sketch.py data.py
    analysis/      recorder.py layer_eval.py law.py theory.py
    eval/          gsm8k.py math500.py ppl.py lm_eval_bridge.py stats.py prompts.py
    vllm_plugin/   routing_fn.py register.py
    utils/         gpu.py logging.py seed.py io.py status.py tiny.py
  configs/
    models/        olmoe_1b_7b.yaml olmoe_1b_7b_instruct.yaml qwen3_30b_a3b.yaml deepseek_v2_lite.yaml qwen15_moe.yaml
    experiments/   kill_test.yaml
    prereg_kill_test.yaml
  scripts/
    vm_setup.sh download_models.py prepare_data.py sanity_real_model.py calibrate.py
    record_analysis.py run_layer_eval.py tune_tau.py run_downstream.py make_kill_report.py
    run_kill_test.sh check_vllm_equivalence.py
  tests/
  docs/            related_work_notes.md decisions.md
  results/         STATUS.md (results files are small JSON, CSV and MD; commit them)
  plots/
  artifacts/       (gitignored: calibration statistics, analysis shards, token caches)
  logs/            (gitignored)
```

### 1.3 Laptop environment (small)
- First check: `df -h .` and `python -c "import torch, sys; print(torch.__version__, sys.version)"`.
- Create the venv on top of the system PyTorch: `python -m venv --system-site-packages .venv`.
- Install only what is missing, without a pip cache:
  `PIP_NO_CACHE_DIR=1 pip install "transformers==<pinned>" safetensors pyyaml pytest numpy`.
- If free space drops below about 150 MB, stop installing, write code and tests anyway, and mark in
  STATUS.md that tests run first on the VM.
- Never download models, datasets or tokenizers on the laptop. Tests set `HF_HUB_OFFLINE=1` and build
  tiny random models from config classes.

### 1.4 Version pinning (important)
- MoE code in `transformers` has changed across releases: some versions store experts as a
  `ModuleList` of MLPs, newer ones may store fused 3D weight tensors with a separate router module.
  Pick one recent stable `transformers` version that natively supports OLMoE, Qwen3-MoE, Qwen2-MoE and
  DeepSeek-V2, pin it exactly in `requirements.txt` and `requirements-cpu.txt`, and record why in
  `docs/decisions.md`.
- Read the installed source of each MoE block (for example `transformers/models/olmoe/modeling_olmoe.py`)
  and write the adapters against that version. Detect the storage layout at runtime and raise a clear
  error for an unknown layout.
- vLLM pins its own `transformers` range, so it lives in a second venv on the VM
  (`requirements-vllm.txt`, `.venv-vllm`). The HF venv is the reference.

### 1.5 Files to write now
- `pyproject.toml` (package `prouting`, src layout, Python >= 3.10).
- `requirements.txt` (VM, HF path): torch (CUDA build on the VM), transformers (pinned), accelerate,
  safetensors, huggingface_hub, datasets, lm-eval, math-verify (verify name), numpy, scipy, pandas,
  matplotlib, pyyaml, pytest, tqdm, scikit-learn (k-means) or a small own k-means.
- `requirements-vllm.txt`: vllm (pinned version), plus the package itself installed with `pip install -e .`.
- `requirements-cpu.txt`: the minimal laptop set.
- `.gitignore`: `.venv*`, `artifacts/`, `logs/`, `*.pt`, `*.safetensors`, `*.npz`, `__pycache__/`,
  `.env`, anything containing tokens.
- `README.md`: purpose (two lines), layout, laptop test command, the VM commands from
  PHASE1_KILL_TEST.md Section 1, and where results appear.
- `results/STATUS.md`, `docs/decisions.md` (implementation decisions with dates).

---

## 2. Code to write

General rules: PyTorch only; vectorised over tokens (no Python loops over tokens; loops over experts
and layers are fine); float32 or float64 for every Gram, solve and error computation even when the
model runs in bf16; deterministic given a seed; type hints and short docstrings.

### 2.1 Adapters (`adapters/`)
One adapter per model family, all implementing `MoEAdapter` in `base.py`:
```python
class MoEAdapter(Protocol):
    def find_moe_blocks(self, model) -> list[tuple[int, str, nn.Module]]   # (layer index, name, block)
    def num_experts(self, block) -> int
    def top_k(self, block) -> int
    def intermediate_size(self, block) -> int
    def router_logits(self, block, x: Tensor) -> Tensor                    # [n, N]
    def native_topk(self, block, logits: Tensor) -> tuple[Tensor, Tensor]  # ids [n, k] (long), gates [n, k]
    def expert_forward(self, block, e: int, x: Tensor) -> Tensor           # [n_e, d], before weighting
    def expert_hidden(self, block, e: int, x: Tensor) -> Tensor            # [n_e, I], SwiGLU product (sketches)
    def expert_weights(self, block, e: int) -> tuple[Tensor, Tensor, Tensor]  # W_gate, W_up, W_down
    def shared_forward(self, block, x: Tensor) -> Tensor | None            # shared experts incl. their gate
    def wrap_output(self, block, y: Tensor, logits: Tensor) -> Any         # same return format as the original block
```
- `native_topk` must reproduce the model's own routing exactly: softmax precision, top-k tie
  handling, renormalisation flag (`norm_topk_prob`), any scaling factor, group-limited routing
  (DeepSeek: check `topk_method`, `n_group`, `topk_group`, `routed_scaling_factor` in the config),
  softmax-after-top-k (gpt-oss). Read the installed modelling code; do not assume.
- `wrap_output` returns exactly what the original block returns (several blocks return a tuple of the
  hidden states and router logits).
- `registry.py`: pick the adapter from `model.config.model_type`.

### 2.2 Reduced MoE block (`routing/block.py`, `routing/patch.py`)
`ReducedMoEBlock(nn.Module)` wraps an original block:
1. Flatten tokens to [n, d]; keep an attention-mask-derived `valid` vector if the caller provides one.
2. `logits = adapter.router_logits(block, x)`; `ids, g = adapter.native_topk(block, logits)`.
3. `sel_ids, w = policy(ctx)` where `ctx` carries x, logits, ids, g, layer index, the layer's
   calibration state and a per-call cache. `sel_ids` is [n, M] (long), `w` is [n, M]; a zero weight
   means "not computed". M may exceed k for policies that pick outside T.
4. Group tokens by expert id over entries with non-zero weight; call `adapter.expert_forward` once per
   expert on its tokens; accumulate `w * E` in float32; add `shared_forward`; cast back; reshape.
5. Policies that need the kept experts' outputs before setting weights (G-cos) get a second hook:
   `policy.reweight(ctx, kept_outputs_norms) -> w`.
6. Optional recorder hook (Section 2.6).
7. Counters: number of computed (token, expert) pairs per layer, for the realised budget.
`patch.py`: `patch_model(model, adapter, policy_factory, layers="all" | list) -> handle` replaces
blocks in place and `handle.restore()` puts the originals back. With the `Native` policy the patched
model must reproduce the original (test T1).

### 2.3 Small batched linear algebra (`routing/linalg.py`)
All functions take batched Grams [n, k, k], gates [n, k] and boolean masks [n, k]; work in float64 on
CPU and float32 on GPU; never loop over tokens.
- `ls_weights(G, g, keep_mask, ridge) -> c_full [n, k]`: solves G_SS c = G_ST g on the masked block.
  Implementation: replace dropped rows and columns with the identity, zero the right-hand side there,
  add ridge times the mean kept diagonal, Cholesky solve (with jitter fallback), zero the dropped
  entries.
- `residual_energy(G, g, c_full) -> [n]`: (c - g)^T G (c - g).
- `greedy_ols(G, g, m_max, tau=None, m_min=1, pool_mask=None) -> keep_mask, order, residual_path`:
  forward selection by largest residual drop (PROJECT.md 5.3). Use the incremental (rank-one update or
  Gram-Schmidt in coefficient space) form so a step is O(k^2) per token.
- `greedy_omp(...)`: same interface, OMP criterion (ablation A2).
- `exhaustive_best(G, g, m) -> keep_mask, err`: all subsets of size m (oracle U1 when given the true
  G; ablation A2 when given G^).
- `uniform_coherence_weights(g, keep_mask, r)`: the closed form in PROJECT.md 5.2.
- `closed_form_errors(g, keep_mask, r) -> e1, e2, e3, y2`: the four closed forms in 5.2.
- `effective_experts(G, g, levels=(0.90, 0.95, 0.99)) -> m90, m95, m99` and
  `participation_ratio(G, g) -> PR`.

### 2.4 Calibration statistics (`calibrate/`)
`stats.py`, `GramAccumulator` per MoE layer, with three modes that can run in one pass:
- **in-set (main):** for every valid token, the per-token Gram G(x) on T is added into an N x N sum
  matrix at positions (T_a, T_b) with `index_put_(..., accumulate=True)`; also co-occurrence counts,
  per-expert sums of squared norms and counts, and sums of cosines. Float64 accumulators.
- **cluster-conditioned:** the same, per cluster id (C x N x N), when clusters are given.
- **full Gram (P3 and ablation A4):** on a subset of tokens (default 20k), compute every expert's
  output (all N) and accumulate the full N x N Gram. GPU-heavy; off by default in the kill test.
- `finalise(n_min=32) -> GramState` with G-avg, cosine matrix R^, mean norms, the layer's
  count-weighted mean cosine r_bar and back-off applied; `save/load` to `.pt`.
`clusters.py`: mini-batch k-means with cosine distance on L2-normalised layer inputs; C in {8, 16};
seeded; `fit(x_subset)`, `assign(x)`.
`sketch.py`: neuron importance accumulation (mean |a_jn| * ||W_down[:, n]||), choice of R_j with
r = ceil(fraction * I), `sketch_forward(adapter, block, e, x)`, kappa_j fitting accumulators.
Two-pass calibration driver (`scripts/calibrate.py`): pass 1 on the first 100k calibration tokens
(collect layer inputs for k-means, neuron importance); pass 2 on all calibration tokens (in-set and
cluster-conditioned Grams, sketch kappa and sketch Grams). Output: `artifacts/<model>/calib_seed<s>.pt`.
`data.py`: builds the calibration, validation and analysis token sets (Section 2.7).

### 2.5 Policies (`routing/policies.py`, `routing/baselines.py`)
Interface: `policy(ctx) -> (sel_ids, w)`, optional `policy.reweight(ctx, norms) -> w`.
Our policies:
- `Native()`.
- `TopMKeep(m)` (B1), `TopMRenorm(m)` (B2).
- `Projection(select="gate"|"ols"|"omp"|"exhaustive", weights="keep"|"renorm"|"ls"|"uniform",
  gram="avg"|"cos"|"clu"|"sk"|"oracle", m=None, tau=None, ridge=1e-3, pool="k"|"2k")`.
  Ours-fixed is `Projection(select="ols", weights="ls", m=m)`; ours-adaptive is the same with `tau`.
- `AddProjection(base_policy, gram=...)`: runs any base policy, keeps its selection, replaces its
  weights with least-squares weights (P3).
- `SketchAdd(m, fraction)`: the honest sketch baseline (PROJECT.md 5.6).
Baselines from papers (`baselines.py`): K2Fix (B3), Elbow (B4), EAT (B5), OMPMoE (B6), SERE (B7),
CacheSub (B8), GateRatioSkip (B9). Implement each from the paper as read in Section 3. If a paper is
unclear, implement the clearest reading, write the open question in `docs/related_work_notes.md`,
and mark the class `FIDELITY_UNCHECKED = True`. Never guess silently.
The vLLM routing function (2.9) must call the same pure-torch selection and weighting code.

### 2.6 Analysis recorder and offline evaluator (`analysis/`)
`recorder.py`: during a native forward on the analysis tokens, for every valid token and MoE layer,
store: T ids [k], gates g [k], the true Gram G(x) [k, k] (float32), the top-32 router probabilities
and ids, ||y||^2, ||h||^2 (residual stream entering the block), the cluster id for C = 8 and 16, and
(if sketches are calibrated) the cross Gram <E_i, kappa_j E^_j> [k, k] and the sketch Gram [k, k].
Write per-layer shards (`artifacts/<model>/analysis/layer_XX.pt`). Expert outputs are kept only
transiently ([n, k, d] per layer per batch).
`layer_eval.py`: loads shards and a `GramState`, runs every policy offline (selection and weights
from that policy's G^, error from the true G(x) through the quadratic form of PROJECT.md 5.1) and
writes per-layer and per-budget tables: relative error energy, median token relative error, error
relative to the residual stream, realised mean active experts. Includes U1 and U2.
`law.py`: m90, m95, m99, PR, mean r, mu, top-gate cosine, per layer and per domain.
`theory.py`: closed-form predictions, lambda_min, exact-recovery fraction, greedy-to-oracle gap.

### 2.7 Data preparation (`calibrate/data.py`, `scripts/prepare_data.py`)
- Sources (VM only, IDs to check): FineWeb-Edu `HuggingFaceFW/fineweb-edu`, config `sample-10BT`,
  streamed (verify); MetaMathQA `meta-math/MetaMathQA` (verify); WikiText-2 `Salesforce/wikitext`,
  `wikitext-2-raw-v1` (verify); GSM8K `openai/gsm8k`, `main` (verify).
- Calibration mix v1 (D8): 60% FineWeb-Edu text, 40% MetaMathQA formatted with the model's chat
  template (question as user, answer as assistant).
- Sequence length 1024. Text: 1024-token chunks from single documents. Maths: one example per
  sequence, right-padded, padding excluded through the attention mask and the `valid` vector.
- Split by document into calibration (1M tokens), validation (20k), analysis (50k OLMoE, 30k Qwen3),
  seed 0. Save token ids and document hashes to `artifacts/data/`; assert pairwise disjointness and
  that no GSM8K or MATH test question appears (exact-match check on normalised question text).
- `--tiny` mode: synthetic random token sequences, no downloads.

### 2.8 Evaluation (`eval/`)
- `gsm8k.py`: chat template, zero-shot, user message = question + "\nPlease reason step by step, and
  put your final answer after '####'."; greedy; max_new_tokens 512; left padding; batched; for Qwen3
  pass `enable_thinking=False` to the chat template (verify the argument name). Extraction: number
  after the last '####'; otherwise the last number in the text; strip commas, currency signs and a
  trailing full stop. Exact match against the gold number. Save every generation to JSONL.
  Fixed subsets: `gsm8k_fixed250` and `gsm8k_fixed500` are seeded random subsets of the test split;
  write their indices to `configs/` once (seed 0) and commit them.
- `math500.py`: boxed answer extraction and equivalence via math-verify (verify), max_new_tokens 2048.
- `ppl.py`: WikiText-2 test perplexity, context 2048, stride 2048, padding excluded.
- `lm_eval_bridge.py`: run lm-evaluation-harness tasks on an already-loaded, patched HF model
  (wrap with the harness's HF model class; check the current API, verify).
- `stats.py`: paired bootstrap (10,000 resamples, seeded) for accuracy differences; recovery ratio
  with its interval; plain accuracy intervals.

### 2.9 vLLM plugin (`vllm_plugin/`, VM only)
- `routing_fn.py`: a pure-torch function with the signature vLLM's fused MoE layer expects for a
  custom routing function (check the installed vLLM; believed to be
  `(hidden_states, gating_output, topk, renormalize) -> (topk_weights, topk_ids)`, verify). It closes
  over the layer index and the layer's G^ tensors on the GPU and calls the same selection and
  weighting code as the HF path. For adaptive budgets, return k slots with zero weights for unused
  slots (correct outputs, no speed-up); for fixed m, return m slots if the kernel accepts it (verify).
- `register.py`: attach the function to every fused MoE layer of a loaded vLLM model. vLLM may run
  the model in a separate process; use its plugin mechanism or an in-process option (check the
  installed version, verify), and run with `enforce_eager=True` first.
- `scripts/check_vllm_equivalence.py` implements check V1 of EXPERIMENTS.md.
- Laptop: only the pure-torch routing function is tested (T15); vLLM itself is never imported there.

### 2.10 Utilities (`utils/`)
- `gpu.py`: `cap_memory(fraction=None, gb=None)` using `torch.cuda.set_per_process_memory_fraction`;
  `free_memory_gb()` from `nvidia-smi` or `torch.cuda.mem_get_info`; prints what else is running.
- `logging.py`: `run_record(config, seed)` returns a dict with git hash (and a dirty flag), package
  versions, GPU name, start and end time; every results file includes it.
- `seed.py`, `io.py` (atomic writes, resumable JSONL), `status.py` (append to results/STATUS.md).
- `tiny.py`: builders for tiny random models (Section 4).

### 2.11 Scripts and configs
Every script: `--config`, `--seed`, `--tiny`, `--force` (otherwise skip finished outputs, so runs
resume after interruption), logs to `logs/`, writes results with the run record.
`run_kill_test.sh [--tiny] [--model olmoe|qwen3|both]` chains: prepare_data, sanity_real_model (K0),
calibrate, record_analysis, run_layer_eval (K1, K2), tune_tau, run_downstream (K3), make_kill_report.
`configs/models/*.yaml`: Hugging Face ID, revision (fill on the VM with the commit hash actually
downloaded), dtype, adapter, k, N, gate convention, layers to patch. Fill values from config.json on
the VM; leave placeholders marked (verify) until then.

---

## 3. Literature notes and the novelty gate (laptop, web allowed)
Read each paper (abstract, method and experiments) and write `docs/related_work_notes.md`, one block
per paper: what it changes; training-free or not; how it chooses experts; how it weights the
remaining experts; models and headline numbers (copied exactly, with table or page); open questions
for reimplementation.
Papers: OMP-MoE (2609.31631), k2 fix (2609.04575), SERE (2602.07616), Elbow (2608.04401),
EAT (2609.33614), score-based cache substitution (2508.18983), CCE (2412.00099), MoE Lens (2603.05806),
2601.03577, LightMoE (2603.12645), BuddyMoE (find the ID), MoE-Spec (2602.16052), gate-ratio skipping
(2402.14800, verify), dynamic top-p routing (2403.07652, verify). Then search arXiv and Google
Scholar for: "mixture of experts" with "least squares" or "projection" or "Gram" or "expert output
correlation" or "matching pursuit" or "reduce top-k at inference"; add anything close.

**Novelty gate.** If any paper (a) estimates inner products or correlations between expert outputs
and (b) uses them to solve for the weights of a reduced expert set, stop and report to Dawar before
continuing, with the exact overlap. Also report if any paper measures the effective number of experts
per token through the span of expert outputs (claim C1).

Also write, in the same file, where the k2 fix's rule sits relative to the projection family of
PROJECT.md 5.2, with the derivation or a numerical check.

---

## 4. Tests (`tests/`, all CPU, no downloads)
Tiny models from config classes, random weights, seeded. Starting points (adjust names to the pinned
transformers version): `OlmoeConfig` and `Qwen3MoeConfig` with hidden_size 32, intermediate or
moe_intermediate size 64, 2 layers, 4 heads, 8 experts, k = 4 (also one case with k = 2),
vocab 128, max positions 128; Qwen3 with norm_topk_prob True and every layer sparse; `Qwen2MoeConfig`
with a shared expert; `DeepseekV2Config` with one shared expert and the first layer dense, if the
pinned version has it (otherwise skip with a message). Float32 for model tests, float64 for algebra.

| ID | Test | Pass condition |
|---|---|---|
| T1 | Patched model with `Native` equals the unpatched model | max abs logit difference <= 1e-6; identical return types |
| T2 | `native_topk` equals the model's own routing | same ids and gates, both gate conventions |
| T3 | Gram identities | g^T G g equals ||y||^2 computed directly (relative 1e-6); quadratic-form error equals direct ||y^ - y||^2 for random (S, c) |
| T4 | Least-squares weights with the true Gram | residual orthogonal to every kept output (relative 1e-6); equals `torch.linalg.lstsq` on the explicit vectors |
| T5 | Special cases and closed forms | G = I gives c = g_S; G = all-ones gives an exact solution; uniform-coherence weights and the closed-form errors e1, e2, e3, ||y||^2 match the numeric solver (1e-8) for 1,000 random (r, m, g) |
| T6 | Greedy OLS | residual non-increasing; equals a brute-force forward selection; exhaustive best never worse than greedy |
| T7 | Adaptive stop | tau = 0 keeps everything needed to reach zero residual; tau = 1 keeps m_min; mean kept count non-increasing in tau |
| T8 | Policies through the block | TopMKeep with m = k and Projection with m = k equal native; with drops, block output equals the hand-computed sum_S c_i E_i |
| T9 | GramAccumulator | recovers a known Gram from synthetic outputs; counts correct; back-off below n_min; save and load round trip |
| T10 | Offline equals online | layer error from recorded Grams equals the error measured by running the patched block against native (relative 1e-5), for B2 and ours |
| T11 | Sketch | with fraction 1.0 the sketch equals the expert output and kappa = 1 |
| T12 | Clustering | deterministic with a seed; assignment of the centroids returns their own ids |
| T13 | Evaluation helpers | GSM8K extraction on 15 hand-written answers (including commas, '####', trailing text); bootstrap on a known case; recovery ratio |
| T14 | Prereg consistency | the YAML block inside PREREG_KILL_TEST.md equals `configs/prereg_kill_test.yaml` |
| T15 | vLLM routing function (pure torch) | same ids and weights as the HF policy on random logits and Grams |
| T16 | End-to-end tiny | `run_kill_test.sh --tiny` writes every output file and a TINY report with the decision logic exercised |

## 5. Tiny end-to-end run
`bash scripts/run_kill_test.sh --tiny` on the tiny OLMoE-style model with synthetic tokens:
data split, calibration, recording, offline evaluation, tau tuning, a fake downstream step (the
tiny model's next-token accuracy on synthetic data stands in for GSM8K), and the report with the
decision table filled. The report header must say: TINY RUN, NUMBERS ARE MEANINGLESS.

## 6. VM bootstrap scripts (write now, run on the VM)
- `scripts/vm_setup.sh`: create `.venv` (HF) and `.venv-vllm`; install; print `nvidia-smi`, free disk
  under `$HF_HOME`, Python and package versions; run `pytest -q`.
- `scripts/download_models.py --model <name>`: `huggingface_hub.snapshot_download` with
  `allow_patterns` limited to `*.json`, `*.safetensors`, tokenizer files and `*.txt`/`*.model` if
  needed; into `$HF_HOME`; records the revision hash in `configs/models/<name>.yaml`.
- Never put a token in a script; the token comes from `hf auth login` on the VM (verify the command
  for the installed huggingface_hub).

## 7. Finish P0
1. Commit `PREREG_KILL_TEST.md` and `configs/prereg_kill_test.yaml` in their own commit; write the hash
   into `results/STATUS.md`.
2. Push. Update STATUS.md: "P0 done, waiting for VM".
3. Report to Dawar: what is ready, test results, the novelty-gate outcome in three lines, anything
   uncertain (with (verify) marks), and the exact first commands for the VM.
