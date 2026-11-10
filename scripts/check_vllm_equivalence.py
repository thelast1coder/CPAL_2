"""Check V1 (EXPERIMENTS.md P2): the vLLM path matches HF before plan A is used. VM only, .venv-vllm.

For a model: with the native policy, greedy outputs on 50 GSM8K prompts match HF in >= 48; the same holds
for B2 at m = 2 and ours at m = 2; WikiText-2 perplexity agrees within 0.5%. HF outputs come from
results/<...>/hf_reference_<policy>.jsonl written by the HF venv (--write-hf-reference, run in .venv),
because vLLM and the HF reference live in different venvs.

The vLLM attachment uses vllm_plugin.register (attribute names and in-process execution: verify on the VM).
Usage:
  .venv/bin/python scripts/check_vllm_equivalence.py --model qwen3_30b_a3b --write-hf-reference
  .venv-vllm/bin/python scripts/check_vllm_equivalence.py --model qwen3_30b_a3b
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

POLICIES = ["native", "B2_m2", "ours_m2"]


def prompts(tok, fam, n=50):
    from datasets import load_dataset
    from prouting.eval.prompts import chat_prompt, gsm8k_user_message
    ds = load_dataset("openai/gsm8k", "main", split="test")  # (verify)
    return [chat_prompt(tok, gsm8k_user_message(ds[i]["question"]), fam) for i in range(n)]


def write_hf_reference(args, out):
    import torch
    from prouting.adapters import get_adapter
    from prouting.calibrate.driver import calib_from_dict
    from prouting.eval.gsm8k import generate_batch
    from prouting.routing.patch import patch_model
    from prouting.routing.policies import Native, Projection, TopMRenorm
    from prouting.utils.cli import load_model, model_cfg
    from prouting.utils.io import load_torch
    a = argparse.Namespace(tiny=False, device="cuda", seed=0)
    model, tok = load_model(args.model, a)
    fam = model_cfg(args.model)["family"]
    calib = calib_from_dict(load_torch(ROOT / "artifacts" / args.model / "calib_seed0.pt"))
    gram = args.gram
    facs = {"native": lambda li: Native(), "B2_m2": lambda li: TopMRenorm(2),
            "ours_m2": lambda li: Projection("ols", "ls", gram, m=2)}
    ps = prompts(tok, fam)
    for name, f in facs.items():
        h = patch_model(model, get_adapter(model), f, calib=calib)
        outs = generate_batch(model, tok, ps, 512)
        h.restore()
        (out / f"hf_reference_{name}.jsonl").write_text("\n".join(json.dumps({"i": i, "text": t}) for i, t in enumerate(outs)))


def run_vllm(args, out):
    import torch
    from vllm import LLM, SamplingParams
    from prouting.utils.cli import model_cfg
    from prouting.utils.io import load_torch
    from prouting.vllm_plugin.register import attach
    from prouting.vllm_plugin.routing_fn import make_routing_fn
    mc = model_cfg(args.model)
    llm = LLM(model=mc["hf_id"], revision=mc.get("revision"), enforce_eager=True, dtype=mc["dtype"])
    tok = llm.get_tokenizer()
    ps = prompts(tok, mc["family"])
    calib = load_torch(ROOT / "artifacts" / args.model / "calib_seed0.pt")
    layers = sorted(calib)
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    report = {}
    for name in POLICIES:
        if name != "native":
            m = 2
            sel, w = ("gate", "renorm") if name == "B2_m2" else ("ols", "ls")
            attach(llm, lambda i: make_routing_fn(calib[layers[i]]["gram"]["G_avg"].cuda(), select=sel, weights=w, m=m))
        outs = [o.outputs[0].text for o in llm.generate(ps, sp)]
        ref = [json.loads(l)["text"] for l in (out / f"hf_reference_{name}.jsonl").read_text().splitlines()]
        same = sum(a == b for a, b in zip(outs, ref))
        report[name] = dict(identical=same, of=len(ref), passed=same >= 48)
    report["note"] = "perplexity agreement (0.5%) to be added once prompt_logprobs support is confirmed (verify)"
    (out / "vllm_equivalence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gram", default="avg", help="primary Gram variant for ours (G-cos needs the HF path)")
    ap.add_argument("--write-hf-reference", action="store_true")
    args = ap.parse_args()
    out = ROOT / "results" / "vllm" / args.model
    out.mkdir(parents=True, exist_ok=True)
    write_hf_reference(args, out) if args.write_hf_reference else run_vllm(args, out)


if __name__ == "__main__":
    main()
