"""Download one model into $HF_HOME (VM only) and record the revision in configs/models/<name>.yaml.

Only weights, configs and tokenizer files: allow_patterns = *.json, *.safetensors, tokenizer files, *.txt,
*.model, *.tiktoken. The token comes from `hf auth login` on the VM (verify the command for the installed
huggingface_hub); it is never written in a script.
After download, prints the routing-relevant values from config.json so the (verify) fields in the model YAML
can be filled; with --fill they are written automatically.

Usage: python scripts/download_models.py --model olmoe_1b_7b [--fp8] [--fill]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {  # yaml key -> config.json keys tried in order
    "num_experts": ["num_experts", "num_local_experts", "n_routed_experts"],
    "top_k": ["num_experts_per_tok"],
    "norm_topk_prob": ["norm_topk_prob"],
    "intermediate_size": ["moe_intermediate_size", "intermediate_size"],
    "num_layers": ["num_hidden_layers"],
    "routed_scaling_factor": ["routed_scaling_factor"],
    "topk_method": ["topk_method"],
    "n_group": ["n_group"],
    "topk_group": ["topk_group"],
}


def set_yaml_field(text: str, key: str, value) -> str:
    v = json.dumps(value) if not isinstance(value, bool) else str(value).lower()
    pat = re.compile(rf"^({re.escape(key)}:\s*)([^#\n]*?)(\s*#.*)?$", re.M)
    if pat.search(text):
        return pat.sub(lambda m: f"{m.group(1)}{v}" + ("   # from config.json" if m.group(3) else ""), text, count=1)
    return text + f"{key}: {v}   # from config.json\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp8", action="store_true", help="download hf_id_fp8 instead (decision D7)")
    ap.add_argument("--fill", action="store_true", help="write config.json values into the model YAML")
    args = ap.parse_args()
    if not os.environ.get("HF_HOME"):
        sys.exit("set HF_HOME to the big disk first (PHASE1_KILL_TEST.md Section 1)")
    import yaml
    from huggingface_hub import HfApi, snapshot_download
    ypath = ROOT / "configs" / "models" / f"{args.model}.yaml"
    cfg = yaml.safe_load(ypath.read_text(encoding="utf-8"))
    repo = cfg["hf_id_fp8"] if args.fp8 else cfg["hf_id"]
    rev = HfApi().model_info(repo).sha
    path = snapshot_download(repo, revision=rev, allow_patterns=[
        "*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model", "*.tiktoken", "*.jinja"])
    conf = json.loads((Path(path) / "config.json").read_text())
    found = {}
    for k, keys in FIELDS.items():
        for c in keys:
            if c in conf:
                found[k] = conf[c]
                break
    found["first_k_dense_replace"] = conf.get("first_k_dense_replace")
    found["n_shared_experts"] = conf.get("n_shared_experts")
    found["mlp_only_layers"] = conf.get("mlp_only_layers")
    found["decoder_sparse_step"] = conf.get("decoder_sparse_step")
    print(json.dumps(dict(repo=repo, revision=rev, path=path, config=found), indent=2))
    text = ypath.read_text(encoding="utf-8")
    text = set_yaml_field(text, "revision", rev)
    if args.fp8:
        text = set_yaml_field(text, "use_fp8", True)
    if args.fill:
        for k in FIELDS:
            if k in found:
                text = set_yaml_field(text, k, found[k])
    ypath.write_text(text, encoding="utf-8")
    print(f"recorded revision {rev} in {ypath}")


if __name__ == "__main__":
    main()
