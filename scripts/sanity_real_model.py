"""K0 sanity checks (PHASE1_KILL_TEST.md Section 3). All must pass before anything else continues.

K0.1 native wrapper exact (stats model in float32): 4 x 256 tokens, max abs logit difference <= 1e-4.
K0.2 native wrapper practical (task model): identical greedy outputs on >= 19 of 20 GSM8K prompts and
     WikiText-2 perplexity within 0.2% (50k tokens).
K0.3 Gram identity: 1,000 tokens per layer, |g^T G g - ||y||^2| / ||y||^2 <= 1e-3.
K0.4 offline equals online (middle MoE layer): B2 at m = 2, offline error energy within 1% of online.
K0.5 Projection with m = k reproduces native: relative layer error <= 1e-4 (fp32) or 1e-3 (bf16).
K0.6 (native GSM8K accuracy) is written by run_downstream.py (B0 runs first) and copied into STATUS.md.
Needs: prepared data only (a throwaway G-avg is built from the K0 tokens). Tiny mode runs the same checks.
Exit code 1 if any check fails.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from prouting.adapters import get_adapter
from prouting.analysis.layer_eval import KILL_METHODS, eval_method
from prouting.analysis.recorder import AnalysisRecorder, load_shard
from prouting.calibrate.data import SplitTokens
from prouting.calibrate.driver import calib_from_dict
from prouting.routing import linalg as L
from prouting.routing.patch import patch_model
from prouting.routing.policies import Native, Projection, TopMRenorm
from prouting.utils.cli import base_parser, concrete_model, load_model, paths, setup
from prouting.utils.io import load_torch, write_json
from prouting.utils.logging import run_record


@torch.no_grad()
def k01(model, ad, toks, tol):
    ref = model(toks).logits.float()
    h = patch_model(model, ad)
    out = model(toks).logits.float()
    h.restore()
    d = float((out - ref).abs().max())
    return dict(max_abs_logit_diff=d, tol=tol, **{"pass": d <= tol})


def quick_calib(model, ad, toks, mask):
    """Throwaway G-avg from the K0 tokens (K0 runs before calibration; only K0.5 needs a Gram object)."""
    from prouting.calibrate.driver import calibrate
    return calibrate(model, ad, [(toks, mask)], clusters=(), pass1_tokens=0, sketch_fraction=None, n_min=1,
                     log=lambda *a: None)


@torch.no_grad()
def gram_checks(model, ad, calib, toks, mask, is_fp32):
    """K0.3, K0.4, K0.5 on the given tokens."""
    dev = next(model.parameters()).device
    h = patch_model(model, ad, calib=calib)
    tmp = Path(tempfile.mkdtemp())
    rec = AnalysisRecorder(h, model, record_sketch=False, clusters=())
    h.set_valid(mask)
    model(input_ids=toks, attention_mask=mask)
    rec.flush(tmp)
    rec.remove()
    worst = 0.0
    for li in h.layers:
        sh = load_shard(tmp, li)
        n = min(1000, sh["ids"].shape[0])
        G, g, y2 = sh["G"][:n].double(), sh["g"][:n].double(), sh["y2"][:n].double()
        rel = ((L.output_energy(G, g) - y2).abs() / y2.clamp_min(1e-30)).max()
        worst = max(worst, float(rel))
    k03 = dict(max_rel_diff=worst, tol=1e-3, **{"pass": worst <= 1e-3})
    mid = h.layers[len(h.layers) // 2]
    xs = {}
    hk = h.wrapper(mid).register_forward_pre_hook(lambda mod, a: xs.__setitem__("x", a[0]))
    h.set_valid(mask)
    model(input_ids=toks, attention_mask=mask)
    hk.remove()
    x = xs["x"].reshape(-1, xs["x"].shape[-1])[mask.reshape(-1).to(dev)]
    w = h.wrapper(mid)
    h.set_valid(None)
    w.policy = Native()
    y_nat = w(x.unsqueeze(0))[0].double()
    w.policy = TopMRenorm(2)
    y_b2 = w(x.unsqueeze(0))[0].double()
    online = float(((y_b2 - y_nat) ** 2).sum() / (y_nat**2).sum())
    offline = eval_method(load_shard(tmp, mid), calib[mid], KILL_METHODS["B2"], m=2)["rel_error_energy"]
    k04 = dict(layer=mid, online=online, offline=offline, rel_gap=abs(offline - online) / max(online, 1e-30))
    k04["pass"] = k04["rel_gap"] <= 0.01
    k = ad.top_k(w.orig)
    w.policy = Projection("ols", "ls", "avg", m=k)
    y_full = w(x.unsqueeze(0))[0].double()
    e = float(((y_full - y_nat) ** 2).sum() / (y_nat**2).sum())
    tol5 = 1e-4 if is_fp32 else 1e-3
    k05 = dict(layer=mid, rel_error=e, tol=tol5, **{"pass": e <= tol5})
    h.restore()
    return k03, k04, k05


def main():
    p = base_parser(__doc__)
    p.add_argument("--skip-k02", action="store_true")
    args = p.parse_args()
    cfg, log = setup(args, "sanity_real_model")
    P = paths(args, cfg)
    rec = run_record({}, args.seed)
    st = SplitTokens.from_dict(load_torch(P["data"] / "validation.pt"))
    checks = {}
    # K0.1: stats model in float32
    model, _ = load_model(concrete_model(args, cfg), args, dtype="float32")
    ad = get_adapter(model)
    dev = next(model.parameters()).device
    L01 = 256 if not args.tiny else st.input_ids.shape[1]
    toks = st.input_ids[:4, :L01].to(dev)
    checks["K0.1"] = k01(model, ad, toks, 1e-4)
    mask = st.attention_mask[:4, :L01].to(dev)
    calib = quick_calib(model, ad, toks, mask)
    checks["K0.3"], checks["K0.4"], checks["K0.5"] = gram_checks(model, ad, calib, toks, mask, True)
    del model
    # K0.2 and the bf16 versions of K0.3 to K0.5: task model in its working precision
    if not args.skip_k02:
        model, tok = load_model(concrete_model(args, cfg, "task"), args)
        ad = get_adapter(model)
        is32 = next(model.parameters()).dtype == torch.float32
        if args.tiny:
            x = st.input_ids[:20].to(dev)
            with torch.no_grad():
                ref = model(x).logits[:, -1].argmax(-1)
                h = patch_model(model, ad)
                out = model(x).logits[:, -1].argmax(-1)
                h.restore()
            same = int((ref == out).sum())
            checks["K0.2"] = dict(identical=same, of=20, **{"pass": same >= 19})
        else:
            checks["K0.2"] = k02_real(model, tok, ad, cfg, args, log)
        a, b, c = gram_checks(model, ad, quick_calib(model, ad, toks, mask), toks, mask, is32)
        checks["K0.3_task"], checks["K0.4_task"], checks["K0.5_task"] = a, b, c
    ok = all(v["pass"] for v in checks.values())
    for k_, v in checks.items():
        log.info(f"[K0] {k_}: {'pass' if v['pass'] else 'FAIL'} {v}")
    write_json(P["res"] / "k0.json", dict(checks=checks, all_pass=ok, run=rec.finish()))
    if not ok:
        log.error("K0 FAILED: fix and rerun; do not continue (PHASE1_KILL_TEST.md Section 3)")
        sys.exit(1)


@torch.no_grad()
def k02_real(model, tok, ad, cfg, args, log):
    from datasets import load_dataset
    from prouting.eval.gsm8k import generate_batch
    from prouting.eval.ppl import perplexity
    from prouting.eval.prompts import chat_prompt, gsm8k_user_message
    from prouting.utils.cli import model_cfg
    src = cfg["data"]["sources"]["gsm8k"]
    ds = load_dataset(src["path"], src["name"], split=src["split"])
    fam = model_cfg(concrete_model(args, cfg, "task"))["family"]
    prompts = [chat_prompt(tok, gsm8k_user_message(ds[i]["question"]), fam) for i in range(20)]
    ref = generate_batch(model, tok, prompts, 512)
    h = patch_model(model, ad)
    out = generate_batch(model, tok, prompts, 512)
    h.restore()
    same = sum(a == b for a, b in zip(ref, out))
    w = cfg["data"]["sources"]["wikitext2"]
    text = "\n\n".join(load_dataset(w["path"], w["name"], split=w["split"])["text"])
    ids = tok(text, return_tensors="pt")["input_ids"][0][:50001]
    p0 = perplexity(model, ids)["ppl"]
    h = patch_model(model, ad)
    p1 = perplexity(model, ids, handle=h)["ppl"]
    h.restore()
    rel = abs(p1 - p0) / p0
    return dict(identical=same, of=20, ppl_native=p0, ppl_wrapped=p1, ppl_rel_diff=rel,
                **{"pass": same >= 19 and rel <= 0.002})


if __name__ == "__main__":
    main()
