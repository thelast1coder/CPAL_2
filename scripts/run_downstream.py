"""K3 downstream (PHASE1_KILL_TEST.md Section 9) on the task model (OLMoE-Instruct / Qwen3 non-thinking).

Step 1: B0 and B2 at m in {1,2,3}; m* = smallest m where B2 loses >= 5.0 points (else 1).
Step 2 at m*: B2+P, B3, B3+P, ours-fixed, ours-adaptive. WikiText-2 perplexity (secondary) for B0, B2 and
ours-fixed at m in {1,2,3}. Same prompts, greedy decoding, max_new_tokens 512, same batches for every config.
Tiny mode: the task is agreement with the native model's greedy next token at the last valid position of
250 synthetic sequences (a stand-in for GSM8K; numbers meaningless).
Output results/.../k3/<config>.json (resumable) and results/.../k3.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from prouting.adapters import get_adapter
from prouting.calibrate.driver import calib_from_dict
from prouting.eval.stats import paired_bootstrap, recovery
from prouting.routing.baselines import K2Fix
from prouting.routing.patch import patch_model
from prouting.routing.policies import AddProjection, Native, Projection, TopMRenorm
from prouting.utils.cli import base_parser, concrete_model, load_model, load_yaml, model_cfg, paths, setup
from prouting.utils.io import load_torch, read_json, write_json
from prouting.utils.logging import REPO_ROOT, run_record
from prouting.utils.tiny import structured_tokens

GRAM = {"ours-avg": "avg", "ours-cos": "cos", "ours-clu": "clu"}


class Task:
    """Runs one configuration on the K3 task and returns per-item scores."""

    def __init__(self, args, cfg, model, tok, handle, log):
        self.args, self.cfg, self.model, self.tok, self.h, self.log = args, cfg, model, tok, handle, log
        self.name = "tiny_agreement" if args.tiny else "gsm8k_fixed250"
        if args.tiny:
            self.seqs = structured_tokens(250, 48, 128, seed=777)
            self.native_pred = None
        else:
            from datasets import load_dataset
            src = cfg["data"]["sources"]["gsm8k"]
            ds = load_dataset(src["path"], src["name"], split=src["split"])
            assert len(ds) == 1319, f"GSM8K test size {len(ds)} (expected 1319, verify)"
            idx = read_json(REPO_ROOT / "configs" / "gsm8k_fixed250.json")["indices"]
            self.items = [dict(idx=i, question=ds[i]["question"], answer=ds[i]["answer"]) for i in idx]

    @torch.no_grad()
    def _tiny_preds(self):
        dev = next(self.model.parameters()).device
        x = self.seqs.to(dev)
        self.h.set_valid(torch.ones_like(x, dtype=torch.bool))
        p = self.model(x).logits[:, -1].argmax(-1).cpu()
        self.h.set_valid(None)
        return p

    def run(self, tag: str, out_dir: Path) -> dict:
        self.h.reset_counters()
        if self.args.tiny:
            pred = self._tiny_preds()
            if self.native_pred is None:
                if tag != "B0":
                    raise RuntimeError("run B0 first in tiny mode")
                self.native_pred = pred
            per_item = (pred == self.native_pred).float().tolist()
        elif self.name == "arc_challenge":
            from prouting.eval.lm_eval_bridge import run_tasks
            res = run_tasks(self.model, self.tok, ["arc_challenge"], batch_size=self.cfg["downstream"]["batch_size"])
            per_item = [float(s["acc"]) for s in res["samples"]["arc_challenge"]]  # (verify field names)
        else:
            from prouting.eval.gsm8k import run_gsm8k
            fam = model_cfg(concrete_model(self.args, self.cfg, "task"))["family"]
            r = run_gsm8k(self.model, self.tok, self.items, fam, out_dir / f"{tag}.jsonl",
                          batch_size=self.cfg["downstream"]["batch_size"],
                          max_new_tokens=self.cfg["downstream"]["max_new_tokens"], tag=tag)
            per_item = [float(v) for v in r["per_item"]]
        b = self.h.realised_budget()
        return dict(tag=tag, task=self.name, accuracy=sum(per_item) / len(per_item), n=len(per_item),
                    per_item=per_item, mean_active=b["mean"], per_layer_active=b["per_layer"])


def ppl_tokens(args, cfg, tok):
    if args.tiny:
        return structured_tokens(1, cfg["downstream"]["ppl_tokens"], 128, seed=4242).reshape(-1)
    from datasets import load_dataset
    src = cfg["data"]["sources"]["wikitext2"]
    text = "\n\n".join(load_dataset(src["path"], src["name"], split=src["split"])["text"])
    return tok(text, return_tensors="pt")["input_ids"][0][: cfg["downstream"]["ppl_tokens"] + 1]


def main():
    p = base_parser(__doc__)
    p.add_argument("--skip-ppl", action="store_true")
    args = p.parse_args()
    args.target = "task"
    cfg, log = setup(args, "run_downstream")
    P = paths(args, cfg, "task")
    k3dir = P["res"] / "k3"
    k3dir.mkdir(exist_ok=True)
    pre = load_yaml(REPO_ROOT / "configs" / "prereg_kill_test.yaml")["k3"]
    primary = read_json(P["res"] / "layer_eval.json")["primary"]
    taus = read_json(P["res"] / "tau_task.json")["taus"]
    rec = run_record(dict(downstream=cfg["downstream"], primary=primary, prereg_k3=pre), args.seed)
    model, tok = load_model(concrete_model(args, cfg, "task"), args)
    ad = get_adapter(model)
    calib = calib_from_dict(load_torch(P["calib"]))
    h = patch_model(model, ad, calib=calib)
    task = Task(args, cfg, model, tok, h, log)
    gram = GRAM[primary]
    k2 = cfg["downstream"].get("b3_k2")

    def run(tag, factory):
        f = k3dir / f"{tag}.json"
        if f.exists() and not args.force:
            r = read_json(f)
            if args.tiny and tag == "B0":
                task.native_pred = None
                h.set_policy(factory)
                task.run(tag, k3dir)   # rebuild native predictions for agreement scoring
            return r
        h.set_policy(factory)
        r = task.run(tag, k3dir)
        write_json(f, r)
        log.info(f"[K3] {tag}: acc {r['accuracy']:.4f} (n={r['n']}), mean active {r['mean_active']:.3f}")
        return r

    R = {"B0": run("B0", lambda li: Native())}
    # OLMoE task rule: decided from the native number only
    if not args.tiny and args.model.startswith("olmoe") and R["B0"]["accuracy"] < pre["olmoe_task_switch"]["if_native_accuracy_below"]:
        log.info("[K3] OLMoE native accuracy below 25%: task switches to ARC-Challenge (prereg)")
        task.name = "arc_challenge"
        R["B0_gsm8k"] = R.pop("B0")
        R["B0"] = run("B0_arc", lambda li: Native())
    for m in pre["m_star_candidates"]:
        R[f"B2_m{m}"] = run(f"B2_m{m}", lambda li, m=m: TopMRenorm(m))
    m_star = next((m for m in pre["m_star_candidates"]
                   if 100 * (R["B0"]["accuracy"] - R[f"B2_m{m}"]["accuracy"]) >= pre["m_star_min_b2_loss_points"]), 1)
    log.info(f"[K3] m* = {m_star}")
    ms = m_star
    R["B2+P"] = run(f"B2P_m{ms}", lambda li: AddProjection(TopMRenorm(ms)))
    R["B3"] = run(f"B3_m{ms}", lambda li: K2Fix(ms, k2))
    R["B3+P"] = run(f"B3P_m{ms}", lambda li: AddProjection(K2Fix(ms, k2)))
    R["ours-fixed"] = run(f"ours_fixed_m{ms}", lambda li: Projection("ols", "ls", gram, m=ms))
    tgt = str(float(ms)) if ms >= 2 else "1.5"
    tau = taus[tgt]["tau"]
    R["ours-adaptive"] = run(f"ours_adaptive_t{tgt}", lambda li: Projection("ols", "ls", gram, tau=tau))

    acc = lambda t: R[t]["per_item"]
    B2 = R[f"B2_m{ms}"]
    gain = paired_bootstrap(acc("ours-fixed"), B2["per_item"], **_bs(pre))
    rec_ = recovery(acc("ours-fixed"), B2["per_item"], acc("B0"), **_bs(pre))
    vs_b3 = paired_bootstrap(acc("ours-fixed"), acc("B3"), **_bs(pre))
    secondary = {t: recovery(acc(t), B2["per_item"], acc("B0"), **_bs(pre)) for t in ("B2+P", "B3+P")}
    if ms == 1:
        secondary["ours-adaptive_vs_B2_m2"] = paired_bootstrap(acc("ours-adaptive"), R["B2_m2"]["per_item"], **_bs(pre))
    ppl = {}
    if not args.skip_ppl:
        from prouting.eval.ppl import perplexity
        toks = ppl_tokens(args, cfg, tok)
        ctx = 2048 if not args.tiny else 64
        for tag, fac in [("B0", lambda li: Native())] + [
                (f"{n}_m{m}", f) for m in (1, 2, 3) for n, f in
                (("B2", lambda li, m=m: TopMRenorm(m)), ("ours-fixed", lambda li, m=m: Projection("ols", "ls", gram, m=m)))]:
            f = k3dir / f"ppl_{tag}.json"
            if f.exists() and not args.force:
                ppl[tag] = read_json(f)
                continue
            h.set_policy(fac)
            ppl[tag] = perplexity(model, toks, context=ctx, stride=ctx, handle=h)
            write_json(f, ppl[tag])
    h.restore()
    summary = dict(task=task.name, m_star=m_star, primary=primary,
                   accuracy={t: R[t]["accuracy"] for t in R}, mean_active={t: R[t]["mean_active"] for t in R},
                   n=R["B0"]["n"], recovery=rec_, gain_vs_B2=gain, ours_minus_B3=vs_b3, b3_implementable=True,
                   secondary=secondary, ppl=ppl, run=rec.finish())
    write_json(P["res"] / "k3.json", summary)
    log.info(f"[K3] recovery {rec_['recovery']:.3f} [{rec_['lo']:.3f}, {rec_['hi']:.3f}]; gain {gain}")


def _bs(pre):
    b = pre["bootstrap"]
    return dict(resamples=b["resamples"], level=b["level"], seed=b["seed"])


if __name__ == "__main__":
    main()
