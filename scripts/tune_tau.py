"""One global tau per target mean of active experts, on validation records only (PHASE1_KILL_TEST.md 8).

Uses the primary Gram variant chosen in run_layer_eval.py. With --target task, tunes on the task model's
validation records (used by K3). Reports realised means on the analysis records when they exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prouting.analysis.layer_eval import KILL_METHODS, eval_method, tune_tau
from prouting.analysis.recorder import list_layers, load_shard
from prouting.calibrate.driver import calib_from_dict
from prouting.utils.cli import base_parser, done, paths, setup
from prouting.utils.io import load_torch, read_json, write_json
from prouting.utils.logging import run_record


def main():
    args = base_parser(__doc__).parse_args()
    cfg, log = setup(args, "tune_tau")
    P = paths(args, cfg)
    out = P["res"] / f"tau_{args.target}.json"
    if done(out, args):
        log.info(f"skip: {out} exists")
        return
    primary = read_json(P["res"] / "layer_eval.json")["primary"]
    spec = KILL_METHODS[primary]
    rec = run_record(dict(primary=primary, targets=cfg["tau_targets"], target_model=args.target), args.seed)
    calib = calib_from_dict(load_torch(P["calib"]))
    val = {li: load_shard(P["validation"], li) for li in list_layers(P["validation"])}
    res = {}
    for t in cfg["tau_targets"]:
        tau, got = tune_tau(val, calib, spec, t)
        entry = dict(tau=tau, validation_mean=got)
        if P["analysis"].exists() and list_layers(P["analysis"]):
            tot = n = 0
            for li in list_layers(P["analysis"]):
                r = eval_method(load_shard(P["analysis"], li), calib[li], spec, tau=tau)
                tot += r["mean_active"] * r["n_tokens"]
                n += r["n_tokens"]
            entry["analysis_mean"] = tot / n
        res[str(t)] = entry
        log.info(f"[tau] target {t}: {entry}")
    write_json(out, dict(primary=primary, taus=res, run=rec.finish()))


if __name__ == "__main__":
    main()
