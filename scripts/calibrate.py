"""Two-pass calibration (PHASE1_KILL_TEST.md Section 5). Output artifacts/<model>/calib_seed<s>.pt."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from prouting.adapters import get_adapter
from prouting.calibrate.data import SplitTokens
from prouting.calibrate.driver import calibrate, calib_to_dict
from prouting.utils.cli import base_parser, concrete_model, done, load_model, paths, setup
from prouting.utils.io import load_torch, save_torch, write_json
from prouting.utils.logging import run_record


def main():
    args = base_parser(__doc__).parse_args()
    cfg, log = setup(args, "calibrate")
    P = paths(args, cfg)
    if done(P["calib"], args):
        log.info(f"skip: {P['calib']} exists")
        return
    rec = run_record(cfg["calibration"], args.seed)
    cc = cfg["calibration"]
    model, _ = load_model(concrete_model(args, cfg), args)
    ad = get_adapter(model)
    split = SplitTokens.from_dict(load_torch(P["data"] / "calibration.pt"))
    batches = list(split.batches(cc["batch_size"]))
    t0 = time.time()
    with torch.no_grad():
        calib = calibrate(model, ad, batches, clusters=tuple(cc["clusters"]), pass1_tokens=cc["pass1_tokens"],
                          sketch_fraction=cc["sketch_fraction"], n_min=cc["n_min"],
                          full_gram_tokens=cc["full_gram_tokens"], seed=args.seed, log=log.info)
    save_torch(P["calib"], calib_to_dict(calib))
    summary = {li: dict(tokens=c.gram.tokens, backoff_share=c.gram.backoff_share, r_bar=c.gram.r_bar)
               for li, c in calib.items()}
    write_json(P["res"] / f"calibration_summary_{args.target}.json",
               dict(tokens=split.n_tokens, wall_time_s=time.time() - t0, layers=summary, run=rec.finish()))


if __name__ == "__main__":
    main()
