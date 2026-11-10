"""Native forward on the analysis and validation tokens, storing per-token records for every MoE layer
(PHASE1_KILL_TEST.md Section 6). Output artifacts/<model>/{analysis,validation}/layer_XX.pt."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from prouting.adapters import get_adapter
from prouting.analysis.recorder import AnalysisRecorder
from prouting.calibrate.data import SplitTokens
from prouting.calibrate.driver import calib_from_dict
from prouting.routing.patch import patch_model
from prouting.utils.cli import base_parser, concrete_model, load_model, paths, setup
from prouting.utils.io import load_torch, write_json
from prouting.utils.logging import run_record


def main():
    p = base_parser(__doc__)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    cfg, log = setup(args, "record_analysis")
    P = paths(args, cfg)
    rec = run_record(dict(batch_size=args.batch_size, target=args.target), args.seed)
    model, _ = load_model(concrete_model(args, cfg), args)
    ad = get_adapter(model)
    calib = calib_from_dict(load_torch(P["calib"]))
    info = {}
    # the task model only needs validation records (tau tuning for K3)
    for split in (("validation", "analysis") if args.target == "stats" else ("validation",)):
        out = P[split]
        if (out / "done.json").exists() and not args.force:
            log.info(f"skip: {out} done")
            continue
        if out.exists():
            shutil.rmtree(out)
        st = SplitTokens.from_dict(load_torch(P["data"] / f"{split}.pt"))
        h = patch_model(model, ad, calib=calib)
        recd = AnalysisRecorder(h, model, record_sketch=True, clusters=("c8", "c16"))
        dev = next(model.parameters()).device
        with torch.no_grad():
            for bi, s in enumerate(range(0, st.input_ids.shape[0], args.batch_size)):
                ids = st.input_ids[s:s + args.batch_size].to(dev)
                mask = st.attention_mask[s:s + args.batch_size].to(dev)
                h.set_valid(mask)
                h.shared["domain"] = st.domain[s:s + args.batch_size].view(-1, 1).expand_as(ids)
                model(input_ids=ids, attention_mask=mask)
                if bi % 16 == 15:
                    recd.flush(out)
            recd.flush(out)
        recd.remove()
        h.restore()
        size = sum(f.stat().st_size for f in out.glob("*.pt"))
        info[split] = dict(tokens=st.n_tokens, bytes=size)
        log.info(f"[record] {split}: {st.n_tokens} tokens, {size / 1e6:.1f} MB")
        write_json(out / "done.json", info[split])
    write_json(P["res"] / f"recording_summary_{args.target}.json", dict(info=info, run=rec.finish()))


if __name__ == "__main__":
    main()
