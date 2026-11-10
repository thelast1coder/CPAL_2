"""Build calibration, validation and analysis token sets (PHASE1_KILL_TEST.md Section 4).

Real: FineWeb-Edu + MetaMathQA (chat template), split by document, disjointness and leakage checks logged.
Tiny: synthetic sequences. Output: <data dir>/{calibration,validation,analysis}.pt and data_info.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prouting.calibrate import data as D
from prouting.utils.cli import base_parser, done, model_cfg, paths, setup
from prouting.utils.io import save_torch, write_json
from prouting.utils.logging import run_record


def main():
    args = base_parser(__doc__).parse_args()
    cfg, log = setup(args, "prepare_data")
    P = paths(args, cfg)
    out = P["data"] / "calibration.pt"
    if done(out, args):
        log.info(f"skip: {out} exists")
        return
    rec = run_record(cfg["data"], args.seed)
    mcfg = cfg["models"][args.model]
    quotas = {"analysis": mcfg["analysis_tokens"], "validation": cfg["data"]["validation_tokens"],
              "calibration": cfg["data"]["calibration_tokens"]}
    info = {}
    if args.tiny:
        splits = D.tiny_splits(quotas, seq_len=cfg["data"]["seq_len"], seed=args.seed)
    else:
        from transformers import AutoTokenizer
        mc = model_cfg(mcfg["stats_model"])
        tok = AutoTokenizer.from_pretrained(mc["hf_id"], revision=mc.get("revision"))
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        from prouting.eval.prompts import math_chat_example
        tpl_tok = tok
        if not mc.get("chat_template"):
            # base model: use the Instruct tokenizer's chat template for maths examples (same tokenizer family)
            tm = model_cfg(mcfg["task_model"])
            tpl_tok = AutoTokenizer.from_pretrained(tm["hf_id"], revision=tm.get("revision"))
        chat = lambda q, a: math_chat_example(tpl_tok, q, a)
        total = sum(quotas.values())
        mix = cfg["data"]["mix"]
        # generous document counts: text docs give one 1024-token chunk at most; maths examples are short
        n_text = int(total * mix["fineweb_edu"] / 300) + 2000
        n_maths = int(total * mix["metamathqa"] / 150) + 2000
        docs = D.load_documents(cfg["data"]["sources"], n_text, n_maths, chat, log=log.info)
        docs, info["leakage"] = D.leakage_filter(docs, D.test_questions(cfg["data"]["sources"]), log=log.info)
        splits = D.split_documents(docs, tok, quotas, {D.TEXT: mix["fineweb_edu"], D.MATHS: mix["metamathqa"]},
                                   cfg["data"]["seq_len"], seed=args.seed)
    for name, st in splits.items():
        save_torch(P["data"] / f"{name}.pt", st.to_dict())
        info[name] = dict(tokens=st.n_tokens, sequences=int(st.input_ids.shape[0]),
                          documents=len(set(st.doc_hash)),
                          domain_tokens={int(d): int(st.attention_mask[st.domain == d].sum()) for d in st.domain.unique().tolist()})
        log.info(f"[data] {name}: {info[name]}")
    info["disjoint_by_document_hash"] = True
    write_json(P["data"] / "data_info.json", dict(info=info, run=rec.finish()))


if __name__ == "__main__":
    main()
