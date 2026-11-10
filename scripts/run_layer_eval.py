"""K1 and K2 offline from stored per-token Grams (PHASE1_KILL_TEST.md Section 7).

1. Primary "ours": among the candidates, lowest relative error energy at m = 2 averaged over layers on the
   VALIDATION records. 2. All methods on the ANALYSIS records at every budget. 3. K1 statistics (law).
4. K2 ratios per layer and aggregate, U1/U2 headroom, rescue (ours-sk) only if the primary is not GO.
Output results/.../layer_eval.json and layer_eval.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import statistics

from prouting.analysis.law import layer_law
from prouting.analysis.layer_eval import KILL_METHODS, eval_method
from prouting.analysis.recorder import list_layers, load_shard
from prouting.analysis.theory import closed_form_vs_measured, layer_theory
from prouting.calibrate.driver import calib_from_dict
from prouting.routing.baselines import K2Fix
from prouting.utils.cli import base_parser, done, load_yaml, paths, setup
from prouting.utils.io import load_torch, write_json, write_text
from prouting.utils.logging import REPO_ROOT, run_record


def method_specs(cfg) -> dict:
    specs = dict(KILL_METHODS)
    k2 = cfg["downstream"].get("b3_k2")
    b3 = K2Fix(m=1, k2=k2)
    specs["B3"] = {"fn": lambda rec, calib, m, _b=b3: _b.offline(rec, calib, m)}
    return specs


def k2_cell(r1, r2, go, amber):
    if r1 >= go and r2 >= go:
        return "GO"
    if r1 >= amber and r2 >= amber:
        return "AMBER"
    return "NO-GO"


def main():
    args = base_parser(__doc__).parse_args()
    cfg, log = setup(args, "run_layer_eval")
    P = paths(args, cfg)
    out = P["res"] / "layer_eval.json"
    if done(out, args):
        log.info(f"skip: {out} exists")
        return
    pre = load_yaml(REPO_ROOT / "configs" / "prereg_kill_test.yaml")
    le = cfg["layer_eval"]
    rec = run_record(dict(layer_eval=le, prereg=pre["k1"] | {"k2": pre["k2"]}), args.seed)
    calib = calib_from_dict(load_torch(P["calib"]))
    layers = list_layers(P["analysis"])
    specs = method_specs(cfg)
    ridge, clus = le["ridge"], le["clusters"]

    # 1. primary variant on validation, m = 2
    val_err = {c: [] for c in le["candidates"]}
    for li in list_layers(P["validation"]):
        sh = load_shard(P["validation"], li)
        for c in le["candidates"]:
            val_err[c].append(eval_method(sh, calib[li], specs[c], m=2, ridge=ridge, clusters=clus)["rel_error_energy"])
    val_mean = {c: sum(v) / len(v) for c, v in val_err.items()}
    primary = min(val_mean, key=val_mean.get)
    log.info(f"[K2] validation mean error at m=2: {val_mean}; primary = {primary}")

    # 2. all methods on analysis; 3. law; secondary theory
    per_layer = {}
    rows = ["layer,method,m,rel_error_energy,median_token_rel_error,median_err_over_resid,mean_active"]
    for li in layers:
        sh = load_shard(P["analysis"], li)
        L = dict(methods={}, law=None)
        for name in le["methods"]:
            for m in le["budgets"]:
                r = eval_method(sh, calib[li], specs[name], m=m, ridge=ridge, clusters=clus)
                L["methods"].setdefault(name, {})[m] = r
                rows.append(f"{li},{name},{m},{r['rel_error_energy']:.6g},{r['median_token_rel_error']:.6g},"
                            f"{r.get('median_err_over_resid', float('nan')):.6g},{r['mean_active']:.4f}")
        law = layer_law(sh)
        law.pop("_m95", None)
        L["law"] = law
        L["closed_form"] = {m: closed_form_vs_measured(sh, m, calib[li].gram.r_bar) for m in (1, 2)}
        L["theory"] = layer_theory(sh, m_values=(1, 2))
        per_layer[li] = L
        log.info(f"[layer {li}] m95 median {law['m95']['median']}, PR median {law['pr']['median']:.2f}")

    # K1
    k1 = pre["k1"]
    med = {li: per_layer[li]["law"]["m95"]["median"] for li in layers}
    share_go = sum(v <= k1["go"]["median_m95_at_most"] for v in med.values()) / len(med)
    share_nogo = sum(v >= k1["nogo"]["median_m95_at_least"] for v in med.values()) / len(med)
    k1_cell = ("GO" if share_go >= k1["go"]["layer_share_at_least"]
               else "NO-GO" if share_nogo >= k1["nogo"]["layer_share_at_least"] else "AMBER")

    # K2
    k2 = pre["k2"]

    def ratios(tag, m):
        out = {}
        for li in layers:
            M = per_layer[li]["methods"]
            ref = min(M["B1"][m]["rel_error_energy"], M["B2"][m]["rel_error_energy"])
            ours = M[tag][m]["rel_error_energy"]
            out[li] = ref / max(ours, 1e-300)
        return out

    agg = {}
    for tag in [primary, "U1", "U2", "W-avg"]:
        agg[tag] = {m: statistics.median(ratios(tag, m).values()) for m in k2["budgets"]}
    r1, r2 = agg[primary][1], agg[primary][2]
    k2_primary = k2_cell(r1, r2, k2["go_ratio_at_least"], k2["amber_ratio_at_least"])
    rescue = None
    k2_final = k2_primary
    if k2_primary != "GO" and k2["rescue"]["allowed_only_if_primary_not_go"]:
        sk = {}
        for li in layers:
            sh = load_shard(P["analysis"], li)
            if "G_sk" not in sh:
                break
            for m in (1, 2, 3):
                sk.setdefault(li, {})[m] = eval_method(sh, calib[li], specs["ours-sk"], m=m, ridge=ridge)["rel_error_energy"]
        if sk:
            rr = {}
            extra_ok = True
            for m in k2["budgets"]:
                vals = []
                for li in layers:
                    M = per_layer[li]["methods"]
                    ref_m = min(M["B1"][m]["rel_error_energy"], M["B2"][m]["rel_error_energy"])
                    vals.append(ref_m / max(sk[li][m], 1e-300))
                    ref_next = min(M["B1"][m + 1]["rel_error_energy"], M["B2"][m + 1]["rel_error_energy"])
                    extra_ok &= sk[li][m] <= ref_next
                rr[m] = statistics.median(vals)
            cell = k2_cell(rr[1], rr[2], k2["go_ratio_at_least"], k2["amber_ratio_at_least"])
            counts = cell == "GO" and extra_ok
            rescue = dict(ratios=rr, cell=cell, extra_condition_met=extra_ok, counts=counts, per_layer=sk)
            if counts:
                k2_final = "GO"
    headroom_note = None
    if min(agg["U1"][1], agg["U1"][2]) < k2["go_ratio_at_least"]:
        headroom_note = "U1 (oracle) is below the 2x bar: no calibrated method can pass K2 on this model."

    result = dict(
        model=args.model, tiny=args.tiny, primary=primary, validation_mean_error_m2=val_mean,
        k1=dict(median_m95_per_layer=med, share_at_most_3=share_go, share_at_least_5=share_nogo, cell=k1_cell),
        k2=dict(aggregate_ratio=agg, primary_cell=k2_primary, rescue=rescue, cell=k2_final,
                per_layer_ratio={m: ratios(primary, m) for m in k2["budgets"]}, headroom_note=headroom_note),
        per_layer=per_layer, run=rec.finish())
    write_json(out, result)
    write_text(P["res"] / "layer_eval.csv", "\n".join(rows) + "\n")
    log.info(f"[K1] {k1_cell} (share m95<=3: {share_go:.2f}); [K2] {k2_final} (ratios m1 {r1:.2f}, m2 {r2:.2f})")


if __name__ == "__main__":
    main()
