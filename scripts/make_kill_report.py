"""Apply the pre-registered decision table and write KILL_REPORT.md and kill_test.json
(PHASE1_KILL_TEST.md Sections 10 and 12). Thresholds are read from configs/prereg_kill_test.yaml only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prouting.utils.cli import base_parser, load_yaml, setup
from prouting.utils.io import read_json, write_json, write_text
from prouting.utils.logging import REPO_ROOT, git_info, run_record


def k3_cell(k3: dict, pre: dict) -> tuple[str, dict]:
    rec = k3["recovery"]["recovery"]
    gain_ok = k3["gain_vs_B2"]["lo"] > 0
    b3_ok = True
    if k3.get("b3_implementable", False):
        b3_ok = 100 * k3["ours_minus_B3"]["lo"] >= pre["go"]["ours_minus_b3_ci_lower_at_least_points"]
    why = dict(recovery=rec, gain_ci_excludes_zero=gain_ok, b3_condition=b3_ok)
    if rec >= pre["go"]["recovery_at_least"] and gain_ok:
        return ("GO" if b3_ok else "AMBER"), why
    if rec >= pre["amber"]["recovery_at_least"] and gain_ok:
        return "AMBER", why
    return "NO-GO", why


def overall(cells: dict[str, dict[str, str]]) -> str:
    """cells: model -> {K1, K2, K3}. Exactly the rule in PREREG_KILL_TEST.md Section 4 (decision)."""
    ms = list(cells)
    if len(ms) < 2 or any(c is None for m in ms for c in cells[m].values()):
        return "INCOMPLETE"
    col = lambda k: [cells[m][k] for m in ms]
    if all(c == "NO-GO" for c in col("K1")) or all(c == "NO-GO" for c in col("K2")) or all(c == "NO-GO" for c in col("K3")):
        return "NO-GO"
    ok = lambda k: "GO" in col(k) and "NO-GO" not in col(k)
    if all(c == "GO" for c in col("K1")) and ok("K2") and ok("K3"):
        return "GO"
    return "AMBER"


def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def try_plots(data: dict, out_dir: Path, log) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed: plots skipped")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m, d in data.items():
        le = d.get("layer_eval")
        if not le:
            continue
        med = le["k1"]["median_m95_per_layer"]
        xs = sorted(med, key=int)
        ax.plot([int(x) for x in xs], [med[x] for x in xs], marker="o", label=m)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("median m95")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "k1_m95_by_layer.png", dpi=120)
    made.append("k1_m95_by_layer.png")
    for m, d in data.items():
        le = d.get("layer_eval")
        if not le:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        methods = list(next(iter(le["per_layer"].values()))["methods"])
        for meth in methods:
            ys, xs = [], []
            for mm in sorted(next(iter(le["per_layer"].values()))["methods"][meth], key=int):
                tot = [L["methods"][meth][mm]["rel_error_energy"] for L in le["per_layer"].values()]
                xs.append(int(mm))
                ys.append(sum(tot) / len(tot))
            ax.semilogy(xs, [max(y, 1e-12) for y in ys], marker="o", label=meth)
        ax.set_xlabel("kept experts m")
        ax.set_ylabel("relative error energy (mean over layers)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"k2_error_vs_m_{m}.png", dpi=120)
        made.append(f"k2_error_vs_m_{m}.png")
    plt.close("all")
    return made


def main():
    args = base_parser(__doc__).parse_args()
    cfg, log = setup(args, "make_kill_report")
    pre = load_yaml(REPO_ROOT / "configs" / "prereg_kill_test.yaml")
    rec = run_record(dict(prereg_version=pre["prereg_version"]), args.seed)
    base = REPO_ROOT / "results" / ("tiny" if args.tiny else "kill_test")
    data, cells = {}, {}
    for m in pre["models"]:
        d = {}
        for name, fn in (("layer_eval", "layer_eval.json"), ("k3", "k3.json"), ("tau", "tau_task.json"),
                         ("k0", "k0.json")):
            f = base / m / fn
            d[name] = read_json(f) if f.exists() else None
        data[m] = d
        c = {"K1": None, "K2": None, "K3": None}
        if d["layer_eval"]:
            c["K1"] = d["layer_eval"]["k1"]["cell"]
            c["K2"] = d["layer_eval"]["k2"]["cell"]
        if d["k3"]:
            c["K3"], d["k3_why"] = k3_cell(d["k3"], pre["k3"])
        cells[m] = c
    verdict = overall(cells)
    L = []
    if args.tiny:
        L += ["# TINY RUN, NUMBERS ARE MEANINGLESS", "",
              "Tiny random models and synthetic tokens; this report only exercises the pipeline and the decision logic.", ""]
    L += ["# Kill test report", "",
          f"Verdict: **{verdict}**.",
          "Plain summary (three lines): " + ("pipeline check only." if args.tiny else "(edit after reading the numbers)"),
          "", "## Setup", "",
          f"- Prereg version {pre['prereg_version']}; git {git_info()['git_hash']} (dirty: {git_info()['git_dirty']}).",
          f"- Models: {', '.join(pre['models'])}. Precision and revisions: see results/STATUS.md (decision D7).", ""]
    L += ["## K0 sanity", ""]
    for m, d in data.items():
        if d["k0"]:
            L.append(f"- {m}: " + ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in d["k0"]["checks"].items()))
        else:
            L.append(f"- {m}: not run")
    L += ["", "## K1: redundancy", "", "| model | share of layers with median m95 <= 3 | share >= 5 | cell |", "|---|---|---|---|"]
    for m, d in data.items():
        le = d["layer_eval"]
        if le:
            L.append(f"| {m} | {fmt(le['k1']['share_at_most_3'])} | {fmt(le['k1']['share_at_least_5'])} | {le['k1']['cell']} |")
    L += ["", "Per-layer median m95 and PR:", ""]
    for m, d in data.items():
        le = d["layer_eval"]
        if le:
            row = ", ".join(f"L{li}: {v['law']['m95']['median']:.0f} / {v['law']['pr']['median']:.2f}"
                            for li, v in le["per_layer"].items())
            L.append(f"- {m}: {row}")
    L += ["", "## K2: exploitable with calibration", "",
          "| model | primary | ratio m=1 | ratio m=2 | U1 m=1 / m=2 | U2 m=1 / m=2 | W-avg m=1 / m=2 | cell |",
          "|---|---|---|---|---|---|---|---|"]
    for m, d in data.items():
        le = d["layer_eval"]
        if le:
            a = le["k2"]["aggregate_ratio"]
            p = le["primary"]
            g = lambda t, mm: a[t][str(mm)] if str(mm) in a[t] else a[t][mm]
            L.append(f"| {m} | {p} | {fmt(g(p, 1))} | {fmt(g(p, 2))} | {fmt(g('U1', 1))} / {fmt(g('U1', 2))} | "
                     f"{fmt(g('U2', 1))} / {fmt(g('U2', 2))} | {fmt(g('W-avg', 1))} / {fmt(g('W-avg', 2))} | {le['k2']['cell']} |")
            L.append(f"  - validation error at m=2 per candidate: {le['validation_mean_error_m2']}")
            if le["k2"]["rescue"]:
                L.append(f"  - rescue (ours-sk): {le['k2']['rescue']['ratios']}, counts: {le['k2']['rescue']['counts']}")
            if le["k2"]["headroom_note"]:
                L.append(f"  - {le['k2']['headroom_note']}")
    L += ["", "## K3: survives the network", "",
          "| model | task | m* | B0 | B2 | B2+P | B3 | B3+P | ours-fixed | ours-adaptive | recovery [95% CI] | gain vs B2 (points) [CI] | ours - B3 (points) [CI] | cell |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in data.items():
        k3 = d["k3"]
        if k3:
            acc = k3["accuracy"]
            ms = k3["m_star"]
            r, gn, b3 = k3["recovery"], k3["gain_vs_B2"], k3["ours_minus_B3"]
            pct = lambda t: f"{100 * acc[t]:.1f}" if t in acc else "n/a"
            L.append(f"| {m} | {k3['task']} | {ms} | {pct('B0')} | {pct(f'B2_m{ms}')} | {pct('B2+P')} | {pct('B3')} | "
                     f"{pct('B3+P')} | {pct('ours-fixed')} | {pct('ours-adaptive')} | {fmt(r['recovery'])} "
                     f"[{fmt(r['lo'])}, {fmt(r['hi'])}] | {100 * gn['diff']:.1f} [{100 * gn['lo']:.1f}, {100 * gn['hi']:.1f}] | "
                     f"{100 * b3['diff']:.1f} [{100 * b3['lo']:.1f}, {100 * b3['hi']:.1f}] | {cells[m]['K3']} |")
            L.append(f"  - n = {k3['n']}; realised mean active experts: " +
                     ", ".join(f"{t} {v:.2f}" for t, v in k3["mean_active"].items()))
            if k3.get("ppl"):
                L.append("  - perplexity: " + ", ".join(f"{t} {v['ppl']:.3f}" for t, v in k3["ppl"].items()))
    L += ["", "## Decision table", "", "| model | K1 | K2 | K3 |", "|---|---|---|---|"]
    for m, c in cells.items():
        L.append(f"| {m} | {c['K1']} | {c['K2']} | {c['K3']} |")
    L += ["", f"Overall: **{verdict}** (rule: {pre['decision']['go']} => GO; {pre['decision']['nogo']} => NO-GO; otherwise AMBER).",
          "", "## What the numbers mean", "", "(to be written after reading the numbers)", ""]
    plots = try_plots(data, (REPO_ROOT / "plots" / ("tiny" if args.tiny else "")), log)
    if plots:
        L += ["Plots: " + ", ".join(plots), ""]
    out_md = base / "KILL_REPORT.md"
    write_text(out_md, "\n".join(L))
    write_json(base / "kill_test.json", dict(verdict=verdict, cells=cells, data=data, tiny=args.tiny, run=rec.finish()))
    log.info(f"[report] {out_md}: {verdict}")


if __name__ == "__main__":
    main()
