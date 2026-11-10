"""T16: decision logic on every outcome, and the tiny end-to-end run writing every output file."""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_T16_decision_table():
    rep = _load("make_kill_report")
    c = lambda k1, k2, k3: {"K1": k1, "K2": k2, "K3": k3}
    o = rep.overall
    assert o({"a": c("GO", "GO", "GO"), "b": c("GO", "AMBER", "AMBER")}) == "GO"
    assert o({"a": c("GO", "GO", "GO"), "b": c("GO", "NO-GO", "GO")}) == "AMBER"
    assert o({"a": c("GO", "GO", "GO"), "b": c("AMBER", "GO", "GO")}) == "AMBER"
    assert o({"a": c("NO-GO", "GO", "GO"), "b": c("NO-GO", "GO", "GO")}) == "NO-GO"
    assert o({"a": c("GO", "NO-GO", "GO"), "b": c("GO", "NO-GO", "GO")}) == "NO-GO"
    assert o({"a": c("GO", "GO", "NO-GO"), "b": c("GO", "GO", "NO-GO")}) == "NO-GO"
    assert o({"a": c("GO", "GO", None), "b": c("GO", "GO", "GO")}) == "INCOMPLETE"
    import yaml
    pre = yaml.safe_load((ROOT / "configs" / "prereg_kill_test.yaml").read_text())["k3"]
    k3 = lambda rec, lo, b3lo: dict(recovery={"recovery": rec}, gain_vs_B2={"lo": lo},
                                    ours_minus_B3={"lo": b3lo}, b3_implementable=True)
    assert rep.k3_cell(k3(0.6, 0.01, -0.005), pre)[0] == "GO"
    assert rep.k3_cell(k3(0.6, 0.01, -0.02), pre)[0] == "AMBER"      # B3 condition fails
    assert rep.k3_cell(k3(0.3, 0.01, 0.0), pre)[0] == "AMBER"
    assert rep.k3_cell(k3(0.6, -0.01, 0.0), pre)[0] == "NO-GO"       # gain interval includes zero
    assert rep.k3_cell(k3(0.2, 0.01, 0.0), pre)[0] == "NO-GO"
    le = _load("run_layer_eval")
    assert le.k2_cell(2.1, 2.0, 2.0, 1.3) == "GO"
    assert le.k2_cell(2.1, 1.5, 2.0, 1.3) == "AMBER"
    assert le.k2_cell(2.1, 1.2, 2.0, 1.3) == "NO-GO"


def _bash():
    for b in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe"):
        if b and Path(b).exists():
            return b
    return None


@pytest.mark.skipif(_bash() is None, reason="bash not available")
def test_T16_tiny_end_to_end():
    """Runs (or resumes) the tiny pipeline; about 6 minutes from scratch on a laptop CPU."""
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    r = subprocess.run([_bash(), "scripts/run_kill_test.sh", "--tiny"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = ROOT / "results" / "tiny"
    report = (res / "KILL_REPORT.md").read_text(encoding="utf-8")
    assert report.startswith("# TINY RUN, NUMBERS ARE MEANINGLESS")
    assert "## Decision table" in report and "Overall: **" in report
    assert (res / "kill_test.json").exists()
    for m in ("olmoe_1b_7b", "qwen3_30b_a3b"):
        for f in ("k0.json", "layer_eval.json", "layer_eval.csv", "tau_task.json", "k3.json",
                  "calibration_summary_stats.json", "recording_summary_stats.json"):
            assert (res / m / f).exists(), (m, f)
        art = ROOT / "artifacts" / "tiny" / m
        assert (art / "calib_seed0.pt").exists() and list((art / "analysis").glob("layer_*.pt"))
