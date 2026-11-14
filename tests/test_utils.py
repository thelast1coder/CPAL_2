"""Plumbing added before P1: pinned model sources, the GPU queue, and the K0 stop in the runner."""
import re
import shutil
import subprocess

import pytest

from prouting.utils.cli import model_cfg, model_source
from prouting.utils.gpu import wait_for_memory
from prouting.utils.logging import REPO_ROOT

SHA = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize("name", ["olmoe_1b_7b", "olmoe_1b_7b_instruct", "qwen3_30b_a3b"])
def test_kill_test_models_are_pinned(name):
    repo, rev = model_source(model_cfg(name), fp8=False)
    assert "/" in repo and rev and SHA.match(rev)


def test_fp8_uses_its_own_revision():
    mc = model_cfg("qwen3_30b_a3b")
    repo8, rev8 = model_source(mc, fp8=True)
    repo, rev = model_source(mc, fp8=False)
    assert repo8.endswith("-FP8") and SHA.match(rev8) and rev8 != rev
    assert model_source({**mc, "use_fp8": True}) == (repo8, rev8)


def test_wait_for_memory_is_a_no_op_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert wait_for_memory(1e6, max_wait_min=0) == 0.0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_runner_rejects_unknown_stop_point():
    r = subprocess.run(["bash", "scripts/run_kill_test.sh", "--tiny", "--stop-after", "k3"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--stop-after takes k0 or k2" in r.stdout
