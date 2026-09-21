"""docs/tutorial.md, Part 1, run from the page itself.

The commands are not copied here: they are extracted from the ```bash blocks of the tutorial's Part 1
and run in order, in one shell, the way a reader would type them. Edit a command on the page and this
test runs the edited command; break the code under a command and this test fails. A block preceded by
`<!-- tutorial-test: skip ... -->` is not run (a command meant to fail, the scoring step that needs the
other repository, the clean-up).

Why it exists: the chain broke in a new place almost every day of its first week on a cluster -- a
recipe that only worked on one site, a runner that bypassed the shared container wrapper, a derived
file reused without a look, a DataLoader that deadlocked -- each piece tested on its own. This is the
test of the pieces together. Steps 0-4 (preparation to validation) run in the fast suite; steps 5-6
(training, inference), a couple of minutes, are marked slow.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "tutorial.md"


def part1_blocks() -> list[tuple[int, str]]:
    """``[(step number, bash block), ...]`` of Part 1, in page order, skips removed."""
    text = PAGE.read_text()
    part1 = text.split("## Part 1", 1)[1].split("## Part 2", 1)[0]
    blocks, step = [], None
    pattern = r"^### Step (\d+)|(<!-- tutorial-test: skip[^>]*-->\s*)?```bash\n(.*?)```"
    for m in re.finditer(pattern, part1, flags=re.S | re.M):
        if m.group(1) is not None:
            step = int(m.group(1))
        elif not m.group(2) and step is not None:
            blocks.append((step, m.group(3)))
    return blocks


def _run(script: str, workdir: Path, timeout: int) -> str:
    # The page writes under $PWD/tutorial; point that at a temporary directory so the test does not
    # touch the clone, and keep the machine's own ~/.config/oceanml3d/env.sh out of it.
    script = script.replace("$PWD/tutorial", str(workdir / "tutorial"))
    env = {**os.environ, "OCEANML3D_USER_ENV": "/dev/null",
           "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
    for var in ("OCEANML3D_SIF", "OCEANML3D_DATA", "OCEANML3D_TARGET_RES", "ARGO_GDAC", "GLORYS_SRC"):
        env.pop(var, None)
    out = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, f"--- stdout\n{out.stdout[-4000:]}\n--- stderr\n{out.stderr[-4000:]}"
    return out.stdout


def _script(blocks, steps) -> str:
    return "\n".join(b for s, b in blocks if s in steps)


@pytest.fixture(scope="module")
def runs_dir():
    """Hydra writes run directories under outputs/tutorial/ in the clone; remove the ones this test made."""
    before = set((REPO / "outputs" / "tutorial").glob("*"))
    yield
    import shutil

    for d in set((REPO / "outputs" / "tutorial").glob("*")) - before:
        shutil.rmtree(d, ignore_errors=True)


def test_part1_has_the_steps_the_other_tests_expect():
    steps = sorted({s for s, _ in part1_blocks()})
    assert steps == [0, 1, 2, 3, 4, 5, 6], steps


def test_steps_0_to_4_prepare_and_validate(tmp_path, runs_dir):
    blocks = part1_blocks()
    log = _run(_script(blocks, range(0, 5)), tmp_path, timeout=900)
    data = tmp_path / "tutorial" / "data" / "res0.5"
    assert sorted(p.name for p in (data / "by_year").iterdir()) == [
        "glorys_gs_multidepth_2018.zarr", "glorys_gs_multidepth_2019.zarr"]
    assert (data / "argo" / "argo_profiles_gs.csv").exists()
    for name in ("pseudo_obs_ssh_gs.nc", "pseudo_obs_sst_gs.nc", "argo_virtual_thetao_gs21.nc"):
        assert (data / "osse" / name).exists(), name
    # what the page says each step prints
    for line in ("[prepare] done: glorys", "[prepare] done: concat", "[argo] per year: 2018",
                 "virtual ARGO:", "exported product empty over the outer rim: lat 1 deg",
                 "configuration 'tutorial' is valid"):
        assert line in log, line
    assert "time: 90" in log and "depth: 26" in log


@pytest.mark.slow
def test_steps_0_to_6_train_and_infer(tmp_path, runs_dir):
    blocks = part1_blocks()
    log = _run(_script(blocks, range(0, 7)), tmp_path, timeout=1800)
    assert "product manifest written to" in log
    assert "absent" not in log and "share no date" not in log
    assert "contract: OK | 64 variables" in log
