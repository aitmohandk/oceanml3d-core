import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from oceanml3d.cli import CONFIG_DIR, build_variables
from oceanml3d.config_schema import validate_config

NATIVE = ("data.patch.lat=144", "data.patch.lon=144", "data.stride.lat=136", "data.stride.lon=136")


def _cfg(*overrides):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="main", overrides=["experiment=osse3d_gs21_multivar_unet", *overrides])


def test_shipped_experiments_are_valid():
    for xp in ["nosc_15m_duacs", "nosc_15m_neurost_sst", "nosc_00m_duacs", "osse3d_gs21_multivar_unet",
               "osse3d_gs21_surface_only", "fourdvarnet_ssh_osse", "smoke", "osse3d_smoke"]:
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            cfg = compose(config_name="main", overrides=[f"experiment={xp}"])
        assert validate_config(cfg, variables=build_variables(cfg)) == [], xp


@pytest.mark.parametrize("override,expected", [
    ("data.stride.lat=99999", "gaps in coverage"),
    ("training.loss=huber", "training.loss='huber'"),
    ("training.loss_combine=magic", "training.loss_combine='magic'"),
    ("training.rec_weight.kind=cosine", "rec_weight.kind"),
    ("training.rec_weight.crop.lat=1000", "removes the whole patch"),
    ("export.split=validation", "export.split"),
    ("export.time=[2005-01-01,2005-12-31]", "outside the 'test' split"),
    ("data.splits.val.time=[2011-01-01,2011-12-31]", "leakage"),
    ("data.splits.test.time=[2020-01-01,2019-01-01]", "is not before stop"),
])
def test_catches_common_mistakes(override, expected):
    # osse3d_gs21 derives lat/lon patch and stride from the grid ("auto"), checked once resolved;
    # pin the native numbers so the geometric checks have something to check here.
    cfg = _cfg(*NATIVE, override)
    assert any(expected in p for p in validate_config(cfg, variables=build_variables(cfg))), override


def test_reports_missing_data_files(tmp_path):
    from oceanml3d.catalog import Catalog

    cfg = _cfg()
    variables = build_variables(cfg)
    problems = validate_config(cfg, Catalog({}, root=tmp_path), variables)
    assert any("not in the catalog" in p for p in problems)


def test_norm_stats_length_mismatch():
    cfg = _cfg()
    with open_dict(cfg):
        cfg.data.norm_stats = [[0.0, 1.0], [1.0, 1.0]]
    assert any("norm_stats has 2 entries" in p for p in validate_config(cfg, variables=build_variables(cfg)))
    assert OmegaConf.is_config(cfg)


def test_multi_stage_pipeline_is_declarative():
    """A model declares its stages; the config can override them and set per-stage trainer options."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    import numpy as np

    from oceanml3d.models.ocean.baselines.model import LinearBaseline
    from oceanml3d.variables import VariableSet

    vs = VariableSet.from_config({"a": {"source": "s", "role": "input"}, "b": {"source": "s", "role": "target"}})
    model = LinearBaseline(vs, 3, np.ones((3, 4, 4), np.float32))
    assert model.stages == ("main",)
    model.set_stage("main")
    with pytest.raises(ValueError, match="declares stages"):
        model.set_stage("prior")

    class TwoStage(LinearBaseline):
        stages = ("prior", "conditional")

    m2 = TwoStage(vs, 3, np.ones((3, 4, 4), np.float32))
    m2.set_stage("conditional")
    assert m2.current_stage == "conditional"


def test_overlap_smaller_than_twice_the_crop_is_rejected():
    """The cropped border of a patch contributes nothing, so the neighbour has to cover it. Both
    shipped tasks sit exactly on the equality (144 - 136 = 8 = 2 x 4), so any edit to patch, stride
    or crop broke the export -- with blank seams, and nothing downstream complaining."""
    cfg = _cfg(*NATIVE, "data.stride.lat=142")  # overlap 2, crop 4 -> needs 8
    problems = validate_config(cfg, variables=build_variables(cfg))
    assert any("uncovered seams" in x for x in problems), problems

    cfg = _cfg(*NATIVE)                         # overlap 8 == 2 x 4: what auto gives at 1/12 deg
    assert validate_config(cfg, variables=build_variables(cfg)) == []


GRIDDED_EXPERIMENTS = ["nosc_15m_duacs", "nosc_15m_neurost_sst", "nosc_00m_duacs",
                       "osse3d_gs21_multivar_unet", "osse3d_gs21_surface_only",
                       "fourdvarnet_ssh_osse", "lorenz96_unet", "lorenz96_enkf",
                       "smoke", "osse3d_smoke"]


def _catalog_from(site: str):
    """The site's catalog with every path pointed at a directory that does not exist, so only the
    *key* lookups matter and the test does not need the data."""
    import yaml

    from oceanml3d.catalog import Catalog

    doc = yaml.safe_load(open(f"{CONFIG_DIR}/paths/{site}.yaml"))
    return Catalog(doc.get("datasets", {}), root="/nonexistent-on-purpose")


def test_every_shipped_experiment_resolves_against_the_local_catalog():
    """`config/data/osse3d_gs21.yaml` referenced seven keys that no shipped site file defined, so
    that experiment could not run anywhere as delivered -- and nothing said so, because the
    validator only resolved `spec.source` and ignored masks and the `prepare` block entirely."""
    catalog = _catalog_from("local")
    for xp in GRIDDED_EXPERIMENTS:
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            cfg = compose(config_name="main", overrides=[f"experiment={xp}"])
        unknown = [x for x in validate_config(cfg, catalog, build_variables(cfg))
                   if "not in the catalog" in x]
        assert unknown == [], f"{xp}: {unknown}"


def test_a_site_file_does_not_invent_keys_of_its_own():
    """Site files differ in paths, not in vocabulary: a key only one of them knows is a typo, or a
    key the others silently lack."""
    import glob
    import os

    reference = set(_catalog_from("local").entries)
    for path in sorted(glob.glob(f"{CONFIG_DIR}/paths/*.yaml")):
        site = os.path.splitext(os.path.basename(path))[0]
        extra = set(_catalog_from(site).entries) - reference
        assert extra == set(), f"{site}.yaml defines keys local.yaml does not: {sorted(extra)}"


def test_an_unknown_mask_key_is_reported():
    catalog = _catalog_from("local")
    # `+` because Hydra composes in struct mode: `mask` is not declared on this variable
    cfg = _cfg("+data.variables.ssh_obs.mask=not_a_key")
    problems = validate_config(cfg, catalog, build_variables(cfg))
    assert any("mask 'not_a_key' is not in the catalog" in x for x in problems), problems


def test_a_prepare_output_need_not_exist_but_must_be_declared():
    """`prepare-obs` writes to catalog keys, so they must be declared before it runs -- but
    requiring the file to exist would make validation impossible before the first run."""
    catalog = _catalog_from("local")
    cfg = _cfg()
    problems = validate_config(cfg, catalog, build_variables(cfg))
    # Only the *outputs* are exempt. `truth` and `profiles` are inputs to the simulation and are
    # expected to be missing here, since this catalog deliberately points at nothing -- as is
    # `variable 'ssh_obs': source`, which reads one of those outputs.
    assert not any(".output" in x and "does not exist" in x for x in problems), problems
    assert any("data.prepare.pseudo_obs.truth" in x and "does not exist" in x for x in problems), \
        "inputs to the simulation must still be checked for existence"

    cfg = _cfg("data.prepare.pseudo_obs.ssh.output=not_a_key")
    problems = validate_config(cfg, catalog, build_variables(cfg))
    assert any("pseudo_obs.output 'not_a_key' is not in the catalog" in x for x in problems), problems


def test_every_site_env_file_has_a_matching_paths_file():
    """A site is two files: jobs/env/<site>.sh (environment) and config/paths/<site>.yaml (data).
    A site script pointing at a catalog that does not exist fails only once a job is queued -- which
    is what happened on Datarmor and Jean Zay, whose scripts said `TO FILL IN` for months."""
    import glob
    import os
    import re

    for path in sorted(glob.glob("jobs/env/*.sh")):
        site = os.path.splitext(os.path.basename(path))[0]
        # A leading underscore means a shared library, not a site -- `_lib.sh` holds load_modules,
        # container_exec and load_site. Same convention `load_site` uses when it lists the available
        # sites in its error message; if you add another, keep the prefix.
        if site.startswith("_"):
            continue
        text = open(path).read()
        m = re.search(r'OCEANML3D_PATHS="\$\{OCEANML3D_PATHS:-(\w+)\}"', text)
        assert m, f"{path} must set OCEANML3D_PATHS"
        declared = m.group(1)
        assert declared == site, f"{path} points at paths={declared}, expected {site}"
        assert os.path.exists(f"{CONFIG_DIR}/paths/{site}.yaml"), \
            f"{path} points at config/paths/{site}.yaml, which does not exist"


def test_the_sites_that_carry_the_osse_task_define_every_key_it_needs():
    """`test_a_site_file_does_not_invent_keys_of_its_own` catches a key too many; this one catches a
    key missing. A partial catalog is legitimate (odyssey only carries the surface datasets), so the
    check is per task: a site that claims a task must define every key that task resolves."""
    import re

    catalog = _catalog_from("local")
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="main", overrides=["experiment=osse3d_gs21_multivar_unet"])
    missed = (re.search(r"'([^']+)' is not in the catalog", p)
              for p in validate_config(cfg, _catalog_from("odyssey"), build_variables(cfg)))
    needed = {m.group(1) for m in missed if m}
    assert needed, "the odyssey catalog is expected to lack the OSSE keys; this test has gone stale"
    assert needed <= set(catalog.entries)
    for site in ("datarmor", "jeanzay"):
        missing = sorted(needed - set(_catalog_from(site).entries))
        assert missing == [], f"{site}.yaml cannot run the OSSE task: {missing} missing"


def test_every_source_of_the_osse_task_is_produced_by_something():
    """The experiment shipped with `bathy: {source: bathy_gs}` and no recipe anywhere writing a
    bathy_gs file: validation passed (the key was in the catalog), training stopped on a missing
    file. A catalog key is a promise that something fills it -- a recipe in scripts/prepare/recipes/
    or the OSSE simulator -- and this test is that promise."""
    import glob
    import re

    import yaml

    data = yaml.safe_load(open(f"{CONFIG_DIR}/data/osse3d_gs21.yaml"))
    prepare = data["prepare"]
    simulated = {prepare["pseudo_obs"]["ssh"]["output"], prepare["pseudo_obs"]["sst"]["output"],
                 prepare["virtual_argo"]["output"]}
    used = {spec["source"] for spec in data["variables"].values() if spec}
    used |= {prepare["pseudo_obs"][k]["truth"] for k in ("ssh", "sst")}
    used |= {prepare["virtual_argo"]["truth"], prepare["virtual_argo"]["profiles"]}

    catalog = yaml.safe_load(open(f"{CONFIG_DIR}/paths/local.yaml"))["datasets"]
    # A recipe's `output` is ${OCEANML3D_DATA}/<the catalog path>; ${YEAR} and friends are filled in
    # by jobs/prepare.sh, so they match anything.
    written = [re.escape(yaml.safe_load(open(r)).get("output", "")).replace(r"\$\{", "${")
               for r in glob.glob("scripts/prepare/recipes/*.yaml")]
    written = [re.sub(r"\$\{[^}]+\}", ".*", w) for w in written]

    for key in sorted(used):
        if key in simulated:
            continue                       # written by `oceanml3d command=prepare-obs`
        path = catalog[key]["path"]
        assert any(re.fullmatch(w, ".*/" + path) or re.fullmatch(w, path) for w in written), \
            (f"catalog key '{key}' ({path}) is used by config/data/osse3d_gs21.yaml but no recipe in "
             f"scripts/prepare/recipes/ writes it: either add the recipe, or drop the variable")


def test_the_bathymetry_is_off_by_default_and_the_ablation_puts_it_back():
    """It is set aside for the first step, not removed: `ablation=bathy` is the one-line way back,
    and it must keep resolving as long as the catalog key is there."""
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="main", overrides=["experiment=osse3d_gs21_multivar_unet"])
    assert "bathy" not in {v.name for v in build_variables(cfg)}
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="main",
                      overrides=["experiment=osse3d_gs21_multivar_unet", "ablation=bathy"])
    variables = build_variables(cfg)
    assert [v.source for v in variables if v.name == "bathy"] == ["bathy_gs"]
    assert validate_config(cfg, variables=variables) == []


def test_regrid_recipe_rejects_depth_indices_without_keep_depth(tmp_path):
    """`depth_indices` selects positions on the source depth axis; with `keep_depth` false the axis
    is dropped first, so the selection could never apply and the recipe silently produced a surface
    file. Ported from the NOSC preparation fixes."""
    import pytest as _pytest

    from scripts.prepare.regrid import run

    recipe = {"input": str(tmp_path / "*.nc"), "output": str(tmp_path / "out.nc"),
              "variables": {"thetao": "thetao"}, "keep_depth": False, "depth_indices": [0, 2]}
    with _pytest.raises(ValueError, match="keep_depth"):
        run(recipe)


def test_regrid_is_idempotent(tmp_path):
    """These jobs get killed on walltime; a re-run that restarts from scratch never finishes."""
    from scripts.prepare.regrid import run

    out = tmp_path / "already_there.nc"
    out.write_bytes(b"")                       # content irrelevant: existence is the contract
    recipe = {"input": str(tmp_path / "*.nc"), "output": str(out), "variables": {}}
    assert run(recipe) == out                  # no input files, yet it returns without reading


def test_container_exec_puts_the_clone_ahead_of_the_image_package():
    """`container/oceanml3d.def` copies `oceanml3d/` into the image at build time, so without this an
    old image runs old code against the clone's new scripts and configs -- which is how a job died on
    `module 'oceanml3d.obs.argo' has no attribute 'coverage_table_local'`, hours after the function
    was committed. PYTHONPATH precedes site-packages in sys.path, so the bound clone wins."""
    import subprocess
    import textwrap

    # A real directory: container_exec skips a bind whose source does not exist on the host, and a
    # PYTHONPATH pointing at an unbound path would be worse than none.
    script = textwrap.dedent("""
        set -eu
        source jobs/env/_lib.sh
        repo=$(mktemp -d)
        export OCEANML3D_SIF=/nonexistent.sif OCEANML3D_CONTAINER_CMD=echo OCEANML3D_NV=0
        REPO_ROOT="$repo" container_exec python -c pass
        echo "PYTHONPATH_S=$SINGULARITYENV_PYTHONPATH"
        echo "PYTHONPATH_A=$APPTAINERENV_PYTHONPATH"
        echo "REPO=$repo"
    """)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    repo = next(x.split("=", 1)[1] for x in out.splitlines() if x.startswith("REPO="))
    assert f"PYTHONPATH_S={repo}" in out, out
    assert f"PYTHONPATH_A={repo}" in out, out
    # and the repository is bound, otherwise the path would point at nothing inside the container
    assert f"--bind {repo}:{repo}" in out, out


def test_the_image_recipe_does_not_claim_the_package_is_not_baked_in():
    """It is: `%files` copies `oceanml3d` to /opt/oceanml3d. The help text used to say the opposite,
    which is exactly the belief that let a stale image go unsuspected for two failed jobs."""
    text = open("container/oceanml3d.def").read()
    assert "oceanml3d      /opt/oceanml3d/oceanml3d" in text, "the copy moved: update this test"
    assert "not baked in" not in text
    assert "PYTHONPATH" in text


def test_every_pbs_job_declares_a_queue():
    """`jobs/pbs/train.pbs` shipped with `##PBS -q gpuq` commented out as "site-specific". Datarmor's
    default queue is a *routing* queue with no destination that accepts `ngpus`, so the submission was
    refused outright -- `qsub: Job rejected by all possible destinations`, which names neither the
    queue nor the resource, before a single line of the file ran. The four preparation jobs all
    declare `-q omp`; a job file that leaves it out is not portable, it is unsubmittable."""
    import glob
    import re

    for path in sorted(glob.glob("jobs/pbs/*.pbs")):
        text = open(path).read()
        assert re.search(r"^#PBS -q \w+", text, re.M), f"{path} declares no queue"
        assert not re.search(r"^##PBS -q ", text, re.M), \
            f"{path} has a commented-out queue: either it is needed, or remove the line"


def test_a_gpu_pbs_job_asks_for_a_gpu_and_a_cpu_one_does_not():
    """The two are different queues on every centre; asking for `ngpus` in a CPU queue is the other
    half of the same rejection."""
    import re

    train = open("jobs/pbs/train.pbs").read()
    assert "ngpus=" in train and re.search(r"^#PBS -q gpuq", train, re.M)
    for name in ("prepare_glorys", "prepare_obs", "prepare_argo", "concat_glorys"):
        text = open(f"jobs/pbs/{name}.pbs").read()
        assert "ngpus=" not in text, f"{name}.pbs asks for a GPU in a CPU queue"


def test_run_sh_goes_through_container_exec_and_not_its_own_apptainer_line():
    """`jobs/run.sh` carried a second copy of the apptainer invocation, and the two drifted: every
    fix made to `container_exec` -- creating the data directory before binding it, skipping a bind
    whose source does not exist, and putting the clone ahead of the image's package on PYTHONPATH --
    reached the preparation jobs and not training, which then failed on `Primary config directory
    not found. Check that '/opt/oceanml3d/config' exists`."""
    import re
    import subprocess
    import tempfile
    import textwrap

    text = open("jobs/run.sh").read()
    assert "container_exec oceanml3d" in text
    assert not re.search(r"^\s*(exec\s+)?\"?\$\{OCEANML3D_CONTAINER_CMD", text, re.M), \
        "run.sh builds its own container command again"

    with tempfile.TemporaryDirectory() as d:
        script = textwrap.dedent(f"""
            set -eu
            printf '#!/bin/sh\\necho CONTAINER "$@"\\n' > {d}/fake; chmod +x {d}/fake
            export OCEANML3D_SIF={d}/img.sif OCEANML3D_CONTAINER_CMD={d}/fake OCEANML3D_NV=0
            export OCEANML3D_DATA={d}/data OCEANML3D_PATHS=local OCEANML3D_USER_ENV=/dev/null
            jobs/run.sh --site local command=validate experiment=smoke
        """)
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                             check=True, cwd=".").stdout
    assert "CONTAINER exec" in out and "oceanml3d command=validate" in out
    repo = __import__("pathlib").Path.cwd().resolve()
    assert f"--bind {repo}:{repo}" in out


def test_training_falls_back_to_csv_when_tensorboard_is_missing(tmp_path, capsys):
    """An image built without tensorboard failed the job in build_trainer, after the data had been
    opened and the model built. TensorBoard is a viewer; the CSV log has every scalar it would show."""
    from oceanml3d.cli import _loggers

    class NoTensorBoard:
        def __init__(self, *a, **k):
            raise ModuleNotFoundError("Neither `tensorboard` nor `tensorboardX` is available.")

    class CSV:
        def __init__(self, *a, **k):
            pass

    loggers = _loggers(str(tmp_path), NoTensorBoard, CSV)
    assert len(loggers) == 1 and isinstance(loggers[0], CSV)
    assert "TensorBoard logging off" in capsys.readouterr().out


def test_the_image_installs_tensorboard():
    """environment.yml always had it; pyproject.toml, which the image installs from, did not."""
    assert "logging" in open("container/oceanml3d.def").read().split("pip install --no-cache-dir -e")[1].split("\n")[0]
    assert "tensorboard" in open("pyproject.toml").read()


def test_the_generic_glorys_recipe_is_per_year_and_keeps_zos():
    """The recipe for sites without a mirror had no ${YEAR}: every task of the array rewrote the same
    2010-2020 store, nothing reached by_year/ for `concat`, it read ${OCEANML3D_RAW} which no site set,
    and it dropped `zos`, which the `glorys_gs_surface` key reads from this very store."""
    import glob

    import yaml

    for path in glob.glob("scripts/prepare/recipes/glorys_gs_multidepth*.yaml"):
        r = yaml.safe_load(open(path))
        assert "${YEAR}" in r["output"] and "/by_year/" in r["output"], path
        assert "zos" in r["variables"], f"{path} drops zos"
        assert "${GLORYS_SRC}" in r["input"], f"{path} does not read through GLORYS_SRC"
        if "download" in r:
            assert "zos" in r["download"]["variables"] and "${YEAR}" in r["download"]["output"], path


def test_every_variable_a_recipe_uses_is_set_by_the_job_layer():
    """`${OCEANML3D_RAW}` appeared in five recipes and no site file set it: the pattern stayed
    literal and matched nothing. Every `${VAR}` in a recipe must be set by load_site, a site file, or
    jobs/prepare.sh -- or be one of the per-run variables the job wrappers export."""
    import glob
    import re

    provided = set()
    for f in glob.glob("jobs/env/*.sh") + ["jobs/prepare.sh"]:
        provided |= set(re.findall(r"export\s+([A-Z_][A-Z0-9_]*)=", open(f).read()))
        provided |= set(re.findall(r"export\s+([A-Z_][A-Z0-9_]*)\s", open(f).read()))
    provided |= {"YEAR", "NEXT", "OCEANML3D_TARGET_RES"}
    for path in glob.glob("scripts/prepare/recipes/*.yaml"):
        used = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", open(path).read()))
        assert used <= provided, f"{path} uses {sorted(used - provided)}, which nothing sets"


def test_prepare_download_is_a_no_op_on_a_site_that_mirrors_glorys(tmp_path):
    """A mirror site's recipe has no `download:` block; running the download script on it failed on
    a missing dataset_id. On a copy of jobs/ with a fabricated mirror site, since the real site files
    load modules that do not exist here."""
    import shutil
    import subprocess

    repo = tmp_path / "repo"
    shutil.copytree("jobs", repo / "jobs")
    (repo / "scripts/prepare/recipes").mkdir(parents=True)
    (repo / "jobs/env/mirror.sh").write_text(
        'export OCEANML3D_DATA="${OCEANML3D_DATA:-/tmp}"\nexport OCEANML3D_PATHS=local\n'
        'source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"\n')
    (repo / "scripts/prepare/recipes/glorys_gs_multidepth.mirror.yaml").write_text(
        "input: /mirror/*_${YEAR}*.nc\noutput: ${OCEANML3D_DATA}/by_year/g_${YEAR}.zarr\n")
    out = subprocess.run(["bash", "-c", f"OCEANML3D_DATA={tmp_path}/data OCEANML3D_USER_ENV=/dev/null "
                          f"{repo}/jobs/prepare.sh --site mirror download 2015"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "nothing to download" in out.stdout


def test_load_site_sets_raw_and_glorys_src_ahead_of_the_resolution_suffix(tmp_path):
    import subprocess

    script = (f"source jobs/env/_lib.sh; OCEANML3D_USER_ENV=/dev/null OCEANML3D_DATA={tmp_path} "
              f"OCEANML3D_TARGET_RES=0.5 load_site local $PWD >/dev/null; "
              f'echo "RAW=$OCEANML3D_RAW"; echo "SRC=$GLORYS_SRC"; echo "DATA=$OCEANML3D_DATA"')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert f"RAW={tmp_path}/raw" in out and f"SRC={tmp_path}/raw/glorys" in out, out
    assert f"DATA={tmp_path}/res0.5" in out, "a download is the same whatever grid it is put on"


def test_the_image_installs_the_download_tools():
    """Jean Zay has no GLORYS or Argo mirror, so its only paths are copernicusmarine and argopy --
    neither of which the image installed, while the runbook said copernicusmarine was in it."""
    text = open("container/oceanml3d.def").read()
    install = text.split("pip install --no-cache-dir -e")[1].split("\n")[0]
    assert "prepare" in install
    assert "copernicusmarine" in text and "argopy" in text
    assert "argopy" in open("pyproject.toml").read()
