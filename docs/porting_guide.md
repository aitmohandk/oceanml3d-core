# How to port a remaining capability

The open items are in `PLAN.md` §3; what each one is and where it comes from is in
`docs/feature_inventory.md`. This file is the *procedure* — the same seven steps for every item,
so that a port is finished when it is proven, not when it runs.

## The rule

> A port is done when the new implementation is shown to agree with the original on a fixed input,
> or when the difference is deliberate and written down.

Rewriting the code is the easy half. The failure mode of this whole project is a silent change of
the science: a port that runs, produces plausible numbers, and is 3 % off. Everything below exists
to make that impossible to miss.

## Seven steps

**1. Read the original and write down the equations.** Not the API — the semantics. Do this before
touching a keyboard, and expect surprises: the two-scale Lorenz-96 of `fm-opencode` was found this
way (its `lorenz96` is not the textbook single-scale system, and porting the wrong one would have
invalidated every toy comparison). Record what you found in the item's docstring.

**2. Pin the reference behaviour.** On a machine where the original runs, add a case to
`tests/legacy/generate_golden.py` and commit the resulting `golden/<case>.npz` (it stores the legacy
commit hash). Keep the case cheap, deterministic and free of trained weights. If the original cannot
be run at all (lost environment, GPU-only), say so in the inventory and fall back to step 5 alone.

**3. Decide where it belongs.** Use the existing seams rather than adding one:
dynamics → `oceanml3d/dynamics/`; a method that turns inputs into targets → a registered model;
a training-time behaviour → `oceanml3d/training/`; an observation-system property → `oceanml3d/obs/`;
a score → `oceanml3d-eval/metrics/`. If nothing fits, that is a design discussion, not a quick fix.

**4. Port in the framework's idioms.** numpy/xarray where torch is not required (so it is testable
without a GPU and usable by `oceanml3d-eval`), `VariableSet` instead of channel indices, catalog keys
instead of paths, config instead of a new subclass. Copying a class verbatim into a new folder is
not a port — it re-creates the dependency tangle the framework exists to remove.

**5. Prove it.** Add the equivalence test in `tests/legacy/test_equivalence.py` with an explicit
tolerance *and the reason it is not zero*. Add behavioural tests that would catch a wrong port even
without the golden file: conserved quantities, known asymptotics, an ordering that must hold
(an EnKF must beat the raw observations; a calibrated ensemble must score ~1 on spread-skill).

**6. Wire it into the config tree and the benchmarks.** A capability nobody can reach from a YAML
file is not ported. Add the `config/` entry, and if it changes a score, run
`oceanml3d-eval baseline --check` on the affected experiments before merging.

**7. Update the paper trail.** Move the line in `docs/feature_inventory.md` from *planned* to *done*,
delete the `PLAN.md` item, add a `CHANGELOG.md` entry (Summary / Files / Rationale / Verification).
An item is not closed until the inventory says so.

## Suggested order

Ports are cheapest when what they depend on already exists.

1. **Independent, no dependency** — ETKF and 4D-Var (§3.4), remaining metrics (§3.5). Same shape as
   the EnKF and the metrics already there; good first tasks, and they immediately give every learned
   method a stronger reference.
2. **Needs one new abstraction** — QG dynamics (§3.1: a `Dynamics` with a 2D state), interior model
   (§3.7: a new `head:`). Small, contained.
3. **Needs the two-stage hook and the ensemble format** (both already in place) — CFM and SDA (§3.2).
   The largest item; do it after 1 and 2 so the classical references are ready to compare against.
4. **Needs a parameter channel** — joint state+parameter estimation (§3.3). Depends on §3.2 for the
   joint generative variants.
5. **Assembly only** — case studies CS1–CS4 and the toy case studies (§3.6). Do last: they are
   config files over capabilities that must already exist.

## Two traps

**Do not port the scripts.** `reports/**`, `demos/`, `eval_*.py` are one-off drivers. The reusable
part of a report is a metric or a plot function; extract that and leave the driver behind.

**Do not port a capability you cannot evaluate.** If nothing in `oceanml3d-eval` can score it, the port
has no way of being checked and no way of being compared. Add the metric first — that is why §3.5
sits early in the order.
