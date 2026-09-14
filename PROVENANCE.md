# Provenance

`oceanml3d-core` is **derived from** two repositories and **independent of** both. It is not a fork:
it has no upstream remote, it does not track their branches, and changes published there after the
import are deliberately not merged.

| Origin | What was taken | Status |
|---|---|---|
| [`CIA-Oceanix/4dvarnet-fm-opencode`](https://github.com/CIA-Oceanix/4dvarnet-fm-opencode) | the 4DVarNet and flow-matching core: models, solvers, Lorenz-63/96 and quasi-geostrophic dynamics, the Lightning training pipeline, the EnKF/ETKF/4D-Var baselines, the Hydra configuration tree, the test suite | imported once, in commit `036491d`; its plan is kept verbatim as `PLAN_upstream.md` |
| [`aitmohandk/NOSC`](https://github.com/aitmohandk/NOSC) | the multivariate ocean reconstruction work: the multivariate model, OSSE observation operators, the Argo layer, the lazy data layer and its configurations | ported through a prototype, then transplanted into this tree (see `CHANGELOG.md` and `docs/feature_inventory.md`) |

## Why not a fork

The two origins answer different questions from this project. `4dvarnet-fm-opencode` is a
research sandbox for flow matching on toy dynamics; `oceanml3d-core` targets 3D multivariate ocean
reconstruction evaluated by a separate benchmark repository (`oceanml3d-eval`), and the
restructuring that requires — one installable `oceanml3d/` package, a model registry, a data layer
built on lazy xarray — is not a change that could be sent back upstream as a patch series.

Keeping a live upstream remote would therefore buy merge conflicts on every structural commit and
nothing else. **Decision: no upstream remote, no rebase, no tracking of later upstream commits.**
The 24 commits published on `4dvarnet-fm-opencode` after the import are knowingly not integrated.

If a specific upstream fix is ever wanted, take it as a patch by hand and record it in
`CHANGELOG.md` — do not add a remote.

## Licensing — open point

Both origins are public repositories, but **neither this repository nor `oceanml3d-eval` carries a
`LICENSE` file**, while both declare `license = { text = "MIT" }` in `pyproject.toml`. That
declaration is unbacked, and a substantial part of this tree is copied from
`4dvarnet-fm-opencode`, whose own terms have to govern that code. Before publishing: check the two
origins' licences, add the matching `LICENSE` here, and retain their notices wherever their code was
copied verbatim.
