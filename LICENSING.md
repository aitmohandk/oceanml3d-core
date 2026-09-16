# Licensing and provenance

**Status: resolved.** This repository is distributed under the **European Union Public Licence
v. 1.2 (EUPL-1.2)**. `LICENSE` holds the verbatim licence text, `NOTICE` holds the copyright and
contributor statement. This note records how that position was arrived at. It is a factual summary,
not legal advice.

## The problem

`pyproject.toml` declared `license = { text = "MIT" }` and the repository carried no `LICENSE` file.
That declaration was unsupported: a substantial part of this tree is copied from two upstream
repositories, and neither permitted it.

## What was verified (2026-09-15)

| Source | Licence as published | Evidence |
|---|---|---|
| `CIA-Oceanix/NOSC` (via `aitmohandk/NOSC`) | **CeCILL-C** | `license.md`, 522 lines of CeCILL-C text; header: *Copyright IMT Atlantique/OceaniX, contributor(s): T. Picard, R. Fablet, S. Ouala, P. Haslée (IMT Atlantique)* |
| `CIA-Oceanix/4dvarnet-fm-opencode` | **none stated** | no `LICENSE`, `COPYING` or licence metadata anywhere in the tree at `2c709f6`. The only CeCILL-C mention is `docs/research_notes_cfm_da_originality_and_benchmarking.md:125`, and it refers to `4dvarnet-starter` / `4dvarnet-core` — the source of the ported `GradSolver` / `ConvLstmGradModel` — not to that repository itself |

Roughly 120 files in `oceanml3d/models/ocean/`, `oceanml3d/obs/`, `oceanml3d/inference/` and
`oceanml3d/data/` come from the first; most of `oceanml3d/legacy/`, `legacy/` and `config/legacy/`
comes from the second. A repository with no stated licence is by default all rights reserved, which
is stricter than CeCILL-C, not looser.

## Why the move to EUPL-1.2 needed permission, not a compatibility argument

CeCILL-C is copyleft at file level: Article 5.3.4 requires derivatives of the covered software to
carry the same terms. The EUPL's own Appendix lists its compatible licences, and it names **CeCILL
v. 2.0 and v. 2.1 — not CeCILL-C**. There was therefore no automatic compatibility route from the
NOSC files to the EUPL, and none was relied on.

What made the change legitimate is simpler: **both upstream repositories have the same rightsholder**
— IMT Atlantique / OceaniX — and that rightsholder granted permission to distribute this work,
including the material derived from both sources, under EUPL-1.2. That permission also settles the
second source, which had published no terms at all.

> **To complete:** record the permission below, so the basis is traceable without leaving the repo.
>
> * Granted by: _<name, role, IMT Atlantique / OceaniX>_
> * Date: _<YYYY-MM-DD>_
> * Reference: _<email thread, ticket, or signed note — keep the original outside the repository>_

## Provenance by directory

Paths reflect the layout after `5948ee5`, which moved the toy / Lorenz / QG family into reserve. All
of it is now distributed under EUPL-1.2; the table records where the code came from, which is what
`NOTICE` and per-file attribution are for.

| Path | Origin |
|---|---|
| `oceanml3d/models/ocean/`, `oceanml3d/obs/`, `oceanml3d/inference/`, `oceanml3d/catalog.py`, `oceanml3d/variables.py`, `oceanml3d/registry.py`, `oceanml3d/data/{open,patches,datamodule,transforms,augment}.py`, `scripts/prepare/` | NOSC (reworked) |
| `oceanml3d/legacy/` (models, evaluation, data, conf, training), `legacy/` (the 20 driver scripts), `config/legacy/`, `demos/`, `reports/`, `batch/` | `4dvarnet-fm-opencode` @ `2c709f6` |
| `oceanml3d/cli.py`, `oceanml3d/config_schema.py`, `oceanml3d/models/ocean/nn/unet_monai.py`, `oceanml3d/training/{loss_grouping,weights,vertical_modes,callbacks}.py`, `config/`, `docs/`, most of `tests/` | written for this repository |

`oceanml3d/models/ocean/nn/unet_monai.py` wraps `monai.networks.nets.DiffusionModelUNet`
(Project MONAI, Apache-2.0). MONAI is a declared dependency, not vendored, so nothing of it is
redistributed here.

See `PROVENANCE.md` for the commit-level history.

## Follow-ups

1. **`oceanml3d-eval` carries the same unsupported MIT declaration** and needs the identical
   treatment — `LICENSE`, `NOTICE`, `pyproject.toml` — including for `product_contract.py`, which is
   duplicated byte for byte across both repositories and is therefore covered twice.
2. **Per-file SPDX headers.** The EUPL is normally applied with a short header per source file:

   ```python
   # SPDX-FileCopyrightText: IMT Atlantique / OceaniX
   # SPDX-License-Identifier: EUPL-1.2
   ```

   Not yet applied. Several modules already name their NOSC counterpart in the docstring
   (`models/ocean/nosc/model.py`, `data/open.py`, `catalog.py`, `inference/predict.py`); that
   attribution should be kept when the headers are added.
3. **Ask the rightsholder to state EUPL-1.2 on `4dvarnet-fm-opencode` itself.** The permission covers
   this repository; the upstream one still publishes no terms, which will trip up the next person who
   forks it.
