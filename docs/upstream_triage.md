# Upstream triage: `4dvarnet-fm-opencode` since the import

`PROVENANCE.md` records that this repository is derived from, and independent of,
`CIA-Oceanix/4dvarnet-fm-opencode` — no remote, no tracking, no rebase. The open question it leaves
is what to do about what has been published upstream since. This note answers it, commit by commit,
so the decision is on record rather than re-litigated every few months.

**Import point:** upstream `2c709f6` (PR #174, 2026-09-08), identified by blob comparison on
`models/fourdvarnet.py` and `train.py`.
**Reviewed:** upstream `a83b12d` (PR #221), **44 commits**, on 2026-09-15.

## The short version

**Forty-one of the forty-four are L96 or QG science, and every file they touch is in this
repository's reserve.** That is not an accident of timing: upstream is a methodology bench on
toy systems, this repository took that methodology to gridded ocean data, and the two have been
working on different things since the split. Taking those commits would mean maintaining a research
programme this project is not running.

Three are worth something, and only one of them is a patch.

## The one that matters: longitudinal wrap-around

`2bb4f6c feat(qg): MONAI circular 2D U-Net for Q1 (doubly-periodic domain) (#190)`

Upstream found that convolving a doubly-periodic field with ordinary zero padding puts an artificial
boundary where the domain wraps, and moved to circular padding.

**The same defect exists here, and it is not hypothetical.** `config/data/surface_currents_15m.yaml`
declares `lon: [-180, 180]` — a *globally periodic* field, 1440 points at 1/4 degrees. Both live
trunks pad with zeros: `unet_nosc.py` uses `nn.Conv2d(..., padding=pad)`, whose `padding_mode`
defaults to `"zeros"`, and MONAI's `DiffusionModelUNet` does the same. So every convolution that
straddles the antimeridian sees zeros where the ocean continues, at every level of the U-Net. With
five levels the receptive field is on the order of a hundred cells, so the affected band is wide, it
sits over the Pacific, and nothing in the loss or the metrics singles it out — the error is simply
part of the reported skill.

**Not fixed here, deliberately.** Two reasons. `unet_nosc.py` is explicitly frozen — its docstring
says "kept, unchanged", because it is how every run published before the MONAI switch is reproduced —
so the fix belongs in the MONAI trunk or in a new option, not there. And circular padding changes
results: it is a modelling decision that needs a before/after comparison on real data, which is
blocked on the same acceptance run as the rest of lot 1.

**Proposed shape**, when someone picks it up: an opt-in `lon_circular: bool = False` on the trunk,
implemented as `F.pad(x, (pad, pad, 0, 0), mode="circular")` before each convolution with
`padding=(pad, 0)` — circular in longitude, ordinary padding in latitude, since latitude is *not*
periodic and wrapping the Arctic onto the Antarctic would be worse than a seam. The natural test is
equivariance: rolling the input along longitude should roll the output by the same amount.
`osse3d_gs21`, a regional box, is unaffected and must stay so.

## The two worth considering

`436bc77 chore: adopt CHANGELOG.d/ fragment files to eliminate cross-PR conflicts (#193)`

One file per change under `CHANGELOG.d/`, concatenated at release. Upstream adopted it for the
reason we have hit three times in this repository's short history: every branch appends to the top of
`CHANGELOG.md`, so every second branch conflicts there and nowhere else. A workflow choice rather
than a patch — worth taking if agent branches keep running in parallel, pointless otherwise.

`45bbe48 feat: checkpoint-resume + archive-not-delete run history for train.py (#178)`

Resume semantics and not clobbering previous run directories. Applies to `legacy/train.py`, but the
*idea* transfers: the gridded CLI resumes by passing `ckpt=` to the first stage only, and does not
archive anything. Worth a look the first time a long gridded run dies at hour nine.

## The rest

Reviewed and declined. Grouped, with what each would touch here:

| Upstream work | Commits | Lands in |
|---|---|---|
| L96 benchmarks, FDV1/FDV2, MONAI backbones, SDA hybrids, FM-prior sampler | 14 | `oceanml3d/legacy/` |
| QG: S0/S1 cases, ETKF/EnKF campaigns, DirectUNet families, obs-density sweeps | 16 | `oceanml3d/legacy/`, `reports/` |
| Reports, figures, animations, benchmark tables | 6 | `reports/` |
| Paper scoping and docs reorganisation (JAMES, ML venues) | 5 | nothing here |
| CI gating the whole test tree (#215), ruff debt (#187), single conda env (#210) | 3 | **already done here, independently** |

That last row is worth a note. Upstream gated its whole test tree in #215 and paid down its ruff debt
in #187; this repository did both in lots 0 and 5, from its own audit, without knowing. Convergent
solutions to the same inherited problem — which is mild evidence that the problems were real rather
than matters of taste.

Two upstream fixes look like bugs but are not ours: `a0d52e5` (double-upsampled CRPS ensembles under
cross-resolution S1) and `53da6d0` (a hardcoded margin in a report) both live in QG reporting code.
`57b28fd` fixes a NaN in FDV2 gradient channels, in `models/solver.py` — reserve here, and the
gridded 4DVarNet is a different class.

## Decision

**Do not adopt a tracking remote.** The trees have diverged by design and 41 of 44 commits are for a
research programme this project does not run. Individual items are taken as patches, by hand, and
recorded in `CHANGELOG.md` — which is what `PROVENANCE.md` already prescribes.

**Review cadence:** on this evidence, once or twice a year is enough, and only when something in the
gridded path looks wrong in a way upstream might already have diagnosed. Re-reading 44 commits cost
an afternoon and yielded one finding; the finding was worth it, but not monthly.

## Open items from this review

- [ ] Longitudinal wrap-around in the trunk for `surface_currents_15m` (see above). **Blocked on the
      lot 1 acceptance run**, which is what would show its effect.
- [ ] Decide on `CHANGELOG.d/`, based on whether parallel branches continue.
- [ ] Revisit resume/archive semantics for the gridded CLI after the first long run.
