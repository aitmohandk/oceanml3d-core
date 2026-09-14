# Equivalence harness

Porting is only finished when the new implementation is shown to agree with the original one.
Two tiers, so the check runs both on a laptop and in CI:

* **Tier A — golden files (always run).** `golden/*.npz` holds outputs produced *by the original
  code*, committed to git. `test_equivalence.py` runs the oceanml3d implementation on the same inputs
  and compares within a documented tolerance. No legacy repo needed.
* **Tier B — direct comparison (run where the originals are checked out).** `generate_golden.py`
  imports the legacy repo and regenerates the files. Run it once per ported item, commit the result,
  and record the legacy commit hash it came from (stored inside the npz).

```bash
python tests/legacy/generate_golden.py --fm ~/src/4dvarnet-fm-opencode --case lorenz96_2scale
pytest tests/legacy -q            # tier A, part of the normal gate
```

A golden file is a scientific artefact: never regenerate it to make a test pass. If the comparison
fails, either the port is wrong, or the difference is intended — in which case document it in
`docs/feature_inventory.md` and bump the case's `tolerance` with the reason in the changelog.
