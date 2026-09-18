# Adding a model

1. Create `oceanml3d/models/ocean/<name>/model.py`. Note the `ocean/`: the gridded family
   lives one level down since the NOSC transplant, and `oceanml3d/models/__init__.py` is
   empty *on purpose*, so that no pre-existing import of the toy family changed behaviour.

```python
from oceanml3d.models.ocean.base import BaseOceanModel
from oceanml3d.registry import register_model

@register_model("my_model")
class MyModel(BaseOceanModel):
    def __init__(self, variables, window, rec_weight, hidden=64, **base_kw):
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        self.net = ...            # n_in = len(variables.inputs), n_out = len(variables.targets)

    def forward(self, batch):     # batch: (B, n_channels, T, H, W) normalised
        x = self.inputs(batch)    # (B, n_in, T, H, W), NaN -> 0
        ...
        return y                  # (B, n_out, T, H, W) normalised
```

   Everything else (loss on finite target cells, logging, denormalised `predict_step`,
   optimizer) is inherited. Override `step` if the loss is non-standard, or
   `configure_optimizers` for exotic schedules.

2. Add `config/model/my_model.yaml` (`name: my_model` + hyper-parameters) and an
   `config/experiment/<xp>.yaml` that picks a `data` task and your model.

3. Register it in `oceanml3d/models/ocean/__init__.py` (built-in) **or**, for an external
   package, in its `pyproject.toml`:
   ```toml
   [project.entry-points."oceanml3d.models"]
   my_model = "mypkg.model:MyModel"
   ```

4. Add `oceanml3d/models/ocean/<name>/README.md` (inputs / targets / window / reference) and a
   test in `tests/` that runs a forward pass on the synthetic data. A forward pass is not
   enough on its own: `test_the_trunk_can_actually_fit_a_batch` exists because a model can
   return correctly shaped output and have learnt nothing, and every other test still
   passes in that state. Overfit one batch and require the loss to fall.

## Models that need more than the default batch

* observation masks / OSSE: give the variable `role: both` and a `mask:` key; the model
  gets the masked field as input and the full field as target.
* several time windows or 3D (depth): extend `PatchSpec` dims — `PatchArray` is generic
  over the `DIMS` tuple; adding `"depth"` is a 3-line change.
* two-stage training (4dvarnet-fm): keep one `BaseOceanModel` per stage and chain them in
  a custom `command` in `cli.py`, or freeze stage-1 weights inside stage-2 `__init__`.

## Where the base class does the work for you

Read `oceanml3d/models/ocean/base.py` before overriding anything:

* `inputs(batch)` / `targets(batch)` slice the channel axis from the `VariableSet`, so a model never
  indexes channels by hand.
* `predict_step` returns **denormalised** values, which is why the inference layer needs to know
  nothing about the model.
* `on_save_checkpoint` / `on_load_checkpoint` carry the normalisation statistics and the channel
  layout with the weights. If you override either, call `super()` — `NOSCUNet` does, and forgetting
  it means a checkpoint that denormalises with whatever statistics the current config happens to
  produce.
* `stages` plus `set_stage` drive multi-stage training; the CLI runs one `fit` per stage.
