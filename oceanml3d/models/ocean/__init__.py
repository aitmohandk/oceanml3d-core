"""Import built-in models so they register themselves."""
from oceanml3d.registry import get_model, list_models, register_model  # noqa: F401

try:
    from oceanml3d.models.ocean.assimilation.model import (  # noqa: F401
        EnsembleKalmanFilter,
        OptimalInterpolation,
    )
    from oceanml3d.models.ocean.baselines.model import (  # noqa: F401
        Climatology,
        LinearBaseline,
        Passthrough,
    )
    from oceanml3d.models.ocean.fourdvarnet.model import FourDVarNet  # noqa: F401
    from oceanml3d.models.ocean.nosc.model import NOSCUNet  # noqa: F401
except ImportError as exc:  # torch/lightning missing: keep the registry usable and say why
    from oceanml3d import registry

    registry.set_import_error(exc)
