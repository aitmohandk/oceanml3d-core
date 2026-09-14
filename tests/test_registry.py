from oceanml3d.registry import _MODELS, get_model, list_models, register_model


def test_register_and_get():
    @register_model("dummy_test_model")
    class Dummy:  # noqa: D401
        pass

    assert get_model("dummy_test_model") is Dummy
    assert "dummy_test_model" in list_models()
    _MODELS.pop("dummy_test_model")


def test_missing_torch_is_explained():
    """Without the training extra the registry is empty; the error must say so, not 'unknown model'."""
    import pytest

    from oceanml3d import registry

    try:
        import pytorch_lightning  # noqa: F401
        import torch  # noqa: F401
        stack_available = True
    except ImportError:
        stack_available = False
    if stack_available:
        assert "nosc_unet" in list_models()
        return
    with pytest.raises(KeyError, match="install the training extra"):
        get_model("nosc_unet")
    assert registry._IMPORT_ERROR is not None
