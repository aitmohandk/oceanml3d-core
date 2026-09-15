import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from oceanml3d.models.ocean.nn.unet_monai import MonaiUNet2d, default_norm_num_groups  # noqa: E402
from oceanml3d.models.ocean.nn.unet_nosc import UNetNosc  # noqa: E402


def _break_zero_init(net: MonaiUNet2d) -> MonaiUNet2d:
    """MONAI zero-initialises the output conv *and* every residual block's second conv, so a fresh
    trunk is exactly the zero map (see :func:`test_zero_initialisation_is_inherited_from_monai`).
    Anything testing the *value* of the output — or that dropout does something — has to break that
    first, or it is comparing zeros."""
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d) and module.weight.abs().sum() == 0:
            torch.nn.init.normal_(module.weight, std=0.1)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=0.1)
    return net


@pytest.mark.parametrize("widths", [(8, 16), (8, 16, 32), (16, 32, 64, 128)])
def test_shape_parity_with_the_nosc_trunk(widths):
    x = torch.randn(2, 6, 32, 32)
    monai_out = MonaiUNet2d(6, 4, widths)(x)
    nosc_out = UNetNosc(6, 4, widths)(x)
    assert monai_out.shape == nosc_out.shape == (2, 4, 32, 32)


def test_auto_norm_num_groups_accepts_narrow_widths():
    """MONAI's own default of 32 would reject widths=(8, 16, 32), which the smoke tests use."""
    assert default_norm_num_groups((8, 16, 32)) == 8
    assert default_norm_num_groups((64, 128, 256, 512, 1024)) == 32
    net = MonaiUNet2d(3, 2, (8, 16, 32))
    assert all(m.num_groups == 8 for m in net.modules() if isinstance(m, torch.nn.GroupNorm))


def test_explicit_norm_num_groups_wins():
    net = MonaiUNet2d(3, 2, (8, 16, 32), norm_num_groups=4)
    assert all(m.num_groups == 4 for m in net.modules() if isinstance(m, torch.nn.GroupNorm))


def test_attention_lands_on_the_requested_level():
    from monai.networks.nets import diffusion_model_unet as dmu

    net = MonaiUNet2d(3, 2, (8, 16, 32), attention_levels=(2,))
    attended = [i for i, b in enumerate(net.backbone.down_blocks)
                if any(isinstance(m, dmu.SpatialAttentionBlock) for m in b.modules())]
    assert attended == [2]                       # level 2 = widths[2], two downsamplings deep
    heads = next(m for m in net.backbone.down_blocks[2].modules() if isinstance(m, dmu.SpatialAttentionBlock))
    assert heads.attn.num_heads == 4             # 32 channels / num_head_channels=8


def test_attention_level_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        MonaiUNet2d(3, 2, (8, 16, 32), attention_levels=(3,))
    with pytest.raises(ValueError, match="out of range"):
        MonaiUNet2d(3, 2, (8, 16, 32), attention_levels=(0,))


def test_attention_heads_must_divide_the_level_width():
    with pytest.raises(ValueError, match="does not divide"):
        MonaiUNet2d(3, 2, (8, 12, 32), attention_levels=(1,), attention_heads=8)


@pytest.mark.parametrize("size", [(30, 30), (17, 33), (16, 16)])
def test_sizes_that_are_not_multiples_of_the_downsampling_factor(size):
    """DiffusionModelUNet concatenates skips, so it needs every dimension to halve exactly;
    UNetNosc padded inside its Up block. The wrapper restores that tolerance."""
    net = _break_zero_init(MonaiUNet2d(3, 2, (8, 16, 32)))
    out = net(torch.randn(1, 3, *size))
    assert out.shape == (1, 2, *size)
    assert torch.isfinite(out).all()


def test_zero_initialisation_is_inherited_from_monai():
    """Documented deviation from UNetNosc, inherited from MONAI's ``zero_module``: the output conv
    and every resblock's conv2 start at zero, so the whole trunk is the zero map at step 0. Pinned
    so it cannot change silently, and so the tests above are read as deliberately breaking it."""
    from monai.networks.nets import diffusion_model_unet as dmu

    net = MonaiUNet2d(3, 2, (8, 16))
    assert net.backbone.out[2].conv.weight.abs().sum() == 0
    blocks = [m for m in net.modules() if isinstance(m, dmu.DiffusionUNetResnetBlock)]
    assert blocks and all(b.conv2.conv.weight.abs().sum() == 0 for b in blocks)
    assert torch.count_nonzero(net(torch.randn(1, 3, 16, 16))) == 0


def test_dropout_is_attached_and_active_only_in_train_mode():
    from monai.networks.nets import diffusion_model_unet as dmu

    net = _break_zero_init(MonaiUNet2d(3, 2, (8, 16, 32), dropout=0.5))
    blocks = [m for m in net.modules() if isinstance(m, dmu.DiffusionUNetResnetBlock)]
    assert blocks and all(hasattr(b, "dropout") for b in blocks)
    x = torch.randn(1, 3, 32, 32)
    net.train()
    assert not torch.allclose(net(x), net(x))
    net.eval()
    assert torch.allclose(net(x), net(x))


def test_no_dropout_by_default():
    from monai.networks.nets import diffusion_model_unet as dmu

    net = MonaiUNet2d(3, 2, (8, 16))
    assert not any(hasattr(m, "dropout") for m in net.modules() if isinstance(m, dmu.DiffusionUNetResnetBlock))


def test_residual_input_channel_adds_that_channel():
    x = torch.randn(2, 3, 32, 32)
    net = _break_zero_init(MonaiUNet2d(3, 1, (8, 16), residual_input_channel=2))
    net.residual_input_channel = None
    plain = net(x)
    assert torch.count_nonzero(plain) > 0
    net.residual_input_channel = 2
    assert torch.allclose(net(x), plain + x[:, 2:3])


def test_backprop_is_finite():
    net = _break_zero_init(MonaiUNet2d(4, 2, (8, 16, 32), dropout=0.1, attention_levels=(2,)))
    out = net(torch.randn(2, 4, 32, 32))
    out.pow(2).mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_widths_needs_at_least_two_levels():
    with pytest.raises(ValueError, match="at least two levels"):
        MonaiUNet2d(3, 2, (8,))
