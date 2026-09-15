"""Deprecated alias. ``UNet2d`` was renamed :class:`~oceanml3d.models.ocean.nn.unet_nosc.UNetNosc`
when MONAI's ``DiffusionModelUNet`` became the default trunk; import it from there.
"""
from oceanml3d.models.ocean.nn.unet_nosc import UNetNosc as UNet2d  # noqa: F401

__all__ = ["UNet2d"]
