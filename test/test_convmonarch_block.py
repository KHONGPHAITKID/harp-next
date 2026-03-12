import torch
import torch.nn as nn

from core.harpnext_core.backbone.harpnext_backbone import ConvMonarchBlock


def test_convmonarch_block_forward_shape_same_channels():
    torch.manual_seed(0)
    block = ConvMonarchBlock(
        inplanes=8,
        planes=8,
        stride=1,
        dw_conv_kernel=3,
        dw_conv_bias=False,
        attn_heads=2,
        attn_block_size=4,
        attn_num_steps=2,
    )
    x = torch.randn(2, 8, 4, 4)
    y = block(x)
    assert y.shape == x.shape


def test_convmonarch_block_forward_shape_downsample():
    torch.manual_seed(0)
    downsample = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(16),
    )
    block = ConvMonarchBlock(
        inplanes=8,
        planes=16,
        stride=2,
        downsample=downsample,
        dw_conv_kernel=3,
        dw_conv_bias=False,
        attn_heads=4,
        attn_block_size=4,
        attn_num_steps=2,
    )
    x = torch.randn(2, 8, 8, 8)
    y = block(x)
    assert y.shape == (2, 16, 4, 4)


if __name__ == "__main__":
    test_convmonarch_block_forward_shape_same_channels()
    test_convmonarch_block_forward_shape_downsample()
    print("OK")
