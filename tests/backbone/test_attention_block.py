import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_attention_block_no_downsample():
    """Block with stride=1 preserves spatial dims and channels."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=1,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 128, 16, 128)
    out = block(x)
    assert out.shape == (2, 128, 16, 128)


def test_attention_block_with_downsample():
    """Block with stride=2 halves spatial dims."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=2,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 128, 32, 256)
    out = block(x)
    assert out.shape == (2, 128, 16, 128)


def test_attention_block_channel_change():
    """Block handles inplanes != planes via downsample."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=64, planes=128, stride=1,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 64, 16, 128)
    out = block(x)
    assert out.shape == (2, 128, 16, 128)


def test_attention_block_gradient_flows():
    """Verify gradients flow through the block."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=1,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 128, 8, 64, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_attention_block_stage3_shapes():
    """Simulate stage 3: input 32x256 with stride=2 -> 16x128."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=2,
        num_heads=8, mlp_ratio=4.0, drop_path=0.1,
    )
    x = torch.randn(1, 128, 32, 256)
    out = block(x)
    assert out.shape == (1, 128, 16, 128)


def test_attention_block_stage4_shapes():
    """Simulate stage 4: input 16x128 with stride=2 -> 8x64."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=2,
        num_heads=8, mlp_ratio=4.0, drop_path=0.1,
    )
    x = torch.randn(1, 128, 16, 128)
    out = block(x)
    assert out.shape == (1, 128, 8, 64)
