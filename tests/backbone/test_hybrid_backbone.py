import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _make_voxel_dict(batch_size=2, num_points=500, num_voxels=200):
    """Create a minimal voxel_dict that mirrors FeaturesEncoder output."""
    voxel_feats = torch.randn(num_voxels, 16)

    # Coordinates: [batch_idx, y, x] with valid ranges for 64x512
    coors_list = []
    for b in range(batch_size):
        n = num_points // batch_size
        ys = torch.randint(0, 64, (n,))
        xs = torch.randint(0, 512, (n,))
        bs = torch.full((n,), b, dtype=torch.long)
        coors_list.append(torch.stack([bs, ys, xs], dim=1))
    coors = torch.cat(coors_list, dim=0)

    # Voxel coors: unique subset
    voxel_coors_list = []
    for b in range(batch_size):
        nv = num_voxels // batch_size
        ys = torch.randint(0, 64, (nv,))
        xs = torch.randint(0, 512, (nv,))
        bs = torch.full((nv,), b, dtype=torch.long)
        voxel_coors_list.append(torch.stack([bs, ys, xs], dim=1))
    voxel_coors = torch.cat(voxel_coors_list, dim=0)

    N = coors.shape[0]
    point_feats = [
        torch.randn(N, 64),
        torch.randn(N, 128),
        torch.randn(N, 256),
        torch.randn(N, 256),
    ]

    return {
        'voxel_feats': voxel_feats,
        'voxel_coors': voxel_coors,
        'coors': coors,
        'point_feats': point_feats,
    }


def test_hybrid_backbone_forward_shapes():
    """Hybrid backbone produces correct output shapes."""
    from core.harpnext_core.backbone.harpnext_backbone import HARPNeXtBackbone
    backbone = HARPNeXtBackbone(
        in_channels=16,
        point_in_channels=384,
        output_shape=[64, 512],
        depth=10,
        stem_channels=128,
        num_stages=4,
        out_channels=[128, 128, 128, 128],
        strides=[1, 2, 2, 2],
        dilations=[3, 3, 3, 3],
        fuse_channels=[256, 128],
        block_type="hybrid",
        block_cfg={
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop_path": 0.0,
            "attn_drop": 0.0,
        },
    )
    backbone = backbone.cpu()
    voxel_dict = _make_voxel_dict()
    result = backbone(voxel_dict)

    # voxel_feats[0] is fused: [B, 128, 64, 512]
    assert result['voxel_feats'][0].shape == (2, 128, 64, 512)
    # 5 entries: fused + stem + 4 stages
    assert len(result['voxel_feats']) == 5
    # point_feats_backbone[0] is fused: [N, 128]
    assert result['point_feats_backbone'][0].shape[1] == 128


def test_hybrid_backbone_uses_both_block_types():
    """Stages 1-2 use ConvSENeXt, stages 3-4 use AttentionBlock."""
    from core.harpnext_core.backbone.harpnext_backbone import HARPNeXtBackbone
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock

    backbone = HARPNeXtBackbone(
        in_channels=16,
        point_in_channels=384,
        output_shape=[64, 512],
        depth=10,
        block_type="hybrid",
        block_cfg={"num_heads": 8, "mlp_ratio": 4.0, "drop_path": 0.0},
    )

    # Check block types by inspecting the res_layers
    layer1 = getattr(backbone, 'layer1')  # stage 1 -> ConvSENeXt
    layer2 = getattr(backbone, 'layer2')  # stage 2 -> ConvSENeXt
    layer3 = getattr(backbone, 'layer3')  # stage 3 -> AttentionBlock
    layer4 = getattr(backbone, 'layer4')  # stage 4 -> AttentionBlock

    assert not isinstance(layer1[0], HARPNeXtAttentionBlock)
    assert not isinstance(layer2[0], HARPNeXtAttentionBlock)
    assert isinstance(layer3[0], HARPNeXtAttentionBlock)
    assert isinstance(layer4[0], HARPNeXtAttentionBlock)
