import argparse
import time

import torch
import yaml

from core.harpnext_core.backbone.harpnext_backbone import (
    ConvMonarchBlock,
    ConvSENeXt,
    HARPNeXtBackbone,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_flops(fn, device: torch.device, iters: int = 1) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(iters):
            fn()
            _sync(device)

    total = 0
    for event in prof.key_averages():
        if event.flops is not None:
            total += event.flops
    return total


def _make_dense_coords(batch_size: int, height: int, width: int, device: torch.device):
    y = torch.arange(height, device=device).repeat_interleave(width)
    x = torch.arange(width, device=device).repeat(height)
    coords = []
    for b in range(batch_size):
        bcol = torch.full_like(y, b)
        coords.append(torch.stack((bcol, y, x), dim=1))
    return torch.cat(coords, dim=0).to(torch.int32)


def _build_voxel_dict(
    batch_size: int,
    height: int,
    width: int,
    in_channels: int,
    point_feat_channels: int,
    device: torch.device,
):
    coords = _make_dense_coords(batch_size, height, width, device)
    num_points = coords.shape[0]
    voxel_feats = torch.randn(num_points, in_channels, device=device)
    point_feats = torch.randn(num_points, point_feat_channels, device=device)

    return {
        "point_feats": [point_feats],
        "voxel_feats": voxel_feats,
        "voxel_coors": coords,
        "coors": coords,
    }


def profile_blocks(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    dilation: int,
    device: torch.device,
):
    x = torch.randn(batch_size, channels, height, width, device=device)

    convsenext = ConvSENeXt(
        inplanes=channels,
        planes=channels,
        stride=1,
        dilation=dilation,
        dw_conv_kernel=7,
        dw_conv_bias=True,
    ).to(device)

    convmonarch = ConvMonarchBlock(
        inplanes=channels,
        planes=channels,
        stride=1,
        dilation=dilation,
        mlp_ratio=4,
        attn_heads=4,
        attn_block_size=16,
        attn_num_steps=2,
        dw_conv_kernel=7,
        dw_conv_bias=True,
    ).to(device)

    convsenext.eval()
    convmonarch.eval()

    with torch.no_grad():
        _ = convsenext(x)
        _ = convmonarch(x)
        _sync(device)

        flops_se = _profile_flops(lambda: convsenext(x), device, iters=1)
        flops_mon = _profile_flops(lambda: convmonarch(x), device, iters=1)

    return flops_se, flops_mon


def profile_backbone(
    netconfig_path: str,
    device: torch.device,
):
    with open(netconfig_path, "r") as f:
        netconfig = yaml.safe_load(f)

    backbone_cfg = netconfig["model"]["backbone"]
    backbone = HARPNeXtBackbone(**backbone_cfg).to(device)
    backbone.eval()

    stem_channels = backbone_cfg["stem_channels"]
    point_in_channels = backbone_cfg["point_in_channels"]
    point_feat_channels = point_in_channels - stem_channels
    if point_feat_channels <= 0:
        raise ValueError(
            "point_in_channels must be larger than stem_channels for mock data."
        )

    def build_voxel_dict():
        return _build_voxel_dict(
            batch_size=2,
            height=backbone_cfg["output_shape"][0],
            width=backbone_cfg["output_shape"][1],
            in_channels=backbone_cfg["in_channels"],
            point_feat_channels=point_feat_channels,
            device=device,
        )

    with torch.no_grad():
        _ = backbone(build_voxel_dict())
        _sync(device)
        flops = _profile_flops(lambda: backbone(build_voxel_dict()), device, iters=1)

    return flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument(
        "--netconfig-conv",
        default="configs/net/harpnext-semantickitti.yaml",
    )
    parser.add_argument(
        "--netconfig-mon",
        default="configs/net/harpnext-semantickitti-convmonarch.yaml",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    flops_se, flops_mon = profile_blocks(
        batch_size=args.batch_size,
        channels=args.channels,
        height=args.height,
        width=args.width,
        dilation=args.dilation,
        device=device,
    )

    flops_backbone_se = profile_backbone(args.netconfig_conv, device=device)
    flops_backbone_mon = profile_backbone(args.netconfig_mon, device=device)

    to_gflops = 1e9
    print(f"Device: {device.type}")
    print(f"Input: B={args.batch_size} C={args.channels} H={args.height} W={args.width}")
    print(f"Block ConvSENeXt:  {flops_se / to_gflops:.3f} GFLOPs")
    print(f"Block ConvMonarch: {flops_mon / to_gflops:.3f} GFLOPs")
    print(f"Backbone ConvSENeXt:  {flops_backbone_se / to_gflops:.3f} GFLOPs")
    print(f"Backbone ConvMonarch: {flops_backbone_mon / to_gflops:.3f} GFLOPs")
    print("Note: profiler FLOPs include conv/linear/matmul ops and may miss scatter/sparse ops.")


if __name__ == "__main__":
    main()
