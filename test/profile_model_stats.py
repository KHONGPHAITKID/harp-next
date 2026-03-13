import argparse
import time

import torch
import yaml

from core.network import Network
import core.harpnext_core.backbone.harpnext_backbone as harpnext_backbone


class NormalSelfAttention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, **kwargs) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.qkv = torch.nn.Linear(dim, dim * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim)
        return out


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


def _build_batch_inputs(
    batch_size: int,
    height: int,
    width: int,
    in_channels: int,
    device: torch.device,
):
    coords = _make_dense_coords(batch_size, height, width, device)
    num_points = coords.shape[0]
    voxels = torch.randn(num_points, in_channels, device=device)
    return {"voxels": {"voxels": voxels, "coors": coords}}


def _resolve_netconfig(dataset: str | None, netconfig: str | None) -> tuple[str, str]:
    if netconfig:
        return netconfig, "(overridden by --netconfig)"
    if dataset is None:
        return "configs/net/harpnext-nuscenes.yaml", "nuscenes"
    dataset = dataset.lower()
    if dataset in ("nuscenes", "nu", "nusc"):
        return "configs/net/harpnext-nuscenes.yaml", "nuscenes"
    if dataset in ("semantic_kitti", "semantickitti", "kitti"):
        return "configs/net/harpnext-semantickitti.yaml", "semantic_kitti"
    if dataset in ("semantic_kitti_convmonarch", "semantickitti_convmonarch"):
        return "configs/net/harpnext-semantickitti-convmonarch.yaml", "semantic_kitti_convmonarch"
    raise ValueError(f"Unknown dataset {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--netconfig",
        default=None,
        help="Path to network config yaml (overrides --dataset).",
    )
    parser.add_argument(
        "--dataset",
        default="nuscenes",
        help="Dataset preset for config: nuscenes or semantic_kitti",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--attention",
        choices=["monarch", "normal"],
        default="monarch",
        help="Use Monarch attention (default) or patch to normal full attention.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    netconfig_path, preset_label = _resolve_netconfig(args.dataset, args.netconfig)

    with open(netconfig_path, "r") as f:
        netconfig = yaml.safe_load(f)

    model_cfg = netconfig["model"]
    backbone_cfg = model_cfg["backbone"]
    voxel_encoder_cfg = model_cfg["voxel_encoder"]

    if args.attention == "normal":
        # Monkey-patch ConvMonarchBlock to use full attention for profiling.
        harpnext_backbone.MonarchSelfAttention = NormalSelfAttention

    model = Network("harpnext", netconfig).build_network().to(device)
    model.eval()

    batch_inputs = _build_batch_inputs(
        batch_size=args.batch_size,
        height=backbone_cfg["output_shape"][0],
        width=backbone_cfg["output_shape"][1],
        in_channels=voxel_encoder_cfg["in_channels"],
        device=device,
    )

    # Params
    params = sum(p.numel() for p in model.parameters())

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(batch_inputs, training=False)
        _sync(device)

    # Time
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(batch_inputs, training=False)
        _sync(device)
        end = time.perf_counter()
    ms_per_iter = (end - start) / args.iters * 1000.0

    # Memory
    mem_gb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(batch_inputs, training=False)
        _sync(device)
        mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    # FLOPs and MACs (MACs ~= FLOPs / 2)
    flops = _profile_flops(lambda: model(batch_inputs, training=False), device, iters=1)
    macs = flops / 2.0

    to_giga = 1e9
    print(f"Device: {device.type}")
    print(f"Config: {netconfig_path}")
    print(f"Dataset preset: {preset_label}")
    print(f"Attention: {args.attention}")
    print(
        f"Batch: {args.batch_size} H={backbone_cfg['output_shape'][0]} W={backbone_cfg['output_shape'][1]}"
    )
    print(f"Params: {params / 1e6:.3f} M")
    print(f"Runtime: {ms_per_iter:.3f} ms/iter")
    print(f"FLOPs: {flops / to_giga:.3f} GFLOPs")
    print(f"MACs: {macs / to_giga:.3f} GMACs")
    if device.type == "cuda":
        print(f"Peak GPU memory: {mem_gb:.3f} GB")
    else:
        print("Peak GPU memory: N/A (CPU)")
    print("Note: profiler FLOPs may miss fused/sparse ops; MACs assume 2 FLOPs per MAC.")


if __name__ == "__main__":
    main()
