import argparse
import math
import time

import torch

from core.harpnext_core.backbone.harpnext_backbone import MonarchSelfAttention


class NormalSelfAttention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
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


def _time_module(module, x, iters: int, warmup: int) -> float:
    device = x.device
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        _sync(device)

        start = time.perf_counter()
        for _ in range(iters):
            _ = module(x)
        _sync(device)
        end = time.perf_counter()

    return (end - start) / iters * 1000.0


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    seq_len = args.height * args.width
    if seq_len > 8192 and not args.force:
        raise SystemExit(
            f"Refusing to run full attention with N={seq_len} tokens. "
            "Use smaller H/W or pass --force to attempt it."
        )

    torch.manual_seed(0)
    x = torch.randn(
        args.batch_size, seq_len, args.channels, device=device
    )

    normal_attn = NormalSelfAttention(args.channels, num_heads=args.heads).to(device)
    monarch_attn = MonarchSelfAttention(
        args.channels,
        num_heads=args.heads,
        block_size=16,
        num_steps=2,
    ).to(device)

    t_normal = _time_module(normal_attn, x, args.iters, args.warmup)
    t_monarch = _time_module(monarch_attn, x, args.iters, args.warmup)

    flops_normal = _profile_flops(lambda: normal_attn(x), device, iters=1)
    flops_monarch = _profile_flops(lambda: monarch_attn(x), device, iters=1)

    to_gflops = 1e9
    print(f"Device: {device.type}")
    print(f"Input: B={args.batch_size} N={seq_len} C={args.channels} H={args.height} W={args.width}")
    print(f"Normal Attention:  {t_normal:.3f} ms/iter, {flops_normal / to_gflops:.3f} GFLOPs")
    print(f"Monarch Attention: {t_monarch:.3f} ms/iter, {flops_monarch / to_gflops:.3f} GFLOPs")
    print("Note: profiler FLOPs may miss fused kernels or custom ops.")


if __name__ == "__main__":
    main()
