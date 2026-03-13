import time

import torch

from core.harpnext_core.backbone.harpnext_backbone import ConvMonarchBlock, ConvSENeXt


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_block(block, x, iters: int) -> float:
    device = x.device
    block.eval()
    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = block(x)
        _sync(device)

        start = time.perf_counter()
        for _ in range(iters):
            _ = block(x)
        _sync(device)
        end = time.perf_counter()

    return (end - start) / iters * 1000.0


def benchmark(
    batch_size: int = 2,
    channels: int = 128,
    height: int = 64,
    width: int = 512,
    iters: int = 50,
    device: str | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x = torch.randn(batch_size, channels, height, width, device=device)
    dilation = 3

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

    t_convsenext = _time_block(convsenext, x, iters)
    t_convmonarch = _time_block(convmonarch, x, iters)

    print(f"Device: {device.type}")
    print(f"Input: B={batch_size} C={channels} H={height} W={width}")
    print(f"Iters: {iters}")
    print(f"Dilation: {dilation}")
    print(f"ConvSENeXt:  {t_convsenext:.3f} ms/iter")
    print(f"ConvMonarch: {t_convmonarch:.3f} ms/iter")


if __name__ == "__main__":
    benchmark()
