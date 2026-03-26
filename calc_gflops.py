import argparse
import importlib.util
from typing import Dict, Tuple

import torch
import yaml
from torch import nn

from core.network import Network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate FLOPs/GFLOPs for HARP-NeXt models from a net config."
    )
    parser.add_argument(
        "--netconfig",
        type=str,
        required=True,
        help="Path to network config yaml (e.g. configs/net/harpnext-semantickitti-rangemamba.yaml).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size used to build dummy inputs.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=20000,
        help="Total synthetic points in the batch (across all samples).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for profiling.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["mamba", "identity"],
        help="Override range_mamba_cfg.backend. If omitted, config value is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic inputs.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Profile model(..., training=True). Default profiles inference path (training=False).",
    )
    return parser.parse_args()


def load_netconfig(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def maybe_patch_range_mamba_backend(cfg: Dict, backend_override: str | None) -> str | None:
    backbone_cfg = cfg.get("model", {}).get("backbone", {})
    range_cfg = backbone_cfg.get("range_mamba_cfg", None)
    if range_cfg is None:
        return None

    if backend_override is not None:
        range_cfg["backend"] = backend_override
        return f"range_mamba_cfg.backend forced to '{backend_override}'"

    backend = range_cfg.get("backend", "mamba")
    if backend == "mamba" and importlib.util.find_spec("mamba_ssm") is None:
        range_cfg["backend"] = "identity"
        return "mamba_ssm not found -> switched range_mamba_cfg.backend to 'identity'"
    return None


def build_dummy_inputs(
    cfg: Dict,
    batch_size: int,
    num_points: int,
    device: torch.device,
) -> Dict:
    voxel_encoder_cfg = cfg["model"]["voxel_encoder"]
    in_channels = int(voxel_encoder_cfg.get("in_channels", 4))
    out_h, out_w = cfg["model"]["backbone"]["output_shape"]

    voxels = torch.randn(num_points, in_channels, device=device, dtype=torch.float32)
    coors = torch.empty(num_points, 3, dtype=torch.long, device=device)
    coors[:, 0] = torch.randint(0, batch_size, (num_points,), device=device)
    coors[:, 1] = torch.randint(0, out_h, (num_points,), device=device)
    coors[:, 2] = torch.randint(0, out_w, (num_points,), device=device)
    # Backbone derives batch size from the last coordinate's batch index.
    coors[-1, 0] = batch_size - 1

    valid = (torch.rand(batch_size, 1, out_h, out_w, device=device) > 0.95).to(torch.float32)
    depth = torch.rand(batch_size, 1, out_h, out_w, device=device) * valid
    intensity = torch.rand(batch_size, 1, out_h, out_w, device=device) * valid

    net_inputs = {
        "voxels": {
            "voxels": voxels,
            "coors": coors,
            "range_aux": {"depth": depth, "intensity": intensity, "valid": valid},
        }
    }
    return net_inputs


def params_count(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def profile_flops_with_torch_profiler(
    model: nn.Module,
    inputs: Dict,
    device: torch.device,
    training: bool = False,
) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model.eval()
    with torch.no_grad():
        _ = model(inputs, training=training)

    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            _ = model(inputs, training=training)

    total_flops = 0
    for evt in prof.key_averages():
        evt_flops = getattr(evt, "flops", 0)
        if evt_flops:
            total_flops += int(evt_flops)
    return total_flops


def fallback_flops_from_hooks(
    model: nn.Module,
    inputs: Dict,
    training: bool = False,
    mul_add_as_two: bool = True,
) -> int:
    flops = 0
    factor = 2 if mul_add_as_two else 1
    hooks = []

    def conv_hook(module: nn.Module, module_inputs, module_output) -> None:
        nonlocal flops
        if not isinstance(module_output, torch.Tensor):
            return
        out = module_output
        batch = out.shape[0]
        out_channels = out.shape[1]
        out_spatial = out.shape[2:]
        out_elems = batch * out_channels
        for v in out_spatial:
            out_elems *= v

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel_elems = 1
            for k in module.kernel_size:
                kernel_elems *= k
            in_per_group = module.in_channels // module.groups
            macs_per_out = in_per_group * kernel_elems
            if module.bias is not None:
                macs_per_out += 1
            flops += int(out_elems * macs_per_out * factor)

    def linear_hook(module: nn.Module, module_inputs, module_output) -> None:
        nonlocal flops
        if not isinstance(module_output, torch.Tensor):
            return
        out = module_output
        out_elems = out.numel()
        macs_per_out = module.in_features + (1 if module.bias is not None else 0)
        flops += int(out_elems * macs_per_out * factor)

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        _ = model(inputs, training=training)

    for h in hooks:
        h.remove()

    return flops


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_points < args.batch_size:
        raise ValueError("--num-points must be >= --batch-size")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    cfg = load_netconfig(args.netconfig)
    backend_note = maybe_patch_range_mamba_backend(cfg, args.backend)

    model = Network(net=cfg["model"]["name"], netconfig=cfg).build_network().to(device)
    inputs = build_dummy_inputs(
        cfg=cfg,
        batch_size=args.batch_size,
        num_points=args.num_points,
        device=device,
    )

    flops = profile_flops_with_torch_profiler(
        model=model,
        inputs=inputs,
        device=device,
        training=args.training,
    )
    method = "torch.profiler(with_flops=True)"

    if flops <= 0:
        flops = fallback_flops_from_hooks(model=model, inputs=inputs, training=args.training)
        method = "forward-hook fallback (Conv/Linear only)"

    trainable_params, total_params = params_count(model)
    gflops = flops / 1e9
    gmacs = flops / 2e9

    print("==== HARP-NeXt FLOPs Report ====")
    print(f"Config file       : {args.netconfig}")
    print(f"Device            : {device}")
    print(f"Mode              : {'training=True' if args.training else 'training=False'}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Total points      : {args.num_points}")
    if backend_note:
        print(f"RangeMamba backend: {backend_note}")
    print(f"Params (trainable): {trainable_params:,}")
    print(f"Params (total)    : {total_params:,}")
    print(f"FLOPs             : {flops:,}")
    print(f"GFLOPs            : {gflops:.4f}")
    print(f"Approx GMACs      : {gmacs:.4f}")
    print(f"Count method      : {method}")
    print("Note: FLOPs depend on synthetic input size and supported ops in profiler/counter.")


if __name__ == "__main__":
    main()
