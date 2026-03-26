import importlib.util

import torch
import yaml

from core.network import Network


def main():
    cfg_path = "./configs/net/harpnext-semantickitti-rangemamba.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r"))

    # Allow running without mamba-ssm installed (smoke only).
    mamba_ok = importlib.util.find_spec("mamba_ssm") is not None
    if not mamba_ok:
        cfg["model"]["backbone"].setdefault("range_mamba_cfg", {})
        cfg["model"]["backbone"]["range_mamba_cfg"]["backend"] = "identity"
        print("NOTE: mamba_ssm not found; using backend='identity' for smoke forward.")

    model = Network(net="harpnext", netconfig=cfg).build_network()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    H, W = cfg["model"]["backbone"]["output_shape"]
    B = 2
    N = 20000

    voxels = torch.randn(N, 4, device=device)
    coors = torch.empty(N, 3, dtype=torch.long, device=device)
    coors[:, 0] = torch.randint(0, B, (N,), device=device)
    coors[:, 1] = torch.randint(0, H, (N,), device=device)
    coors[:, 2] = torch.randint(0, W, (N,), device=device)
    coors[-1, 0] = B - 1  # backbone derives batch_size from the last coord entry

    valid = (torch.rand(B, 1, H, W, device=device) > 0.95).to(torch.float32)
    depth = torch.rand(B, 1, H, W, device=device) * valid
    intensity = torch.rand(B, 1, H, W, device=device) * valid

    net_inputs = {
        "voxels": {
            "voxels": voxels,
            "coors": coors,
            "range_aux": {"depth": depth, "intensity": intensity, "valid": valid},
        }
    }

    with torch.no_grad():
        out = model(net_inputs, training=False)
    print("seg_logits:", tuple(out["seg_logits"].shape))


if __name__ == "__main__":
    main()

