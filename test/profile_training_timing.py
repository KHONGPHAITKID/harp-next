import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from main import load_configs, get_train_augmentations
from core.network import Network
from datasets import LIST_DATASETS, Collate_fn
from utils.loss.lovasz import Lovasz_softmax
from utils.loss.boundary_loss import BoundaryLoss
from torch.nn import CrossEntropyLoss
from utils.metrics.semanticsegmentation import fast_hist


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingStats:
    def __init__(self):
        self.values = defaultdict(list)

    def add(self, key, value):
        self.values[key].append(value)

    def summary(self):
        rows = []
        for key, vals in self.values.items():
            arr = np.array(vals, dtype=np.float64)
            rows.append(
                (key, arr.mean(), np.percentile(arr, 50), np.percentile(arr, 90), arr.max())
            )
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows

    def total_iter_time(self):
        vals = self.values.get("iter_total_ms", [])
        if not vals:
            return 0.0
        return float(np.mean(vals))


class TimingRunner:
    def __init__(self, model, loss, netconfig, device, preproc_gpu=False):
        self.model = model
        self.loss = loss
        self.netconfig = netconfig
        self.device = device
        self.preproc_gpu = preproc_gpu

    def get_network_inputs(self, batch):
        if self.preproc_gpu:
            net_inputs = {
                "points": batch["points"],
                "voxels": {"voxels": batch["voxels"], "coors": batch["coors"]},
            }
            return net_inputs

        net_inputs = {
            "points": [pt.cuda(self.device, non_blocking=True) for pt in batch["points"]],
            "voxels": {
                "voxels": batch["voxels"].cuda(self.device, non_blocking=True),
                "coors": batch["coors"].cuda(self.device, non_blocking=True),
            },
        }
        return net_inputs

    def get_labels(self, batch):
        if self.preproc_gpu:
            proj_range_sem_label = batch["proj_labels"].long().cuda(
                self.device, non_blocking=True
            )
            pt_sem_label = batch["pt_labels"].long().cuda(
                self.device, non_blocking=True
            )
        else:
            proj_range_sem_label = torch.stack(batch["proj_labels"], dim=0).long().cuda(
                self.device, non_blocking=True
            )
            pt_sem_label = torch.cat(batch["pt_labels"], dim=0).long().cuda(
                self.device, non_blocking=True
            )
        return {"proj_labels": proj_range_sem_label, "pt_labels": pt_sem_label}

    def get_predictions(self, confusion_matrix, labels, out):
        with torch.no_grad():
            nb_class = out.shape[1]
            pred_label = out.max(1)[1]
            where = labels != self.netconfig["classif"]["ignore_class"]
            confusion_matrix += fast_hist(pred_label[where], labels[where], nb_class)
        return confusion_matrix


def build_dataloaders(args, mainconfig, netconfig, disable_augs=False, disable_cutmix=False):
    kwargs = {
        "dataset": args.dataset,
        "rootdir": args.path_dataset,
        "input_feat": netconfig["input_feat"],
        "range_H": netconfig["range_proj"]["range_H"],
        "range_W": netconfig["range_proj"]["range_W"],
        "fov_up": netconfig["range_proj"]["fov_up"],
        "fov_down": netconfig["range_proj"]["fov_down"],
        "batch_size": args.batch_size,
        "preproc_gpu": netconfig["preproc"]["gpu"],
        "rank": args.gpu,
    }

    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    if disable_augs:
        train_augs = None
    else:
        train_augs = get_train_augmentations(args, mainconfig, netconfig)

    instance_cutmix = mainconfig["augmentations"]["instance_cutmix"]
    if disable_cutmix:
        instance_cutmix = False

    train_dataset = DATASET(
        phase="train",
        train_augmentations=train_augs,
        instance_cutmix=instance_cutmix,
        **kwargs,
    )
    val_dataset = DATASET(phase="val", **kwargs)

    if Collate_fn is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=Collate_fn(),
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=Collate_fn(),
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    return train_loader, val_loader


def build_loss(netconfig, gpu):
    ignore = netconfig["classif"]["ignore_class"]
    lovasz = Lovasz_softmax(ignore=ignore).cuda(gpu)
    bd = BoundaryLoss(ignore_index=ignore).cuda(gpu)
    ce = CrossEntropyLoss(ignore_index=ignore).cuda(gpu)
    return {"lovasz": lovasz, "bd": bd, "ce": ce}


def main():
    parser = argparse.ArgumentParser("Profile training timing")
    parser.add_argument("--mainconfig", type=str, default="configs/main/main-config.yaml")
    parser.add_argument("--netconfig", type=str, default="configs/net/harpnext-semantickitti.yaml")
    parser.add_argument("--dataset", type=str, default="semantic_kitti")
    parser.add_argument("--path_dataset", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_backward", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--no_cutmix", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    mainconfig, netconfig = load_configs(args.mainconfig, args.netconfig)

    torch.cuda.set_device(args.gpu)
    device = args.gpu

    net = Network("harpnext", netconfig)
    model = net.build_network().cuda(device)
    model.train() if args.mode == "train" else model.eval()

    loss = build_loss(netconfig, args.gpu)

    train_loader, val_loader = build_dataloaders(
        args, mainconfig, netconfig, disable_augs=args.no_aug, disable_cutmix=args.no_cutmix
    )
    loader = train_loader if args.mode == "train" else val_loader
    dataset = loader.dataset

    runner = TimingRunner(model, loss, netconfig, device, preproc_gpu=netconfig["preproc"]["gpu"])

    stats = TimingStats()

    it = iter(loader)
    total_iters = args.warmup + args.iters
    for i in range(total_iters):
        t_iter_start = time.perf_counter()

        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t1 = time.perf_counter()
        stats.add("dataloader_ms", (t1 - t0) * 1000.0)

        idx = i % len(dataset)
        t0 = time.perf_counter()
        if runner.preproc_gpu:
            pc, labels = dataset.load_batch_to_gpu(idx)
            batch, _ = dataset.process_batch_gpu(pc, labels)
        elif args.mode == "val":
            batch, _ = dataset.process_batch_cpu(idx)
        t1 = time.perf_counter()
        stats.add("preproc_ms", (t1 - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        net_inputs = runner.get_network_inputs(batch)
        labels = runner.get_labels(batch)
        _sync()
        t1 = time.perf_counter()
        stats.add("h2d_ms", (t1 - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        with torch.autocast("cuda", enabled=args.fp16):
            if args.mode == "train":
                out = model(net_inputs, training=True)
            else:
                with torch.no_grad():
                    out = model(net_inputs, training=False)
        _sync()
        t1 = time.perf_counter()
        stats.add("forward_ms", (t1 - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        out_losses = out["losses_seg_logits"]
        if args.mode == "train":
            lamda = netconfig["train"]["lamda"]
            loss_points = loss["ce"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])
            loss_aux_0 = loss["ce"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"]) + 1.5 * loss["lovasz"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"]) + loss["bd"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"])
            loss_aux_1 = loss["ce"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"]) + 1.5 * loss["lovasz"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"]) + loss["bd"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"])
            loss_aux_2 = loss["ce"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"]) + 1.5 * loss["lovasz"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"]) + loss["bd"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"])
            loss_aux_3 = loss["ce"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"]) + 1.5 * loss["lovasz"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"]) + loss["bd"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"])
            loss_val = loss_points + lamda * (loss_aux_0 + loss_aux_1 + loss_aux_2 + loss_aux_3)
        else:
            loss_val = loss["ce"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])
        _sync()
        t1 = time.perf_counter()
        stats.add("loss_ms", (t1 - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        if args.mode == "train" and not args.no_backward:
            loss_val.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
        _sync()
        t1 = time.perf_counter()
        stats.add("backward_ms", (t1 - t0) * 1000.0)

        _sync()
        t0 = time.perf_counter()
        logits = out["seg_logits"]
        _ = runner.get_predictions(
            torch.zeros(
                (netconfig["classif"]["nb_class"], netconfig["classif"]["nb_class"]),
                device=logits.device,
                dtype=torch.int64,
            ),
            labels["pt_labels"],
            logits,
        )
        _sync()
        t1 = time.perf_counter()
        stats.add("metrics_ms", (t1 - t0) * 1000.0)

        t_iter_end = time.perf_counter()
        stats.add("iter_total_ms", (t_iter_end - t_iter_start) * 1000.0)

        if i < args.warmup:
            continue

    print("\nTiming summary (ms):")
    print(f"  iterations (measured): {args.iters}")
    print(f"  mode: {args.mode} | fp16: {args.fp16} | preproc_gpu: {runner.preproc_gpu}")
    print(f"  no_aug: {args.no_aug} | no_cutmix: {args.no_cutmix}")
    print("")
    print("  key                 mean     p50     p90     max")
    for key, mean, p50, p90, vmax in stats.summary():
        if key == "iter_total_ms":
            continue
        print(f"  {key:<18} {mean:7.2f} {p50:7.2f} {p90:7.2f} {vmax:7.2f}")
    print("")
    print(f"  iter_total_ms mean: {stats.total_iter_time():.2f}")


if __name__ == "__main__":
    main()
