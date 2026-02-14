# Copyright 2025 CEA LIST - Samir Abou Haidar
# Modifications based on code from Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai

# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
import tempfile
import yaml
import torch
import torch.nn as nn
import random
import warnings
import argparse
import numpy as np
try:
    from torch.serialization import add_safe_globals as _torch_add_safe_globals
except ImportError:
    _torch_add_safe_globals = None
import utils.transformations.transforms as tr
from utils.metrics.semanticsegmentation import SemSegLoss
from trainer.scheduler import WarmupCosine
from trainer.manager import Manager
from core.network import Network
from datasets import LIST_DATASETS, Collate_fn

from utils.loss.lovasz import Lovasz_softmax
from utils.loss.boundary_loss import BoundaryLoss
from torch.nn import CrossEntropyLoss


def _allow_numpy_safe_globals():
    """Allowlist numpy scalar implementations used in checkpoints (PyTorch >=2.6)."""
    if _torch_add_safe_globals is None:
        return

    scalar_types = []
    for attr in ("core", "_core"):
        numpy_core = getattr(np, attr, None)
        if numpy_core is None:
            continue
        multiarray = getattr(numpy_core, "multiarray", None)
        scalar = getattr(multiarray, "scalar", None)
        if scalar is not None:
            scalar_types.append(scalar)

    # Avoid calling torch multiple times if nothing was found.
    if scalar_types:
        _torch_add_safe_globals(scalar_types)


def _set_local_tmpdir():
    """Force temp files into a local ./tmp directory to avoid /tmp exhaustion."""
    tmp_dir = os.path.abspath("./tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    tempfile.tempdir = tmp_dir


def load_configs(mainfile, netfile):
    with open(mainfile, "r") as mf:
        mainconfig = yaml.safe_load(mf)

    with open(netfile, "r") as nf:
        netconfig = yaml.safe_load(nf)

    return mainconfig, netconfig


def get_train_augmentations(args, mainconfig, netconfig):

    list_of_transf = []

    # Optional augmentations
    for aug_name in netconfig["augmentations"].keys():
        if aug_name == "pointsample":
            list_of_transf.append(tr.PointSample(inplace=True, num_points = netconfig["augmentations"]["pointsample"]))
        elif aug_name == "randomflip":
            list_of_transf.append(tr.RandomFlip3D(inplace=True, 
                                                    sync_2d = netconfig["augmentations"]["randomflip"]["sync_2d"],
                                                    flip_ratio_bev_horizontal = netconfig["augmentations"]["randomflip"]["flip_ratio_bev_horizontal"],
                                                    flip_ratio_bev_vertical = netconfig["augmentations"]["randomflip"]["flip_ratio_bev_vertical"]))
        elif aug_name == "GlobalRotScaleTrans":
            rot_range = netconfig["augmentations"]["GlobalRotScaleTrans"]["rot_range"]
            scale_ratio_range = netconfig["augmentations"]["GlobalRotScaleTrans"]["scale_ratio_range"]
            translation_std = netconfig["augmentations"]["GlobalRotScaleTrans"]["translation_std"]
            list_of_transf.append(tr.GlobalRotScaleTrans(inplace=True, rot_range=rot_range, scale_ratio_range=scale_ratio_range, translation_std=translation_std))
        else:
            raise ValueError("Unknown transformation")
    print("List of transformations:", list_of_transf)
    return tr.Compose(list_of_transf)



def get_datasets(netconfig, args):
    kwargs = {
        "dataset": args.dataset,
        "rootdir": args.path_dataset,
        "input_feat": netconfig["input_feat"],
        "range_H": netconfig["range_proj"]["range_H"],
        "range_W": netconfig["range_proj"]["range_W"],
        "fov_up": netconfig["range_proj"]["fov_up"],
        "fov_down": netconfig["range_proj"]["fov_down"],
        "batch_size": mainconfig["dataloader"]["batch_size"],
        "preproc_gpu": netconfig["preproc"]["gpu"],
        "rank": args.gpu
    }

    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")
    
    # Train dataset
    train_dataset = DATASET(
        phase="trainval" if args.trainval else "train",
        train_augmentations=get_train_augmentations(args, mainconfig, netconfig),
        instance_cutmix=mainconfig["augmentations"]["instance_cutmix"],
        **kwargs,
    )

    # Validation dataset
    val_dataset = DATASET(
        phase="val",
        **kwargs,
    )

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args, mainconfig):

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    if Collate_fn is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=Collate_fn(),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
            collate_fn=Collate_fn())
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
        )

    return train_loader, val_loader, train_sampler


def get_test_dataset(netconfig, args, mainconfig):
    kwargs = {
        "dataset": args.dataset,
        "rootdir": args.path_dataset,
        "input_feat": netconfig["input_feat"],
        "range_H": netconfig["range_proj"]["range_H"],
        "range_W": netconfig["range_proj"]["range_W"],
        "fov_up": netconfig["range_proj"]["fov_up"],
        "fov_down": netconfig["range_proj"]["fov_down"],
        "batch_size": 1,
        "preproc_gpu": False,
        "rank": args.gpu,
    }

    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    test_dataset = DATASET(
        phase="test",
        train_augmentations=None,
        instance_cutmix=False,
        **kwargs,
    )

    return test_dataset


def get_test_dataloader(test_dataset, args, mainconfig):
    if Collate_fn is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=Collate_fn(),
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    return test_loader


def load_checkpoint_for_inference(model, checkpoint_path, gpu):
    if checkpoint_path is None:
        return
    if os.path.isdir(checkpoint_path):
        ckpt_best = os.path.join(checkpoint_path, "ckpt_best.pth")
        ckpt_last = os.path.join(checkpoint_path, "ckpt_last.pth")
        if os.path.exists(ckpt_best):
            checkpoint_path = ckpt_best
        elif os.path.exists(ckpt_last):
            checkpoint_path = ckpt_last
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    map_location = f"cuda:{gpu}" if gpu is not None else "cpu"
    _allow_numpy_safe_globals()
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("net", ckpt)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        fixed = {}
        for key, val in state_dict.items():
            if key.startswith("module."):
                fixed[key[len("module."):]] = val
            else:
                fixed[key] = val
        model.load_state_dict(fixed)


def run_semantickitti_test(args, mainconfig, netconfig):
    if args.dataset.lower() != "semantic_kitti":
        raise ValueError("Test dumping is only implemented for semantic_kitti.")

    args.batch_size = 1
    args.workers = mainconfig["dataloader"]["num_workers"]

    net = Network(args.net, netconfig)
    model = net.build_network()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        device = args.gpu
    else:
        model = model.cuda()
        device = None
    load_checkpoint_for_inference(model, args.checkpoint, args.gpu)

    test_dataset = get_test_dataset(netconfig, args, mainconfig)
    test_loader = get_test_dataloader(test_dataset, args, mainconfig)
    test_root = args.test_output or os.path.join(args.log_path, "test_pred")
    os.makedirs(test_root, exist_ok=True)

    learning_map_inv = test_dataset.learning_map_inv
    max_key = max(learning_map_inv.keys())
    lut = np.zeros(max_key + 1, dtype=np.uint32)
    for key, value in learning_map_inv.items():
        lut[int(key)] = np.uint32(value)

    model.eval()
    with torch.no_grad():
        for it, _ in enumerate(test_loader):
            batch, _ = test_dataset.process_batch_cpu(it)
            net_inputs = dict()
            net_inputs["points"] = [
                pt.cuda(device, non_blocking=True) for pt in batch["points"]
            ]
            voxel_dict = dict()
            voxel_dict["voxels"] = batch["voxels"].cuda(device, non_blocking=True)
            voxel_dict["coors"] = batch["coors"].cuda(device, non_blocking=True)
            net_inputs["voxels"] = voxel_dict

            with torch.autocast("cuda", enabled=args.fp16):
                out = model(net_inputs, training=False)
                seg_logits = out["seg_logits"]
                pred = seg_logits.max(1)[1].detach().cpu().numpy().astype(np.int32)

            orig_len = int(batch["orig_len"][0])
            pred = pred[:orig_len]
            # Training labels are shifted by -1 (learning labels 1..19 become 0..18).
            # Shift back before applying learning_map_inv to export official SemanticKITTI IDs.
            pred_mapped = lut[pred + 1]

            scan_path = batch["filename"][0]
            seq = os.path.basename(os.path.dirname(os.path.dirname(scan_path)))
            scan_name = os.path.basename(scan_path).replace(".bin", ".label")
            out_dir = os.path.join(test_root, "sequences", seq, "predictions")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, scan_name)

            pred_label = pred_mapped.astype(np.uint32)
            pred_label.tofile(out_path)


def get_optimizer(parameters, mainconfig):
    return torch.optim.AdamW(
        parameters,
        lr=mainconfig["optim"]["lr"],
        weight_decay=mainconfig["optim"]["weight_decay"],
        betas=mainconfig["optim"]["betas"],
        eps = mainconfig["optim"]["eps"],
    )


def get_scheduler(optimizer, mainconfig, len_train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            mainconfig["scheduler"]["epoch_warmup"] * len_train_loader,
            mainconfig["scheduler"]["max_epoch"] * len_train_loader,
            mainconfig["scheduler"]["min_lr"] / mainconfig["optim"]["lr"],
        ),
    )
    return scheduler


def distributed_training(gpu, ngpus_per_node, args, mainconfig, netconfig):

    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    net = Network(args.net, netconfig)
    model = net.build_network()
    
    # ---
    args.batch_size = mainconfig["dataloader"]["batch_size"]
    args.workers = mainconfig["dataloader"]["num_workers"]
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(mainconfig["dataloader"]["batch_size"] / ngpus_per_node)
        args.workers = int(
            (mainconfig["dataloader"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.gpu == 0 or args.gpu is not None:
        #print(f"Model:\n{model}")
        nb_param = sum([p.numel() for p in model.parameters()]) / 1e6
        print(f"{nb_param} x 10^6 trainable parameters ")

    # --- Optimizer
    optim = get_optimizer(model.parameters(), mainconfig)

    # --- Dataset
    train_dataset, val_dataset = get_datasets(netconfig, args)
    train_loader, val_loader, train_sampler = get_dataloader(
        train_dataset, val_dataset, args, mainconfig
    )

    # --- Loss function
    lovasz = Lovasz_softmax(ignore=netconfig["classif"]["ignore_class"]).cuda(args.gpu)
    bd = BoundaryLoss(ignore_index=netconfig["classif"]["ignore_class"]).cuda(args.gpu)
    ce = CrossEntropyLoss(ignore_index=netconfig["classif"]["ignore_class"]).cuda(args.gpu)
    loss = {
        "lovasz": lovasz,
        "bd": bd,
        "ce": ce
    }

    if(args.eval is False):
        scheduler = get_scheduler(optim, mainconfig, len(train_loader))
    else:
        scheduler = None

    # --- Training
    mng = Manager(
        model,
        loss,
        train_loader,
        val_loader,
        train_sampler,
        optim,
        scheduler,
        mainconfig["scheduler"]["max_epoch"],
        args.log_path,
        args.gpu,
        args.world_size,
        args.fp16,
        args.net,
        LIST_DATASETS.get(args.dataset.lower()).CLASS_NAME,
        tensorboard=(not args.eval),
        checkpoint= args.checkpoint,
        netconfig=netconfig,
        preproc_gpu=netconfig["preproc"]["gpu"],
        perf=args.perf,
        mlflow_cfg=mainconfig.get("mlflow", {}),
        args=args,
        mainconfig=mainconfig,

    )
    if args.restart:
        mng.load_state(best=True) #True
    if args.eval:
        mng.one_epoch(training=False)
        mng._mlflow_log_epoch("val", mng.current_epoch)
        mng.close_mlflow()
    else:
        mng.train()
        mng.close_mlflow()


def main(args, mainconfig, netconfig):

    _set_local_tmpdir()
    args.device = "cuda"
    args.rank = 0
    args.world_size = 1
    args.dist_url = "to-specify"
    args.dist_backend = "nccl"
    args.distributed = args.multiprocessing_distributed

    os.makedirs(args.log_path, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    if args.gpu is not None:
        args.distributed = False
        args.multiprocessing_distributed = False
        warnings.warn(
            "You chose a specific GPU. Data parallelism is disabled."
        )

    if args.test:
        run_semantickitti_test(args, mainconfig, netconfig)
        return

    # Extract instances for cutmix
    if mainconfig["augmentations"]["instance_cutmix"]:
        get_datasets(netconfig, args)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(
            distributed_training,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, mainconfig, netconfig),
        )
    else:
        distributed_training(args.gpu, ngpus_per_node, args, mainconfig, netconfig)


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--net",
        type=str,
        help="Network name (harpnext)",
        default="harpnext"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of dataset",
        default="semantic_kitti",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="../dataset/SemanticKitti/data_odometry_velodyne",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=False,
        default="./logs/harpnext-semantickitti",
        help="Path to log folder",
    )
    parser.add_argument(
        "-r", "--restart", action="store_true", default=False, help="Restart training"
    )
    parser.add_argument(
        "--seed", default=219, type=int, help="Seed for initializing training"
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="Set to any number to use gpu 0"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--mainconfig",
        type=str, 
        required=False,
        default="configs/main/main-config.yaml",
        help="Path to main config"
    )
    parser.add_argument(
        "--netconfig",
        type=str, 
        required=False,
        default="configs/net/harpnext-semantickitti.yaml",
        help="Path to specific network model config"
    )
    parser.add_argument(
        "--trainval",
        action="store_true",
        default=False,
        help="Use train + val as train set",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run validation only",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        default=False,
        help="To run in Performance Mode, ensure a batch size of 1",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default="./logs/harpnext-cutmix-semantickitti-64x512-retrain",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run test set inference and dump SemanticKITTI labels",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        required=False,
        default=None,
        help="Output root for test label dumps (default: <log_path>/test_pred)",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    mainconfig, netconfig = load_configs(args.mainconfig, args.netconfig)
    main(args, mainconfig, netconfig)
