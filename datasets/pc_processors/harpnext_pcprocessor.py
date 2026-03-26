# Copyright 2025 CEA LIST - Samir Abou Haidar

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import utils.transformations.transforms as tr
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

from core.harpnext_core.preprocessing.laserscan import SemLaserScan
from core.harpnext_core.preprocessing.laserscan_cuda import SemLaserScanCUDA
from typing import Any, List

from utils.transformations.transforms import ClusterMix, InstanceCopy, RangeInterpolation, RangeInterpolationCUDA

import yaml


class HARPNeXtPCProcessor(Dataset):
    def __init__(
        self,
        dataset=None,
        rootdir=None,
        input_feat="none",
        phase="train",
        train_augmentations=None,
        tta=False,
        instance_cutmix=False,
        range_H=64,
        range_W=1024,
        fov_up= 3.0,
        fov_down= -25.0,
        batch_size = None,
        preproc_gpu = False,
        rank = None

    ):
        super().__init__()

        # Dataset split
        self.phase = phase
        assert self.phase in ["train", "val", "trainval", "test"]

        # Dataset
        self.dataset = dataset
        self.rootdir = rootdir

        self.range_H = range_H
        self.range_W = range_W
        self.fov_up = fov_up
        self.fov_down = fov_down

        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        if self.dataset == "semantic_kitti":
            config_dataset = os.path.join(parent_dir, "semantickitti/semantic-kitti.yaml")
            self.W_interpolation = 2048
            self.instances_classes = [2, 3, 4, 5, 6, 7, 8, 12, 16, 18, 19]
        elif self.dataset == "nuscenes":
            config_dataset = os.path.join(parent_dir, "nuscenes/nuscenes.yaml")
            self.W_interpolation = 1920
            self.instances_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            raise Exception(f"Dataset {self.dataset} is not supported yet.")
        
        # open config file of dataset
        try:
            # print("Opening config file %s" % config_dataset)
            self.config_file = yaml.safe_load(open(config_dataset, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()

        # Get color dictionary of the dataset
        self.color_dict = self.config_file["color_map"]
        self.learning_map_inv = self.config_file["learning_map_inv"]
        self.learning_map = self.config_file["learning_map"]
        self.color_dict = {key:self.color_dict[self.learning_map_inv[self.learning_map[key]]] for key, value in self.color_dict.items()}

        # Input features to compute for each point
        self.input_feat = input_feat

        # Test time augmentation
        if tta:
            assert self.phase in ["test", "val"]
            self.tta = tr.Compose(
                (
                    tr.Rotation(inplace=True, dim=2),
                    tr.RandomApply(tr.FlipXY(inplace=True), prob=2.0 / 3.0),
                    tr.Scale(inplace=True, dims=(0, 1, 2), range=0.1),
                )
            )
        else:
            self.tta = None

        # Train time augmentations
        if train_augmentations is not None:
            assert self.phase in ["train", "trainval"]
        self.train_augmentations = train_augmentations

        # Flag for instance cutmix
        self.instance_cutmix = instance_cutmix

        # Flag for pre_process on GPU:
        self.preproc_gpu = preproc_gpu
        self.rank = rank

        self.misc = None #just a miscallaneous variable, useless

        # Add batch_size as a parameter
        self.batch_size = batch_size

        self.frustum_mix = ClusterMix(H=self.range_H,
                                     W=self.range_W, 
                                     fov_up= self.fov_up, 
                                     fov_down=self.fov_down,
                                     num_areas=[3, 4, 5, 6],
                                     prob=1.0)
        
        self.instance_copy = InstanceCopy(instance_classes=self.instances_classes,
                                          prob=1.0)
        
        if self.preproc_gpu:
            self.device = 'cuda'

            # create laser scan
            self.scan_gpu = SemLaserScanCUDA(self.dataset, self.color_dict, project_range=True, range_H=self.range_H, range_W=self.range_W, fov_up= self.fov_up, fov_down= self.fov_down, device=self.device, rank = self.rank)

            self.range_interpolation = RangeInterpolationCUDA(H=self.range_H,
                                                    W=self.W_interpolation,
                                                    fov_up=self.fov_up,
                                                    fov_down=self.fov_down,
                                                    ignore_index=255,
                                                    device=self.device)
            
            # Collate fn
            self.collate = Collate(device=self.device)

            # to map labels from original to learning format
            self.map = MapLabels(device=self.device, rank=self.rank)

        else:
            self.device = 'cpu'

            # create laser scan
            self.scan = SemLaserScan(self.dataset, self.color_dict, project_range=True, range_H=self.range_H, range_W=self.range_W, fov_up= self.fov_up, fov_down= self.fov_down)
            
            self.range_interpolation = RangeInterpolation(H=self.range_H,
                                                        W=self.W_interpolation,
                                                        fov_up=self.fov_up,
                                                        fov_down=self.fov_down,
                                                        ignore_index=255)
            # Collate fn
            self.collate = Collate(device=self.device)

            # to map labels from original to learning format
            self.map = MapLabels(device=self.device)
        

    def prepare_input_features(self, pc_orig):
        if self.input_feat == "none":
            return pc_orig
        else:
            # Concatenate desired input features to coordinates
            pc = [pc_orig[:, :3]]  # Initialize with coordinates
            for type in self.input_feat:
                if type == "intensity":
                    pc.append(pc_orig[:, 3:])
                elif type == "height":
                    pc.append(pc_orig[:, 2:3])
                elif type == "radius":
                    r_xyz = np.linalg.norm(pc_orig[:, :3], axis=1, keepdims=True)
                    pc.append(r_xyz)
                else:
                    raise ValueError(f"Unknown feature: {type}")
            return np.concatenate(pc, 1)

    def load_pc(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        if self.phase == "train" or self.phase == "trainval":
            if self.preproc_gpu:
                # DataLoader still calls __getitem__; return index-only to avoid CPU work.
                return index
            # Load original point cloud
            pc_orig, labels_learning, labels_orig, filename, labelname, eval_filename  = self.load_pc(index)


            # Randomly sample a point cloud from the dataset for mixing
            if self.train_augmentations is not None:
                mix_index = np.random.randint(0, self.__len__())
                mix_pc_orig, _, mix_labels, _, _, _  = self.load_pc(mix_index)
                mix_pc = self.prepare_input_features(mix_pc_orig)

            # Prepare input feature
            pc = self.prepare_input_features(pc_orig)

            # Test time augmentation
            if self.tta is not None:
                pc, labels = self.tta(pc, labels_orig)
            else:
                pc, labels = pc, labels_orig

            # Augment data
            if self.train_augmentations is not None:
                pc, labels = self.train_augmentations(pc, labels)
                mix_pc, mix_labels = self.train_augmentations(mix_pc, mix_labels)
                pc, labels = self.frustum_mix(pc, labels, mix_pc, mix_labels)
                pc, labels = self.instance_copy(pc, labels, mix_pc, mix_labels)
            
            pc, labels = self.range_interpolation(pc, labels)

            points = pc[:, 0:3]    # get xyz
            remissions = pc[:, 3]  # get remission

            # set points in scan and do projections
            self.scan.set_points(points, remissions)  

            # set labels in scan and do labels projection
            self.scan.set_label(labels)  

            # map unused classes to used classes (also for projection)
            # Do it here and make ignore index 255 for noise, since labels_orig is considered and not labels_learning
            self.scan.sem_label = self.map(self.scan.sem_label, self.learning_map)
            self.scan.range_proj_sem_label = self.map(self.scan.range_proj_sem_label, self.learning_map)

            # Step 3: Create a structure similar to the output of the first script
            
            # Get the projected points, coordinates, and relevant information
            points_xyzi = np.hstack([self.scan.points, self.scan.remissions[:, None]])
            voxels = torch.tensor(points_xyzi)  # Original points, as voxels in first script
            res_coors = torch.tensor(np.stack([self.scan.proj_range_y, self.scan.proj_range_x], axis=-1), dtype=torch.int64)  # Range image coordinates

            # Get the segmentation labels from the range projection
            # proj_semantic_labels = torch.tensor(self.scan.range_proj_sem_label, dtype=torch.long)
            proj_semantic_labels = self.scan.range_proj_sem_label
            pt_labels = self.scan.sem_label

            # Full-resolution range-view aux maps (used by late-stage RangeMamba blocks)
            # Keep full resolution here; downsample inside the model with valid-weighted pooling.
            valid0 = (self.scan.proj_range_idx >= 0).astype(np.float32)
            depth0 = self.scan.proj_range.astype(np.float32)
            intensity0 = self.scan.proj_range_remission.astype(np.float32)
            depth0 = np.where(valid0 > 0, depth0, 0.0).astype(np.float32)
            intensity0 = np.where(valid0 > 0, intensity0, 0.0).astype(np.float32)
            range_aux = {"depth": depth0, "intensity": intensity0, "valid": valid0}

            out = {
            'points': points_xyzi,
            'voxels': voxels,
            'coors' : res_coors,
            'proj_labels': proj_semantic_labels,
            'pt_labels': pt_labels,
            'range_aux': range_aux,
            'filename': filename,
            'eval_filename': eval_filename
            }

            return out

        elif self.phase == "val" or self.phase == "test":
            return index #just for the sake of returning anything
        else:
            raise Exception("Phase is not indicated as train, val, trainval or test")


    # @profile
    def process_batch_cpu(self, index):
        # Load original point cloud
        pc_orig, labels_learning, labels_orig, filename, labelname, eval_filename  = self.load_pc(index)
        orig_len = pc_orig.shape[0]

        # Randomly sample a point cloud from the dataset for mixing
        if self.train_augmentations is not None:
            mix_index = np.random.randint(0, self.__len__())
            mix_pc_orig, _, mix_labels, _, _, _  = self.load_pc(mix_index)
            mix_pc = self.prepare_input_features(mix_pc_orig)

        # Prepare input feature
        pc = self.prepare_input_features(pc_orig)

        # Test time augmentation
        if self.tta is not None:
            pc, labels = self.tta(pc, labels_orig)
        else:
            pc, labels = pc, labels_orig

        # Augment data
        if self.train_augmentations is not None:
            pc, labels = self.train_augmentations(pc, labels)
            mix_pc, mix_labels = self.train_augmentations(mix_pc, mix_labels)
            pc, labels = self.frustum_mix(pc, labels, mix_pc, mix_labels)
            pc, labels = self.instance_copy(pc, labels, mix_pc, mix_labels)

        pc, labels = self.range_interpolation(pc, labels)

        points = pc[:, 0:3]    # get xyz
        remissions = pc[:, 3]  # get remission

        # set points in scan and do projections
        self.scan.set_points(points, remissions)  

        # set labels in scan and do labels projection
        self.scan.set_label(labels)  

        # map unused classes to used classes (also for projection)
        # Do it here and make ignore index 255 for noise, since labels_orig is considered and not labels_learning
        self.scan.sem_label = self.map(self.scan.sem_label, self.learning_map)
        self.scan.range_proj_sem_label = self.map(self.scan.range_proj_sem_label, self.learning_map)
        
        # Get the projected points, coordinates, and relevant information
        points_xyzi = np.hstack([self.scan.points, self.scan.remissions[:, None]])
        voxels = torch.tensor(points_xyzi)  # Original points, as voxels in first script
        res_coors = torch.tensor(np.stack([self.scan.proj_range_y, self.scan.proj_range_x], axis=-1), dtype=torch.int64)  # Range image coordinates

        # Get the segmentation labels from the range projection
        # proj_semantic_labels = torch.tensor(self.scan.range_proj_sem_label, dtype=torch.long)
        proj_semantic_labels = self.scan.range_proj_sem_label
        pt_labels = self.scan.sem_label

        # Full-resolution range-view aux maps (used by late-stage RangeMamba blocks)
        valid0 = (self.scan.proj_range_idx >= 0).astype(np.float32)
        depth0 = self.scan.proj_range.astype(np.float32)
        intensity0 = self.scan.proj_range_remission.astype(np.float32)
        depth0 = np.where(valid0 > 0, depth0, 0.0).astype(np.float32)
        intensity0 = np.where(valid0 > 0, intensity0, 0.0).astype(np.float32)
        range_aux = {"depth": depth0, "intensity": intensity0, "valid": valid0}

        out = {
        'points': points_xyzi,
        'voxels': voxels,
        'coors' : res_coors,
        'proj_labels': proj_semantic_labels,
        'pt_labels': pt_labels,
        'range_aux': range_aux,
        'orig_len': orig_len,
        'filename': filename,
        'eval_filename': eval_filename
        }

        out = self.collate([out])
        return out, pc
    

    def load_batch_to_gpu(self, index):
        # Load original point cloud
        pc_orig, labels_learning, labels_orig, filename, labelname, eval_filename  = self.load_pc(index)

        # Randomly sample a point cloud from the dataset for mixing
        if self.train_augmentations is not None:
            mix_index = np.random.randint(0, self.__len__())
            mix_pc_orig, _, mix_labels, _, _, _  = self.load_pc(mix_index)
            mix_pc = self.prepare_input_features(mix_pc_orig)

        # Prepare input feature
        pc = self.prepare_input_features(pc_orig)

        # Test time augmentation
        if self.tta is not None:
            pc, labels = self.tta(pc, labels_orig)
        else:
            pc, labels = pc, labels_orig

        # Augment data
        if self.train_augmentations is not None:
            pc, labels = self.train_augmentations(pc, labels)
            mix_pc, mix_labels = self.train_augmentations(mix_pc, mix_labels)
            pc, labels = self.frustum_mix(pc, labels, mix_pc, mix_labels)
            pc, labels = self.instance_copy(pc, labels, mix_pc, mix_labels)

        on_gpu_data = self.collate([{'pc': pc}])

        pc = on_gpu_data['pc'][0]

        # move labels data outside Collate to explicitely sepecify dtype=torch.int32
        labels = torch.tensor(labels, dtype=torch.int32).cuda(self.rank, non_blocking=True)  

        return pc, labels
        
    def process_batch_gpu(self, pc, labels):
            pc, labels = self.range_interpolation(pc, labels)

            points = pc[:, 0:3]    # get xyz
            remissions = pc[:, 3]  # get remission

            # set points in scan and do projections
            self.scan_gpu.set_points(points, remissions)  

            # set labels in scan and do labels projection
            self.scan_gpu.set_label(labels)

            # map unused classes to used classes (also for projection)
            # Do it here and make ignore index 255 for noise, since labels_orig is considered and not labels_learning
            self.scan_gpu.sem_label = self.map(self.scan_gpu.sem_label, self.learning_map)
            self.scan_gpu.range_proj_sem_label = self.map(self.scan_gpu.range_proj_sem_label, self.learning_map)

            # Get the projected points, coordinates, and relevant information
            points_xyzi = torch.cat([self.scan_gpu.points, self.scan_gpu.remissions[:, None]], dim=-1)  # Concatenate along the last dimension
            voxels = points_xyzi 
            res_coors = torch.stack([self.scan_gpu.proj_range_y, self.scan_gpu.proj_range_x], dim=-1).to(torch.int64)  # Range image coordinates
            # Prepend batch index (0) for single-sample GPU preprocessing.
            batch_id = torch.zeros((res_coors.shape[0], 1), dtype=torch.int64, device=res_coors.device)
            res_coors = torch.cat([batch_id, res_coors], dim=1)

            # Get the segmentation labels from the range projection
            proj_semantic_labels = self.scan_gpu.range_proj_sem_label.to(torch.float32)
            pt_labels = self.scan_gpu.sem_label.to(torch.float32)

            # Full-resolution range-view aux maps (used by late-stage RangeMamba blocks)
            valid0 = (self.scan_gpu.proj_range_idx >= 0).to(torch.float32)
            depth0 = torch.where(valid0 > 0, self.scan_gpu.proj_range, torch.zeros_like(self.scan_gpu.proj_range))
            intensity0 = torch.where(valid0 > 0, self.scan_gpu.proj_range_remission, torch.zeros_like(self.scan_gpu.proj_range_remission))
            range_aux = {"depth": depth0, "intensity": intensity0, "valid": valid0}

            out = {
            'points': points_xyzi,
            'voxels': voxels,
            'coors' : res_coors,
            'proj_labels': proj_semantic_labels,
            'pt_labels': pt_labels,
            'range_aux': range_aux,
            }
            
            return out, pc

        
class Collate:
    def __init__(self, misc=None, device='cpu'):
        self.misc = misc
        self.device = device  # Specify the target device (e.g., 'cuda' for GPU)

    def __call__(self, inputs: List[Any]):
        if isinstance(inputs[0], dict):
            output = {}
            for name in inputs[0].keys():
                if name == "coors":
                    coors_with_batch = []
                    for batch_id, sample in enumerate(inputs):
                        coors = sample[name]
                        if isinstance(coors, np.ndarray):
                            coors = torch.tensor(coors, dtype=torch.int64, device=self.device)
                        elif isinstance(coors, torch.Tensor):
                            coors = coors.to(self.device).to(torch.int64)
                        else:
                            raise TypeError("coors must be a numpy array or torch tensor")
                        batch_col = torch.full(
                            (coors.shape[0], 1),
                            batch_id,
                            dtype=torch.int64,
                            device=coors.device,
                        )
                        coors_with_batch.append(torch.cat([batch_col, coors], dim=1))
                    output[name] = torch.cat(coors_with_batch, dim=0)
                elif isinstance(inputs[0][name], np.ndarray):
                    output[name] = [torch.tensor(input[name], dtype=torch.float32, device=self.device) for input in inputs]
                elif isinstance(inputs[0][name], torch.Tensor):
                    output[name] = torch.cat([input[name].to(self.device) for input in inputs], dim=0)
                elif isinstance(inputs[0][name], list) and all(isinstance(item, torch.Tensor) for item in inputs[0][name]):
                    output[name] = torch.cat([tensor.to(self.device) for input in inputs for tensor in input[name]], dim=0)
                else:
                    output[name] = [input[name] for input in inputs]
            return output
        else:
            return inputs


class MapLabels:
    def __init__(self, misc=None, device = 'cuda', rank = 0):
        self.misc = misc
        self.device = device
        self.rank = rank

    def __call__(self, label, mapdict):
        # Convert the mapdict to a tensor or array where the index represents the key, and the value at that index is the map value
        max_key = max(mapdict.keys())
        if self.device == 'cuda':
            lookup = torch.zeros(max_key + 1, dtype=torch.int).cuda(self.rank, non_blocking=True)
        elif self.device == 'cpu':
            lookup = np.zeros(max_key + 1, dtype=int)
        else:
            raise Exception("device can only be set to \'cuda\' or \'cpu\'")
        # Fill the lookup tensor with the mapped values
        for key, value in mapdict.items():
            lookup[key] = value

        # Use tensor indexing to map the array values
        mapped_tensor = lookup[label]

        # Shift all values by -1
        mapped_tensor = mapped_tensor - 1

        # Replace all -1 by 255 for ignore index
        mapped_tensor[mapped_tensor == -1] = 255

        return mapped_tensor
