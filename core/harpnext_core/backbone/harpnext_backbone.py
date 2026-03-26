# Copyright 2025 CEA LIST - Samir Abou Haidar
# Modifications based on code from Open-MMLab, 2018-2019

# Copyright 2018-2019 Open-MMLab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import Tensor

from ma.monarch_attention import MonarchAttention, PadType
from core.harpnext_core.backbone.range_mamba import RangeMambaSECore

class ConvSENeXt(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 reduction: int = 16,
                 norm_cfg=dict(type='BN2d', eps=1e-6, momentum=0.01),
                 act_cfg=dict(type='HSwish', inplace=True),
                 dw_conv_kernel = 7,
                 dw_conv_bias = True) -> None:
        super(ConvSENeXt, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dwconv layer
        self.conv_dw = nn.Conv2d(
            in_channels=inplanes, 
            out_channels=inplanes, 
            kernel_size=dw_conv_kernel,
            stride=stride, 
            padding=dilation,
            groups=inplanes, 
            bias=dw_conv_bias,
            device=device
        )
        self.norm1 = nn.BatchNorm2d(num_features=planes, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'], device=device)

        # pwconv layer
        self.pointwise = nn.Conv2d(
            in_channels=inplanes, 
            out_channels=planes, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=dw_conv_bias,
            device=device
        )
        self.norm2 = nn.BatchNorm2d(num_features=planes, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'], device=device)

         # Activation
        self.activation = nn.Hardswish(inplace=act_cfg['inplace'])

        # Squeeze-and-Excitation (SE) module
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes // reduction, kernel_size=1, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // reduction, planes, kernel_size=1, device=device),
            nn.Hardsigmoid(inplace=True)
        )

        # Downsample for skip connections
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Depthwise convolution
        out = self.conv_dw(x)
        out = self.norm1(out)
        out = self.activation(out)

        # Pointwise convolution
        out = self.pointwise(out)
        out = self.norm2(out)

        # Apply Squeeze-and-Excitation (SE) module
        se_weight = self.se(out)
        out = out * se_weight

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out
	 

class _PixelStage(nn.Module):
    """Stage wrapper with a stable forward signature.

    Some stage blocks require range-view aux maps (RangeMamba stages), while
    baseline stages only take x. This wrapper keeps the backbone loop unchanged.
    """

    def __init__(self, block: nn.Module, needs_range_aux: bool = False) -> None:
        super().__init__()
        self.block = block
        self.needs_range_aux = needs_range_aux

    def forward(self, x: Tensor, range_aux: Optional[dict] = None) -> Tensor:
        if self.needs_range_aux:
            if range_aux is None:
                raise ValueError("range_aux is required for this stage block")
            return self.block(x, range_aux)
        return self.block(x)


class RangeMambaStageBlock(nn.Module):
    """Conv stem + RangeMambaSECore + residual, preserving stage stride/downsample."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16,
        norm_cfg=dict(type="BN2d", eps=1e-3, momentum=0.01),
        act_cfg=dict(type="HSwish", inplace=True),
        dw_conv_kernel: int = 7,
        dw_conv_bias: bool = True,
        use_col_mamba: bool = False,
        range_mamba_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        range_mamba_cfg = range_mamba_cfg or {}

        # Local conv stem (same layout as ConvSENeXt up to the PW conv output).
        self.conv_dw = nn.Conv2d(
            in_channels=inplanes,
            out_channels=inplanes,
            kernel_size=dw_conv_kernel,
            stride=stride,
            padding=dilation,
            groups=inplanes,
            bias=dw_conv_bias,
            device=device,
        )
        self.norm1 = nn.BatchNorm2d(
            num_features=inplanes, eps=norm_cfg["eps"], momentum=norm_cfg["momentum"], device=device
        )
        self.pointwise = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=dw_conv_bias,
            device=device,
        )
        self.norm2 = nn.BatchNorm2d(
            num_features=planes, eps=norm_cfg["eps"], momentum=norm_cfg["momentum"], device=device
        )
        self.activation = nn.Hardswish(inplace=act_cfg["inplace"])

        # RangeMamba context + gating core (stage resolution).
        self.range_core = RangeMambaSECore(
            channels=planes,
            use_col_mamba=use_col_mamba,
            reduction=range_mamba_cfg.get("reduction", reduction),
            d_state=range_mamba_cfg.get("d_state", 16),
            expand=range_mamba_cfg.get("expand", 2),
            backend=range_mamba_cfg.get("backend", "mamba"),
        )

        self.downsample = downsample

    def forward(self, x: Tensor, range_aux: dict) -> Tensor:
        identity = x

        out = self.conv_dw(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.pointwise(out)
        out = self.norm2(out)

        out = self.range_core(out, range_aux)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation(out)
        return out


class MonarchSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        block_size: int = 16,
        num_steps: int = 2,
        pad_type: PadType = PadType.post,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn = MonarchAttention(
            block_size=block_size, num_steps=num_steps, pad_type=pad_type
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, C)
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = self.attn(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim)
        return out


class ConvMonarchBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        mlp_ratio: int = 4,
        attn_heads: int = 4,
        attn_block_size: int = 16,
        attn_num_steps: int = 2,
        attn_pad_type: PadType = PadType.post,
        dw_conv_kernel: int = 7,
        dw_conv_bias: bool = True,
    ) -> None:
        super().__init__()

        # local modeling
        padding = (dw_conv_kernel // 2) * dilation
        self.dwconv = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size=dw_conv_kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=inplanes,
            bias=dw_conv_bias,
        )
        self.proj = (
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=dw_conv_bias)
            if inplanes != planes
            else nn.Identity()
        )

        self.norm1 = nn.LayerNorm(planes)

        # global modeling
        self.attn = MonarchSelfAttention(
            planes,
            num_heads=attn_heads,
            block_size=attn_block_size,
            num_steps=attn_num_steps,
            pad_type=attn_pad_type,
        )

        self.norm2 = nn.LayerNorm(planes)

        # feedforward
        hidden = int(planes * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(planes, hidden),
            nn.GELU(),
            nn.Linear(hidden, planes),
        )

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        x = self.dwconv(x)
        x = self.proj(x)

        bsz, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(bsz, height * width, channels)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        x = x.view(bsz, height, width, channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        return x + shortcut


class EfficientTransformationPipeline(nn.Module):
    def __init__(self, nx, ny):
        super(EfficientTransformationPipeline, self).__init__()
        self.nx = nx
        self.ny = ny
    
    def point2cluster(self, point_features: torch.Tensor, pts_coors: torch.Tensor, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        coors = pts_coors.clone()
        coors[:, 1:3] //= stride

        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0, sorted=True)
        
        cluster_features = torch.zeros(voxel_coors.size(0), point_features.size(1), 
                                       dtype=point_features.dtype, device=point_features.device)
        cluster_features = torch_scatter.scatter_max(point_features, inverse_map, dim=0, out=cluster_features)[0]
        
        return voxel_coors, cluster_features
    
    def cluster2pixel(self, cluster_features: torch.Tensor, coors: torch.Tensor, batch_size: int, stride: int = 1) -> torch.Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        
        indices = coors.t().long()
        values = cluster_features
        size = (batch_size, ny, nx, cluster_features.shape[-1])
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
        pixel_features = sparse_tensor.to_dense()
        
        return pixel_features.permute(0, 3, 1, 2).contiguous()
    
    def pixel2point(self, pixel_features: torch.Tensor, coors: torch.Tensor, stride: int = 1) -> torch.Tensor:
        batch_indices = coors[:, 0]
        y_indices = coors[:, 1] // stride
        x_indices = coors[:, 2] // stride
        
        return pixel_features[batch_indices, :, y_indices, x_indices].contiguous()


class HARPNeXtBackbone(nn.Module):
    arch_settings = {
    10: (ConvSENeXt, (1, 1, 1, 1))
    }

    def __init__(self,
                 in_channels: int = 16,
                 point_in_channels: int = 384,
                 output_shape: Sequence[int] = [],
                 depth: int = 34,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 dw_conv_kernel = 7,
                 dw_conv_bias = True,
                 inter_align_corners = True,
                 use_range_mamba_stage3: bool = False,
                 use_range_mamba_stage4: bool = False,
                 use_col_mamba_stage3: bool = False,
                 use_col_mamba_stage4: bool = True,
                 range_mamba_cfg: Optional[dict] = None,
                 block_type: str = "convsenext",
                 block_cfg: Optional[dict] = None,
                 stage_block_types: Optional[Sequence[str]] = None) -> None:
        super(HARPNeXtBackbone, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for HARPNeXtBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.block_cfg = block_cfg or {}
        self.block_type = block_type.lower()
        self._validate_block_type(self.block_type)
        if stage_block_types is not None:
            self.stage_block_types = [bt.lower() for bt in stage_block_types]
            if len(self.stage_block_types) != num_stages:
                raise ValueError(
                    f"stage_block_types length ({len(self.stage_block_types)}) "
                    f"must match num_stages ({num_stages})."
                )
            for bt in self.stage_block_types:
                self._validate_block_type(bt)
        else:
            self.stage_block_types = [self.block_type] * num_stages
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]

        self.use_range_mamba_stage3 = use_range_mamba_stage3
        self.use_range_mamba_stage4 = use_range_mamba_stage4
        self.use_col_mamba_stage3 = use_col_mamba_stage3
        self.use_col_mamba_stage4 = use_col_mamba_stage4
        self.range_mamba_cfg = range_mamba_cfg or {}
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
        dilations) == num_stages, \
        'The length of stage_blocks, out_channels, strides and ' \
        'dilations should be equal to num_stages.'

        self.stem = self._make_stem_layer(in_channels, stem_channels)
        self.point_stem = self._make_point_layer(point_in_channels, stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2, stem_channels)

        self.etp = EfficientTransformationPipeline(self.nx, self.ny)

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        self.inter_align_corners = inter_align_corners

        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stage_block_type = self.stage_block_types[i]
            stage_block = self._resolve_block_class(stage_block_type)
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            
            res_layer = self._make_res_layer(
                block=stage_block,
                block_type=stage_block_type,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                dw_conv_kernel=dw_conv_kernel,
                dw_conv_bias=dw_conv_bias,
                index=i,
            )
            self.point_fusion_layers.append(self._make_point_layer(inplanes + planes, planes))
            self.pixel_fusion_layers.append(self._make_fusion_layer(planes * 2, planes))
            self.attention_layers.append(self._make_attention_layer(planes))

            inplanes = planes
            # self.res_layers.append(res_layer)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []

        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = self._make_fusion_layer(in_channels, fuse_channel)
            point_fuse_layer = self._make_point_layer(in_channels, fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    def _validate_block_type(self, block_type: str) -> None:
        if block_type not in (
            "tinyvim",
            "tvim",
            "convmonarch",
            "monarch",
            "convsenext",
            "convsennext",
            "convse",
        ):
            raise KeyError(f"invalid block_type {block_type} for HARPNeXtBackbone.")

    def _resolve_block_class(self, block_type: str) -> nn.Module:
        if block_type in ("tinyvim", "tvim"):
            from core.tinyvim_core.tvimblock import HARPNeXtTinyViMBlock
            return HARPNeXtTinyViMBlock
        if block_type in ("convmonarch", "monarch"):
            return ConvMonarchBlock
        return ConvSENeXt

    #pixels stem
    def _make_stem_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True)
        )

    # points stem
    def _make_point_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    # fusion layer
    def _make_fusion_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True)
        )
    

    # attention layer
    def _make_attention_layer(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
            nn.Sigmoid()
        )

    # residual ConvSENeXt
    def _make_res_layer(self, block: nn.Module, block_type: str, inplanes, planes, num_blocks, stride, dilation, dw_conv_kernel, dw_conv_bias, index: int = 0):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, device= self.device),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01, device=self.device),
            )

        use_range_mamba = (
            block_type in ("convsenext", "convsennext", "convse")
            and (
                (index == 2 and self.use_range_mamba_stage3)
                or (index == 3 and self.use_range_mamba_stage4)
            )
        )

        if use_range_mamba:
            use_col_mamba = self.use_col_mamba_stage4 if index == 3 else self.use_col_mamba_stage3
            stage_block = RangeMambaStageBlock(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dw_conv_kernel=dw_conv_kernel,
                dw_conv_bias=dw_conv_bias,
                use_col_mamba=use_col_mamba,
                range_mamba_cfg=self.range_mamba_cfg,
            )
            return _PixelStage(stage_block, needs_range_aux=True)

        layers = []
        if block_type in ("tinyvim", "tvim"):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    index=index,
                    **self.block_cfg))
        elif block_type in ("convmonarch", "monarch"):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    dw_conv_kernel=dw_conv_kernel,
                    dw_conv_bias=dw_conv_bias,
                    **self.block_cfg))
        else:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='HSwish', inplace=True),
                    dw_conv_kernel = dw_conv_kernel,
                    dw_conv_bias=dw_conv_bias))
        inplanes = planes
        for _ in range(1, num_blocks):
            if block_type in ("tinyvim", "tvim"):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        downsample=None,
                        index=index,
                        **self.block_cfg))
            elif block_type in ("convmonarch", "monarch"):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        downsample=None,
                        dw_conv_kernel=dw_conv_kernel,
                        dw_conv_bias=dw_conv_bias,
                        **self.block_cfg))
            else:
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                        act_cfg=dict(type='HSwish', inplace=True),
                        dw_conv_kernel = dw_conv_kernel,
                        dw_conv_bias=dw_conv_bias))

        return nn.Sequential(*layers)
    
    def forward(self, voxel_dict):
        point_feats = voxel_dict['point_feats'][-1]
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        range_aux = voxel_dict.get("range_aux", None)
        batch_size = pts_coors[-1, 0].item() + 1

        x = self.etp.cluster2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)  
        map_point_feats = self.etp.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)

        stride_voxel_coors, cluster_feats = self.etp.point2cluster(point_feats, pts_coors, stride=1)
        pixel_feats = self.etp.cluster2pixel(cluster_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        outs = [x]
        out_points = [point_feats]
        prev_pixel_feats = pixel_feats

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if isinstance(res_layer, _PixelStage):
                x = res_layer(x, range_aux)
            else:
                x = res_layer(x)

            # cluster-to-point fusion
            map_point_feats = self.etp.pixel2point(x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-cluster fusion (skip for the last layer)
            if i < len(self.res_layers) - 1:
                stride_voxel_coors, cluster_feats = self.etp.point2cluster(point_feats, pts_coors, stride=self.strides[i])
                pixel_feats = self.etp.cluster2pixel(cluster_feats, stride_voxel_coors, batch_size, stride=self.strides[i])

            prev_pixel_feats_resized = F.interpolate(prev_pixel_feats, size=x.shape[2:], mode='bilinear', align_corners=self.inter_align_corners)

            # Concatenate the resized tensor
            fusion_pixel_feats = torch.cat((prev_pixel_feats_resized, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x

            outs.append(x)
            out_points.append(point_feats)
            
            if i < len(self.res_layers) - 1:
                prev_pixel_feats = pixel_feats

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(outs[i], size=outs[0].size()[2:], mode='bilinear', align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers, self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict
