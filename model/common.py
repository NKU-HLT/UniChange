# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class FusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(FusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self._initialize_weights() # 调用自定义的初始化方法
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, 1)
        if self.norm.bias is not None:
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class AggregatorModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, padding: int = 0):
        super(AggregatorModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self._initialize_weights() # 调用自定义的初始化方法
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, 1)
        if self.norm.bias is not None:
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class MultiScaleFusionNeck(nn.Module):
    #Multi-scale Feature Fusion Neck for Change Detection.
    
    def __init__(self,
                 in_channels: int = 512,
                 out_channels: int = 256,
                 num_scales: int = 5,
                 use_bn: bool = True,
                 activation: str = 'relu') -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        self.scale_generators = nn.ModuleList()
        
        self.scale_generators.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2),
            nn.BatchNorm2d(in_channels // 2) if use_bn else nn.Identity(),
            self.activation,
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 2, 2)
        ))
        
        self.scale_generators.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        ))
        
        self.scale_generators.append(nn.Sequential(
            nn.Identity()
        ))
        
        self.scale_generators.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        ))
        
        self.scale_generators.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4)
        ))
        
        scale_output_channels = [
            in_channels // 4,  
            in_channels // 2,  
            in_channels,       
            in_channels,     
            in_channels        
        ]
        

        self.lateral_convs = nn.ModuleList()
        for i in range(num_scales):
            input_ch = scale_output_channels[i]
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(input_ch, out_channels, 1, bias=not use_bn),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                self.activation
            ))
        
        self.fpn_convs = nn.ModuleList()
        for i in range(num_scales):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_bn),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                self.activation
            ))
        
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * num_scales, out_channels, 1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            self.activation
        )
        
       
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 512, 64, 64)
            
        Returns:
            Output tensor of shape (B, 256, 64, 64)
        """
        
        multi_scale_features = []
        for generator in self.scale_generators:
            multi_scale_features.append(generator(x))
        
        
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(multi_scale_features[i]))
        
       
        fpn_outputs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            fpn_outputs.append(fpn_conv(laterals[i]))
        
       
        unified_features = []
        for feat in fpn_outputs:
            if feat.shape[2:] != (64, 64):
                 
                feat_float32 = feat.float()
                feat_resized = F.interpolate(feat_float32, size=(64, 64), mode='bilinear', align_corners=False)
                
                feat = feat_resized.to(feat.dtype)
               
            unified_features.append(feat)
        
       
        concatenated = torch.cat(unified_features, dim=1)  # (B, 256*5, 64, 64)
        
       
        output = self.fusion_conv(concatenated)  # (B, 256, 64, 64)
        
        return output