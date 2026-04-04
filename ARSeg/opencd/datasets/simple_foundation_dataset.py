# Copyright (c) Open-CD. All rights reserved.
"""
纯PyTorch版本的FoundationDataset实现
完全脱离mmengine框架，保持核心数据处理逻辑一致
"""
import copy
import os.path as osp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
import logging
import random
import json

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

# 导入原项目的transform类
from opencd.datasets.transforms.loading_modified import (
    MultiImgLoadImageFromFile_Modified,
    MultiImgMultiAnnLoadAnnotations_Modified,
)
from opencd.datasets.transforms.formatting import MultiImgPackSegInputs_Modified
# 裁剪在 transforms_modified.py，翻转/时间交换/光照扰动在 transforms.py
from opencd.datasets.transforms.transforms_modified import (
    MultiImgRandomCrop_Modified,
    RandomMosaic_Modified,
)
from opencd.datasets.transforms.transforms import (
    MultiImgRandomFlip,
    MultiImgExchangeTime,
    MultiImgPhotoMetricDistortion,
)


class SimpleFoundationDataset(Dataset):
    """
    纯PyTorch版本的FoundationDataset
    保持与原版本相同的数据处理逻辑
    """
    
    METAINFO = dict(
        classes=('unchanged / no_building', 'changed / building'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self,
                 data_list: str = '',
                 format_seg_map=None,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 test_mode: bool = False,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 transform=None):
        """
        初始化数据集
        
        Args:
            data_list: 数据列表文件路径
            format_seg_map: 分割图格式
            metainfo: 元信息
            data_root: 数据根目录
            data_prefix: 数据前缀
            filter_cfg: 过滤配置
            indices: 索引
            test_mode: 测试模式
            ignore_index: 忽略索引
            reduce_zero_label: 是否减少零标签
            transform: 数据变换
        """
        self.data_list = data_list
        self.format_seg_map = format_seg_map
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.test_mode = test_mode
        self.transform = transform
        
        # 设置元信息
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        
        # 获取标签映射
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label
            )
        )
        
        # 更新调色板
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))
        
        # 连接路径
        if self.data_root is not None:
            self._join_prefix()
        
        # 加载数据 - 根据文件扩展名选择加载方式
        if self.data_list.endswith('.json'):
            self.data_list = self.load_geopixelcdjson_data_list()
        else:
            self.data_list = self.load_data_list()
        self.data_list = self.filter_data()
        
        # 根据索引获取子集
        if self._indices is not None:
            self.data_list = self._get_subset(self._indices)
        
        # 预构建所有样本索引（简化版本，因为只有only_cd_label类型）
        # 这样可以实现O(1)的Mosaic采样，避免每次O(n)遍历
        self.all_indices = list(range(len(self.data_list)))
        
        # 初始化与原项目一致的数据处理pipeline
        if not self.test_mode:
            # 训练：基础pipeline（加载 -> 标注 -> 随机裁剪 -> 翻转 -> 时相交换 -> 光照扰动）
            self.base_pipeline = [
                MultiImgLoadImageFromFile_Modified(),
                MultiImgMultiAnnLoadAnnotations_Modified(),
                MultiImgRandomCrop_Modified(crop_size=(512, 512), cat_max_ratio=0.95),
                MultiImgRandomFlip(prob=0.5, direction='horizontal'),
                MultiImgRandomFlip(prob=0.5, direction='vertical'),
                # MultiImgExchangeTime(prob=0.5),
                MultiImgPhotoMetricDistortion(
                    brightness_delta=10,
                    contrast_range=(0.8, 1.2),
                    saturation_range=(0.8, 1.2),
                    hue_delta=10,
                ),
            ]
            # 外层pipeline（Mosaic增强 -> 打包）
            self.outer_pipeline = [
                RandomMosaic_Modified(img_scale=(512, 512), center_ratio_range=(0.25, 0.75), prob=0.5),
                MultiImgPackSegInputs_Modified(),
            ]
        else:
            # 验证/测试：加载 -> 标注 -> 打包
            self.pipeline = [
                MultiImgLoadImageFromFile_Modified(),
                MultiImgMultiAnnLoadAnnotations_Modified(),
                MultiImgPackSegInputs_Modified(),
            ]
        
        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    def _load_metainfo(self, metainfo: Optional[dict] = None) -> dict:
        """加载元信息"""
        if metainfo is None:
            metainfo = copy.deepcopy(self.METAINFO)
        else:
            metainfo = copy.deepcopy(metainfo)
            # 如果提供了新的classes，更新METAINFO
            if 'classes' in metainfo:
                metainfo.update(
                    dict(
                        classes=metainfo['classes'],
                        palette=metainfo.get('palette', self.METAINFO['palette'])
                    )
                )
        return metainfo

    def _join_prefix(self):
        """连接数据前缀"""
        if self.data_root is not None:
            for key in self.data_prefix:
                if self.data_prefix[key] is not None:
                    self.data_prefix[key] = osp.join(self.data_root, self.data_prefix[key])

    def _get_subset(self, indices: Union[int, Sequence[int]]) -> List[dict]:
        """获取数据子集"""
        if isinstance(indices, int):
            indices = [indices]
        return [self.data_list[i] for i in indices]

    def load_data_list(self) -> List[dict]:
        """
        加载数据列表 - 保持与原版本完全一致的逻辑
        
        Returns:
            list[dict]: 所有数据信息
        """
        data_list = []
        lines_specific_data_lists = []
        
        # 读取储存具体数据txt的data_list，忽略注释、空行，并去重
        with open(self.data_list, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            lines_specific_data_lists.append(line)
        
        # 读取每个data_list，该循环遍历每个数据集的train/val/test的list
        for specific_data_list_path in lines_specific_data_lists:
            with open(specific_data_list_path, 'r') as f:
                lines = f.readlines()
            
            name = specific_data_list_path.split('/')[-2] + '_' + specific_data_list_path.split('/')[-1].split('.')[0]
            
            for data_pair in lines:
                data_pair = data_pair.strip()
                if not data_pair:
                    continue
                    
                img_a, img_b, label_cd, label_a, label_b = data_pair.split('\t')
                data_info = dict(img_a_path=img_a)
                data_info['img_b_path'] = img_b
                data_info['label_cd'] = label_cd
                data_info['label_a'] = label_a
                data_info['label_b'] = label_b

                # 保持与原版本完全一致的type判断逻辑
                if data_info['img_b_path'] == '**' and data_info['label_cd'] == '**' and data_info['label_b'] == '**':
                    data_info['type'] = 'only_building_label'
                elif data_info['label_a'] == '**' and data_info['label_b'] == '**':
                    data_info['type'] = 'only_cd_label'
                else:
                    data_info['type'] = 'all_label'
                
                data_info['dataset_name'] = name

                # 保持与原版本一致的字段
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []

                data_list.append(data_info)
        
        return data_list

    def load_geopixelcdjson_data_list(self) -> List[dict]:
        """
        从GeoPixelCD JSON文件加载数据列表
        
        Returns:
            list[dict]: 所有数据信息
        """
        data_list = []
        
        # 读取JSON文件
        with open(self.data_list, 'r') as f:
            json_data = json.load(f)
        
        # 从JSON文件路径中提取数据集名称
        dataset_name = osp.basename(osp.dirname(self.data_list)) + '_' + osp.splitext(osp.basename(self.data_list))[0]
        
        for item in json_data:
            # 提取图像路径和标签路径
            img_a = item['image1']  # 时相A图像
            img_b = item['image2']  # 时相B图像
            label_cd = item['label']  # 变化检测标签
            
            # 构建数据信息字典
            data_info = dict(img_a_path=img_a)
            data_info['img_b_path'] = img_b
            data_info['label_cd'] = label_cd
            data_info['label_a'] = '**'  # 默认设置为'**'
            data_info['label_b'] = '**'  # 默认设置为'**'
            
            # 设置为only_cd_label类型
            data_info['type'] = 'only_cd_label'
            data_info['dataset_name'] = dataset_name
            
            # 保持与原版本一致的字段
            data_info['label_map'] = self.label_map
            data_info['format_seg_map'] = self.format_seg_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            
            data_list.append(data_info)
        
        return data_list

    def filter_data(self) -> List[dict]:
        """过滤数据 - 保持与原版本一致的逻辑"""
        if self.filter_cfg is None:
            return self.data_list
        
        # 简单的过滤逻辑，可以根据需要扩展
        filtered_data = []
        for data_info in self.data_list:
            # 这里可以添加具体的过滤条件
            filtered_data.append(data_info)
        
        return filtered_data

    @classmethod
    def get_label_map(cls, new_classes: Optional[Sequence] = None) -> Union[Dict, None]:
        """获取标签映射 - 保持与原版本完全一致的逻辑"""
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """更新调色板 - 保持与原版本完全一致的逻辑"""
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette


    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        """
        获取数据项 - 使用原项目的pipeline处理，只改变数据结构
        
        Args:
            idx: 数据索引
            
        Returns:
            dict: 包含inputs和data_samples的字典
        """
        data_info = self.data_list[idx]
        
        # 使用原项目的pipeline处理数据
        results = data_info.copy()
        
        if not self.test_mode:
            # 训练模式：使用双层pipeline
            # 1. 应用基础pipeline
            for transform in self.base_pipeline:
                results = transform(results)
            
            # 2. 应用外层pipeline（包含Mosaic增强）
            img_augmented_temp = None
            for transform in self.outer_pipeline:
                if hasattr(transform, 'get_indices'):
                    # Mosaic增强需要其他样本（使用预构建索引实现O(1)采样）
                    import random
                    # 使用预构建的索引进行高效采样（O(1)复杂度）
                    mix_indices = random.sample(self.all_indices, 3)
                    mix_results = [copy.deepcopy(self.data_list[i]) for i in mix_indices]
                    # 对mix_results也应用基础pipeline
                    for mix_result in mix_results:
                        for base_transform in self.base_pipeline:
                            mix_result = base_transform(mix_result)
                    results['mix_results'] = mix_results
                
                # 在MultiImgPackSegInputs_Modified之前提取数据增强后的图像
                if transform.__class__.__name__ == 'MultiImgPackSegInputs_Modified':
                    # 提取数据增强后的双时相图像（numpy数组，0-255像素值）
                    img_a_np = results['img'][0].copy()  # 时相A，numpy数组 (H, W, 3)，确保连续
                    img_b_np = results['img'][1].copy()  # 时相B，numpy数组 (H, W, 3)，确保连续
                    img_augmented_temp = {
                        'img_a': img_a_np,
                        'img_b': img_b_np
                    }
                
                results = transform(results)
                
                # 清理mix_results
                if 'mix_results' in results:
                    results.pop('mix_results')
            
            # 在pipeline结束后重新设置img_augmented
            if img_augmented_temp is not None:
                results['img_augmented'] = img_augmented_temp
        else:
            # 测试模式：使用简单pipeline
            img_augmented_temp = None
            for transform in self.pipeline:
                # 在MultiImgPackSegInputs_Modified之前提取原始numpy数组
                if transform.__class__.__name__ == 'MultiImgPackSegInputs_Modified':
                    # 提取原始加载的双时相图像（numpy数组，0-255像素值）
                    img_a_np = results['img'][0].copy()  # 时相A，numpy数组 (H, W, 3)，确保连续
                    img_b_np = results['img'][1].copy()  # 时相B，numpy数组 (H, W, 3)，确保连续
                    img_augmented_temp = {
                        'img_a': img_a_np,
                        'img_b': img_b_np
                    }
                
                results = transform(results)
            
            # 在pipeline结束后重新设置img_augmented
            if img_augmented_temp is not None:
                results['img_augmented'] = img_augmented_temp
        
        # 从pipeline结果中提取数据
        inputs = results['inputs']  # 已经是tensor格式 (6, H, W)
        data_sample = results['data_samples']  # 这是SegDataSample对象
        
        # 提取数据增强后的双时相图像（numpy数组，0-255像素值）
        if 'img_augmented' in results:
            img_augmented = results['img_augmented']
        else:
            # 如果没有img_augmented，说明是旧版本，设置为None
            img_augmented = None

        # 注意：不要在这里修改 PixelData.data，改为在构造成dict后再做尺寸对齐
        
        # 将SegDataSample转换为字典格式，保持兼容性
        data_sample_dict = {
            'metainfo': data_sample.metainfo if hasattr(data_sample, 'metainfo') else {},
            'gt_sem_seg': {
                'data': data_sample.gt_sem_seg.data if hasattr(data_sample, 'gt_sem_seg') and data_sample.gt_sem_seg is not None else None
            }
        }
        
        # 如果有其他字段，也转换
        if hasattr(data_sample, 'gt_sem_seg_from') and data_sample.gt_sem_seg_from is not None:
            data_sample_dict['gt_sem_seg_from'] = {
                'data': data_sample.gt_sem_seg_from.data
            }
        if hasattr(data_sample, 'gt_sem_seg_to') and data_sample.gt_sem_seg_to is not None:
            data_sample_dict['gt_sem_seg_to'] = {
                'data': data_sample.gt_sem_seg_to.data
            }
        
        # 返回与原版本兼容的格式，并添加数据增强后的图像
        sample = {
            'inputs': inputs,
            'data_samples': data_sample_dict
        }
        
        # 添加数据增强后的双时相图像（如果存在）
        if img_augmented is not None:
            sample['inputs_augmented'] = img_augmented

        # 训练稳定性保障：在dict阶段统一进行尺寸对齐（不触碰PixelData）
        if not self.test_mode:
            import torch.nn.functional as F
            _, H, W = sample['inputs'].shape
            sizes = [(H, W)]
            if 'gt_sem_seg' in sample['data_samples'] and sample['data_samples']['gt_sem_seg'] is not None:
                sizes.append(tuple(sample['data_samples']['gt_sem_seg']['data'].shape[-2:]))
            if 'gt_sem_seg_from' in sample['data_samples'] and sample['data_samples']['gt_sem_seg_from'] is not None:
                sizes.append(tuple(sample['data_samples']['gt_sem_seg_from']['data'].shape[-2:]))
            if 'gt_sem_seg_to' in sample['data_samples'] and sample['data_samples']['gt_sem_seg_to'] is not None:
                sizes.append(tuple(sample['data_samples']['gt_sem_seg_to']['data'].shape[-2:]))

            target_h = max([s[0] for s in sizes] + [512])
            target_w = max([s[1] for s in sizes] + [512])

            def pad_to(t, value):
                th, tw = t.shape[-2:]
                ph, pw = target_h - th, target_w - tw
                if ph > 0 or pw > 0:
                    return F.pad(t, (0, pw, 0, ph), value=value)
                return t

            sample['inputs'] = pad_to(sample['inputs'], 0)
            if 'gt_sem_seg' in sample['data_samples'] and sample['data_samples']['gt_sem_seg'] is not None:
                seg = sample['data_samples']['gt_sem_seg']['data']
                sample['data_samples']['gt_sem_seg']['data'] = pad_to(seg.float(), float(self.ignore_index)).to(seg.dtype)
            if 'gt_sem_seg_from' in sample['data_samples'] and sample['data_samples']['gt_sem_seg_from'] is not None:
                segf = sample['data_samples']['gt_sem_seg_from']['data']
                sample['data_samples']['gt_sem_seg_from']['data'] = pad_to(segf.float(), float(self.ignore_index)).to(segf.dtype)
            if 'gt_sem_seg_to' in sample['data_samples'] and sample['data_samples']['gt_sem_seg_to'] is not None:
                segt = sample['data_samples']['gt_sem_seg_to']['data']
                sample['data_samples']['gt_sem_seg_to']['data'] = pad_to(segt.float(), float(self.ignore_index)).to(segt.dtype)

            # 同步更新metainfo中的img_shape
            if 'metainfo' in sample['data_samples'] and isinstance(sample['data_samples']['metainfo'], dict):
                sample['data_samples']['metainfo']['img_shape'] = (target_h, target_w, 3)

        return sample

    @property
    def metainfo(self) -> dict:
        """返回元信息"""
        return self._metainfo.copy()


# 为了保持兼容性，创建一个别名
SimpleFoundationDataset = SimpleFoundationDataset
