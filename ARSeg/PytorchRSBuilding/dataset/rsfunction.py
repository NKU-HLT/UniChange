import sys
import os
import copy
import torch
import numpy as np
from PIL import Image

sys.path.append('./ARSeg') #Change the absolute path of the ARseg folder

from opencd.datasets.transforms.loading_modified import (
    MultiImgLoadImageFromFile_Modified,
    MultiImgMultiAnnLoadAnnotations_Modified,
)
from opencd.datasets.transforms.formatting import MultiImgPackSegInputs_Modified

from opencd.datasets.transforms.transforms_modified import (
    MultiImgRandomResize_Modified,
)
from opencd.datasets.transforms.transforms import (
    MultiImgRandomFlip,
    MultiImgPhotoMetricDistortion,
)


def test_process_single_sample(img_a_path, img_b_path, label_cd_path):
    
    data_info = {
        'img_a_path': img_a_path,
        'img_b_path': img_b_path,
        'label_cd': label_cd_path,
        'label_a': '**',  
        'label_b': '**',  
        'type': 'only_cd_label',  
        'dataset_name': 'custom_dataset',
        'label_map': None,
        'format_seg_map': None,
        'reduce_zero_label': False,
        'seg_fields': []
    }
    
    pipeline = [
        MultiImgLoadImageFromFile_Modified(),
        MultiImgMultiAnnLoadAnnotations_Modified(),
        MultiImgPackSegInputs_Modified(),
    ]
    
    results = data_info.copy()
    
    for transform in pipeline:
        results = transform(results)
    
    inputs = results['inputs']  
    data_sample = results['data_samples']  
    
    
    data_sample_dict = {
        'metainfo': data_sample.metainfo if hasattr(data_sample, 'metainfo') else {},
        'gt_sem_seg': {
            'data': data_sample.gt_sem_seg.data if hasattr(data_sample, 'gt_sem_seg') and data_sample.gt_sem_seg is not None else None
        }
    }
    
   
    if hasattr(data_sample, 'gt_sem_seg_from') and data_sample.gt_sem_seg_from is not None:
        data_sample_dict['gt_sem_seg_from'] = {
            'data': data_sample.gt_sem_seg_from.data
        }
    if hasattr(data_sample, 'gt_sem_seg_to') and data_sample.gt_sem_seg_to is not None:
        data_sample_dict['gt_sem_seg_to'] = {
            'data': data_sample.gt_sem_seg_to.data
        }
    
    
    sample = {
        'inputs': inputs,
        'data_samples': data_sample_dict
    }
    
    return sample


def train_process_single_sample(img_a_path, img_b_path, label_cd_path):
   
    data_info = {
        'img_a_path': img_a_path,
        'img_b_path': img_b_path,
        'label_cd': label_cd_path,
        'label_a': '**',
        'label_b': '**',
        'type': 'only_cd_label',
        'dataset_name': 'custom_dataset',
        'label_map': None,
        'format_seg_map': None,
        'reduce_zero_label': False,
        'seg_fields': []
    }
    
    pipeline = [
        MultiImgLoadImageFromFile_Modified(),
        MultiImgMultiAnnLoadAnnotations_Modified(),
        MultiImgPackSegInputs_Modified(),
    ]
    
    results = data_info.copy()
    
    for transform in pipeline:
        results = transform(results)
    
    inputs = results['inputs']  
    data_sample = results['data_samples']  
    
    
    data_sample_dict = {
        'metainfo': data_sample.metainfo if hasattr(data_sample, 'metainfo') else {},
        'gt_sem_seg': {
            'data': data_sample.gt_sem_seg.data if hasattr(data_sample, 'gt_sem_seg') and data_sample.gt_sem_seg is not None else None
        }
    }
    
    if hasattr(data_sample, 'gt_sem_seg_from') and data_sample.gt_sem_seg_from is not None:
        data_sample_dict['gt_sem_seg_from'] = {
            'data': data_sample.gt_sem_seg_from.data
        }
    if hasattr(data_sample, 'gt_sem_seg_to') and data_sample.gt_sem_seg_to is not None:
        data_sample_dict['gt_sem_seg_to'] = {
            'data': data_sample.gt_sem_seg_to.data
        }
    
    sample = {
        'inputs': inputs,
        'data_samples': data_sample_dict
    }
    
    return sample

