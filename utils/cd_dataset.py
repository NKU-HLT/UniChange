import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

from ARSeg.PytorchRSBuilding.dataset.rsfunction import train_process_single_sample


class LEVIR_CD_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir, 
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        levir_cd_data="LEVIR-CD+|train",
    ):
        self.exclude_val = exclude_val
        self.levir_cd_data = levir_cd_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        levir_cd_data, splits= self.levir_cd_data.split("|")
        
        json_path=self.base_image_dir+"/"+"LEVIR-CD+/train/train.json"
        
        imagest1=[]
        imagest2=[]
        labels=[]
        texts=[]
        
        
        if not os.path.exists(json_path):
            print(f"Error：JSON file not exist: {json_path}")
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            
            for item in data:
                imagest1.append(item['imaget1'])
                imagest2.append(item['imaget2'])
                labels.append(item['label'])
                texts.append(item['text'])
        self.levir_cd_data = (imagest1, imagest2, labels, texts)

    
    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        imagest1,imagest2,labels,texts= self.levir_cd_data
        idx = random.randint(0, len(imagest1) - 1)
        imaget1_path = imagest1[idx]
        imaget2_path = imagest2[idx]
        label_path=labels[idx]
        text=texts[idx]

        imaget1 = cv2.imread(imaget1_path)
        imaget1 = cv2.cvtColor(imaget1, cv2.COLOR_BGR2RGB)
        ori_size = imaget1.shape[:2]
        # preprocess image for clip
        # (C*H*W)
        imaget1_clip = self.clip_image_processor.preprocess(imaget1, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget1 = self.transform.apply_image(imaget1)  # preprocess image for sam
        resize = imaget1.shape[:2]
        imaget1 = self.preprocess(torch.from_numpy(imaget1).permute(2, 0, 1).contiguous())
        
        imaget2 = cv2.imread(imaget2_path)
        imaget2 = cv2.cvtColor(imaget2, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        # (C*H*W)
        imaget2_clip = self.clip_image_processor.preprocess(imaget2, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget2 = self.transform.apply_image(imaget2)  # preprocess image for sam
        imaget2 = self.preprocess(torch.from_numpy(imaget2).permute(2, 0, 1).contiguous())
        
        label = torch.ones(ori_size) * self.ignore_label

        conversations=[]
        conversations.append(text)
        
        question=None
        questions=[]
        sampled_sents=[]
        questions.append(question)
        sampled_sents.append(question)
        
        test_sample = train_process_single_sample(imaget1_path, imaget2_path, label_path)
        rsinputs=test_sample['inputs']
        rsdata_samples=test_sample['data_samples']
        
        
        
        return (
            rsinputs,
            rsdata_samples,
            imaget1_path,
            imaget1,
            imaget1_clip,
            imaget2_path,
            imaget2,
            imaget2_clip,   
            conversations,
            None,
            None,
            label,
            resize,
            questions,
            sampled_sents,
        )
        
class S2LookingDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        s2looking_data="S2Looking|train",
    ):
        self.exclude_val = exclude_val
        self.s2looking_data = s2looking_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        json_path=self.base_image_dir+"/"+"S2Looking/train/train.json"
        print(json_path)
        
        imagest1=[]
        imagest2=[]
        labels=[]
        texts=[]
        
        
        if not os.path.exists(json_path):
            print(f"Error：JSON file not exist: {json_path}")
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

           
            for item in data:
                imagest1.append(item['imaget1'])
                imagest2.append(item['imaget2'])
                labels.append(item['label'])
                texts.append(item['text'])
        self.levir_cd_data = (imagest1, imagest2, labels, texts)

    
    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        
        imagest1,imagest2,labels,texts= self.levir_cd_data
        idx = random.randint(0, len(imagest1) - 1)
        imaget1_path = imagest1[idx]
        imaget2_path = imagest2[idx]
        label_path=labels[idx]
        text=texts[idx]

        imaget1 = cv2.imread(imaget1_path)
        imaget1 = cv2.cvtColor(imaget1, cv2.COLOR_BGR2RGB)
        ori_size = imaget1.shape[:2]
        # preprocess image for clip
        # (C*H*W)
        imaget1_clip = self.clip_image_processor.preprocess(imaget1, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget1 = self.transform.apply_image(imaget1)  # preprocess image for sam
        resize = imaget1.shape[:2]
        imaget1 = self.preprocess(torch.from_numpy(imaget1).permute(2, 0, 1).contiguous())
        
        imaget2 = cv2.imread(imaget2_path)
        imaget2 = cv2.cvtColor(imaget2, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        # (C*H*W)
        imaget2_clip = self.clip_image_processor.preprocess(imaget2, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget2 = self.transform.apply_image(imaget2)  # preprocess image for sam
        imaget2 = self.preprocess(torch.from_numpy(imaget2).permute(2, 0, 1).contiguous())
        
        label = torch.ones(ori_size) * self.ignore_label

        conversations=[]
        conversations.append(text)
         
        question=None
        questions=[]
        sampled_sents=[]
        questions.append(question)
        sampled_sents.append(question)
        
        test_sample = train_process_single_sample(imaget1_path, imaget2_path, label_path)
        rsinputs=test_sample['inputs']
        rsdata_samples=test_sample['data_samples']
        
        
        return (
            rsinputs,
            rsdata_samples,
            imaget1_path,
            imaget1,
            imaget1_clip,
            imaget2_path,
            imaget2,
            imaget2_clip,   
            conversations,
            None,
            None,
            label,
            resize,
            questions,
            sampled_sents,
        )
        
        
class WHU_CD_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir, 
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        whu_data="WHU|train",
    ):
        self.exclude_val = exclude_val
        self.whu_data = whu_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        json_path=self.base_image_dir+"/"+"WHU-CD/train/train.json"
        print(json_path)
        
        imagest1=[]
        imagest2=[]
        labels=[]
        texts=[]
        
        
        if not os.path.exists(json_path):
            print(f"Error：JSON file not exist: {json_path}")
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

           
            for item in data:
                imagest1.append(item['imaget1'])
                imagest2.append(item['imaget2'])
                labels.append(item['label'])
                texts.append(item['text'])
        self.levir_cd_data = (imagest1, imagest2, labels, texts)

    
    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        
        imagest1,imagest2,labels,texts= self.levir_cd_data
        idx = random.randint(0, len(imagest1) - 1)
        imaget1_path = imagest1[idx]
        imaget2_path = imagest2[idx]
        label_path=labels[idx]
        text=texts[idx]

        imaget1 = cv2.imread(imaget1_path)
        imaget1 = cv2.cvtColor(imaget1, cv2.COLOR_BGR2RGB)
        ori_size = imaget1.shape[:2]
        # preprocess image for clip
        # (C*H*W)
        imaget1_clip = self.clip_image_processor.preprocess(imaget1, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget1 = self.transform.apply_image(imaget1)  # preprocess image for sam
        resize = imaget1.shape[:2]
        imaget1 = self.preprocess(torch.from_numpy(imaget1).permute(2, 0, 1).contiguous())
        
        imaget2 = cv2.imread(imaget2_path)
        imaget2 = cv2.cvtColor(imaget2, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        # (C*H*W)
        imaget2_clip = self.clip_image_processor.preprocess(imaget2, return_tensors="pt")[
            "pixel_values"
        ][0]
        imaget2 = self.transform.apply_image(imaget2)  # preprocess image for sam
        imaget2 = self.preprocess(torch.from_numpy(imaget2).permute(2, 0, 1).contiguous())
        
        label = torch.ones(ori_size) * self.ignore_label

        conversations=[]
        conversations.append(text)
        
        
        question=None
        questions=[]
        sampled_sents=[]
        questions.append(question)
        sampled_sents.append(question)
        
        test_sample = train_process_single_sample(imaget1_path, imaget2_path, label_path)
        rsinputs=test_sample['inputs']
        rsdata_samples=test_sample['data_samples']
        
        
        return (
            rsinputs,
            rsdata_samples,
            imaget1_path,
            imaget1,
            imaget1_clip,
            imaget2_path,
            imaget2,
            imaget2_clip,   
            conversations,
            None,
            None,
            label,
            resize,
            questions,
            sampled_sents,
        )
        
class SECONDDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir, 
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        second_data="SECOND|train",
    ):
        self.exclude_val = exclude_val
        self.second_data = second_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        json_path=self.base_image_dir+"/"+"SECOND/train/train.json"
        
        print(json_path)
        
        imagest1=[]
        imagest2=[]
        labels=[]
        label1=[]
        label2=[]
        texts=[]
        
        
        if not os.path.exists(json_path):
            print(f"Error：JSON file not exist: {json_path}")
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

           
            for item in data:
                imagest1.append(item['imaget1'])
                imagest2.append(item['imaget2'])
                labels.append(item['label'])
                label1.append(item['label1'])
                label2.append(item['label2'])
                texts.append(item['text'])
        self.levir_cd_data = (imagest1, imagest2, label1, label2, labels,  texts)

    
    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        
        imagest1,imagest2,label1,label2,labels,texts= self.levir_cd_data
        idx = random.randint(0, len(imagest1) - 1)
        imaget1_path = imagest1[idx]
        imaget2_path = imagest2[idx]
        label1_path=label1[idx]
        label2_path=label2[idx]
        label_path=labels[idx]
        text=texts[idx]

        imaget1 = cv2.imread(imaget1_path)
        imaget1 = cv2.cvtColor(imaget1, cv2.COLOR_BGR2RGB)
        ori_size = imaget1.shape[:2]
        # preprocess image for clip
        # (C*H*W)
        imaget1_clip = self.clip_image_processor.preprocess(imaget1, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget1 = self.transform.apply_image(imaget1)  # preprocess image for sam
        resize = imaget1.shape[:2]
        imaget1 = self.preprocess(torch.from_numpy(imaget1).permute(2, 0, 1).contiguous())
        
        imaget2 = cv2.imread(imaget2_path)
        imaget2 = cv2.cvtColor(imaget2, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        # (C*H*W)
        imaget2_clip = self.clip_image_processor.preprocess(imaget2, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        imaget2 = self.transform.apply_image(imaget2)  # preprocess image for sam
        imaget2 = self.preprocess(torch.from_numpy(imaget2).permute(2, 0, 1).contiguous())
        
        label = torch.ones(ori_size) * self.ignore_label

        conversations=[]
        conversations.append(text)
        
        
        question=None
        questions=[]
        sampled_sents=[]
        questions.append(question)
        sampled_sents.append(question)
        
        test_sample = train_process_single_sample(imaget1_path, imaget2_path, label_path)
        rsinputs=test_sample['inputs']
        rsdata_samples=test_sample['data_samples']
        
        
        return (
            rsinputs,
            rsdata_samples,
            imaget1_path,
            imaget1,
            imaget1_clip,
            imaget2_path,
            imaget2,
            imaget2_clip,   
            conversations,
            label1_path,
            label2_path,
            label,
            resize,
            questions,
            sampled_sents,
        )