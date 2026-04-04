import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX,DEFAULT_CD_T1_TOKEN,DEFAULT_CD_T2_TOKEN)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN,DEFAULT_CD_T1_TOKEN,DEFAULT_CD_T2_TOKEN)
from .cd_dataset import LEVIR_CD_Dataset,S2LookingDataset,WHU_CD_Dataset,SECONDDataset
import json
from ARSeg.PytorchRSBuilding.dataset.rsfunction import test_process_single_sample

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1,dataset="LEVIR-CD+"
):
    rsinputs_list = []
    rsdata_samples_list = []
    imaget1_path_list = []
    imaget1_list = []
    imaget1_clip_list = []
    imaget2_path_list = []
    imaget2_list = []
    imaget2_clip_list = []
    conversation_list = []
    label1_list=[]
    label2_list=[]
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    
    for (
        rsinputs,
        rsdata_samples,
        imaget1_path,
        imaget1,
        imaget1_clip,
        imaget2_path,
        imaget2,
        imaget2_clip,  
        conversations,
        # masks,
        label1_path,
        label2_path,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        rsinputs_list.append(rsinputs)
        rsdata_samples_list.append(rsdata_samples)
        imaget1_path_list.append(imaget1_path)
        imaget1_list.append(imaget1)
        imaget1_clip_list.append(imaget1_clip)
        imaget2_path_list.append(imaget2_path)
        imaget2_list.append(imaget2)
        imaget2_clip_list.append(imaget2_clip)
        conversation_list.extend(conversations)
        label1_list.append(label1_path)
        label2_list.append(label2_path)
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <imaget1> and <imaget2> token
        for i in range(len(conversation_list)):
            replace_token1 = DEFAULT_CD_T1_TOKEN
            replace_token1 = (
                DEFAULT_IM_START_TOKEN + replace_token1 + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_CD_T1_TOKEN, replace_token1
            )
            
            replace_token2 = DEFAULT_CD_T2_TOKEN
            replace_token2 = (
                DEFAULT_IM_START_TOKEN + replace_token2 + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_CD_T2_TOKEN, replace_token2
            )
    
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()

    sep =" USER: " 
    for conversation, target in zip(conversation_list, targets):
        
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        
        rounds = conversation.split("</s>")
        
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_CD_T1_TOKEN in conversation or DEFAULT_CD_T2_TOKEN in conversation:
                
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            
            cur_len += round_len
        
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
    return {
        "rsinputs": torch.stack(rsinputs_list, dim=0),
        "rsdata_samples": rsdata_samples_list,
        "imaget1_paths": imaget1_path_list,
        "imaget1": torch.stack(imaget1_list, dim=0),
        "imaget1_clip": torch.stack(imaget1_clip_list, dim=0),
        "imaget2_paths": imaget2_path_list,
        "imaget2": torch.stack(imaget2_list, dim=0),
        "imaget2_clip": torch.stack(imaget2_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets, 
        "attention_masks": attention_masks, 
        "label1_list":label1_list,
        "label2_list":label2_list,
        "label_list": label_list, 
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="LEVIR-CD+",
        sample_rate=[1],
        levir_cd_data="LEVIR-CD+|train",
        s2looking_data="S2Looking|train",
        whu_data="WHU-CD|train",
        second_data="SECOND|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "LEVIR-CD+":
                self.all_datasets.append(
                    LEVIR_CD_Dataset(
                        base_image_dir, 
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        levir_cd_data,
                    )
                )
            elif dataset == "S2Looking":
                self.all_datasets.append(
                    S2LookingDataset(
                        base_image_dir, 
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        s2looking_data,
                    )
                )
            elif dataset == "WHU-CD":
                self.all_datasets.append(
                    WHU_CD_Dataset(
                        base_image_dir, 
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        whu_data,
                    )
                )
            elif dataset == "SECOND":
                self.all_datasets.append(
                    SECONDDataset(
                        base_image_dir, 
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        second_data,
                    )
                )
            

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        if val_dataset == "LEVIR-CD+":
            json_path=self.base_image_dir+"/"+"LEVIR-CD+"+"/"+"test"+"/"+"test.json"
            print(json_path)
            
            imagest1=[]
            imagest2=[]
            label1=[]
            label2=[]
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
                    label1.append(item['imaget1'])
                    label2.append(item['imaget2'])
                    labels.append(item['label'])
                    texts.append(item['text'])
            self.levir_cd_val_data = (imagest1, imagest2, label1, label2,labels, texts)
            self.data_type = "LEVIR-CD+"
            self.length=len(imagest1)
            ds="LEVIR-CD+"
        elif val_dataset == "S2Looking":
            json_path=self.base_image_dir+"/"+"S2Looking"+"/"+"test"+"/"+"test.json"
            print(json_path)
            
            imagest1=[]
            imagest2=[]
            label1=[]
            label2=[]
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
                    label1.append(item['imaget1'])
                    label2.append(item['imaget2'])
                    labels.append(item['label'])
                    texts.append(item['text'])
            self.levir_cd_val_data = (imagest1, imagest2, label1, label2,labels, texts)
            self.data_type = "S2Looking"
            self.length=len(imagest1)
            ds="S2Looking"
        elif val_dataset == "WHU-CD":
            json_path=self.base_image_dir+"/"+"WHU-CD"+"/"+"test"+"/"+"test.json"
            print(json_path)
            
            imagest1=[]
            imagest2=[]
            label1=[]
            label2=[]
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
                    label1.append(item['imaget1'])
                    label2.append(item['imaget2'])
                    labels.append(item['label'])
                    texts.append(item['text'])
            self.levir_cd_val_data = (imagest1, imagest2, label1, label2,labels, texts)
            self.data_type = "WHU-CD"
            self.length=len(imagest1)
            ds="WHU-CD"
        elif val_dataset == "SECOND":
            json_path=self.base_image_dir+"/"+"SECOND"+"/"+"test"+"/"+"test.json"
            print(json_path)
            
            imagest1=[]
            imagest2=[]
            label1=[]
            label2=[]
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
                    label1.append(item['label1'])
                    label2.append(item['label2'])
                    labels.append(item['label'])
                    texts.append(item['text'])
            self.levir_cd_val_data = (imagest1, imagest2, label1, label2, labels, texts)
            self.data_type = "SECOND"
            self.length=len(imagest1)
            ds="SECOND"
    
        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "LEVIR-CD+":
            return self.length
        elif self.data_type == "S2Looking":
            return self.length
        elif self.data_type == "WHU-CD":
            return self.length
        elif self.data_type == "SECOND":
            return self.length
        else:
            return len(self.images)

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
        
        imagest1,imagest2,label1,label2,labels,texts= self.levir_cd_val_data
        imaget1_path = imagest1[idx]
        imaget2_path = imagest2[idx]
        if label1 is not None and label2 is not None: 
            label1_path=label1[idx]
            label2_path=label2[idx]
        else:
            label1_path=None
            label2_path=None
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
        
        question="<imaget1> is the earlier remote sensing image, and <imaget2> is the later remote sensing image. Please segment the changed buildings in the image."
        questions=[]
        sampled_sents=[]
        questions.append(question)
        sampled_sents.append(question)
        inference = True
        
        test_sample = test_process_single_sample(imaget1_path, imaget2_path, label_path)
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
            None,
            None,
            inference,
        )