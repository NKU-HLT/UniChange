# geopixel rsbuilding融合dataset
import random
import cv2
import json
import numpy as np
from ixc_utils import R560_HD18_Identity_transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools import mask as M
import pdb
import torch
from rsfunction import test_process_single_sample

def conv2text(sources):
    END_HUMAN = '[UNUSED_TOKEN_145]\n'
    END_BOT = '[UNUSED_TOKEN_145]\n'
    conversation = ''

    for idx, sentence in enumerate(sources):
        BEGIN_SIGNAL = ''

        from_str = sentence['from']
        if from_str.lower() == 'human' or from_str.lower() == 'user':
            from_str = '[UNUSED_TOKEN_146]user\n'
            temp = (
                BEGIN_SIGNAL + from_str + sentence['value'].strip() +
                END_HUMAN)
        else:
            from_str = '[UNUSED_TOKEN_146]assistant\n'
            temp = (
                BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_BOT)
        conversation += temp

    return conversation + '</s>'


class ImageProcessorHD:

    def __init__(self, resolution=560, hd_num=18):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.resolution = resolution
        self.hd_num = hd_num
        print(f'hd_num = {self.hd_num}')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        item = Image.open(item).convert('RGB')
        return self.transform(
            R560_HD18_Identity_transform(
                item, resolution=self.resolution, hd_num=self.hd_num))


class Mix_dataset(Dataset):

    def __init__(self,
                json_datas,
                batch_size=1,
                local_rank=0,
                resolution=560,
                resolution_gr = 1024,
                hd_num=18):
        """vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file."""
        super().__init__()
        print(f'initializing mix data at rank {local_rank}')
        self.datasets_text, self.datasets_multi, self.datasets_grounding = [], [], []
        self.data_num_text, self.data_num_multi, self.data_num_grounding = [], [], []

        self.batch_size = batch_size
        self.set_seed = False
        self.local_rank = local_rank
        
        for _, d in json_datas.items():# 只循环一次
            
            has_img =  'image1' in d[0].keys()
            has_mask = ('polygons' in d[0].keys()) or ('segmentation' in d[0].keys()) or ('label' in d[0].keys())

            sub_data_set = Sample_dataset(
                d, 
                batch_size, 
                has_img=has_img,
                has_mask=has_mask,
                resolution=resolution, 
                resolution_gr=resolution_gr, 
                hd_num=hd_num
            )
            
            
            if has_img:
                if has_mask:
                    self.datasets_grounding.append(sub_data_set)
                    self.data_num_grounding.append(len(sub_data_set))
                else:
                    self.datasets_multi.append(sub_data_set)
                    self.data_num_multi.append(len(sub_data_set))
            else:
                self.datasets_text.append(sub_data_set)
                self.data_num_text.append(len(sub_data_set))
        
        self.data_ratio_grounding = [
            float(ratio) / sum(self.data_num_grounding)
            for ratio in self.data_num_grounding
        ]
        self.data_ratio_multi = [
            float(ratio) / sum(self.data_num_multi)
            for ratio in self.data_num_multi
        ]
        self.data_ratio_text = [
            float(ratio) / sum(self.data_num_text)
            for ratio in self.data_num_text
        ]
        self.data_num = np.sum(self.data_num_grounding) + np.sum(self.data_num_multi) + np.sum(self.data_num_text)
        self.num_of_ds =sum(1 for dataset in [self.datasets_text, self.datasets_multi, self.datasets_grounding] if dataset)
        self.use_grounding = 0
        self.use_multi = batch_size*(self.num_of_ds-1)  

    def __len__(self):
        return int(self.data_num / self.batch_size)

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(index)
            self.set_seed = True
            print(f'Set seed {index} for rank {self.local_rank}')
        
       
        if len(self.datasets_grounding) == 0 and len(self.datasets_multi) == 0 and len(self.datasets_text) == 0:
            raise ValueError(
                'All _grounding, _multi and _text are empty. Cannot sample any data.')
        
        if len(self.datasets_grounding) > 0 and (self.use_grounding < self.batch_size
                                             or ( len(self.datasets_multi) == 0 and len(self.datasets_text) == 0 )):
            data_idx = random.choices(
                range(len(self.data_ratio_grounding)),
                weights=self.data_ratio_grounding,
                k=1)[0]
            sample = self.datasets_grounding[data_idx].get_item()
        elif len(self.datasets_multi) > 0 and (self.use_multi < self.batch_size
                                             or len(self.datasets_text) == 0):
            data_idx = random.choices(
                range(len(self.data_ratio_multi)),
                weights=self.data_ratio_multi,
                k=1)[0]
            sample = self.datasets_multi[data_idx].get_item()
        elif len(self.datasets_text) > 0:
            data_idx = random.choices(
                range(len(self.data_ratio_text)),
                weights=self.data_ratio_text,
                k=1)[0]
            sample = self.datasets_text[data_idx].get_item()
        else:
            raise ValueError('Unable to select a dataset for sampling.')
        
        self.use_grounding += 1
        self.use_multi += 1
        if self.use_grounding == self.batch_size * self.num_of_ds:
            self.use_grounding = 0
        if self.use_multi == self.batch_size * self.num_of_ds:
            self.use_multi = 0
       
        return dict(samples=sample)


class Sample_dataset(Dataset):

    def __init__(self,
                 raw_data,
                 batch_size,
                 has_img=False,
                 has_mask=False,
                 resolution=560,
                 resolution_gr = 1024,
                 hd_num=18):
        self.raw_data = raw_data
        print(f'initilized Sample_dataset with {len(self.raw_data)}')
        self.batch_size = batch_size
        self.vis_processor = ImageProcessorHD(
            resolution=resolution, hd_num=hd_num)
        # self.vis_processor_gr = SAM2Transforms(
        #     resolution=resolution_gr,mask_threshold=0.0,max_hole_area=0.0,max_sprinkle_area=0.0)
        self.text_processor = conv2text
        self.has_img = has_img
        self.has_mask = has_mask

    def __len__(self):
        return len(self.raw_data)

    def __get_item__(self, i): # 处理单个数据
        conv_text = conv2text(self.raw_data[i]['conversations'])
        sample = dict(text_input=conv_text, )
        if self.has_img:
            image1_file = self.raw_data[i]['image1']
           
            if type(image1_file) == str:
                image1 = self.vis_processor(image1_file) # ImageProcessorHD
            elif type(image1_file) == list:
                image1 = [self.vis_processor(i) for i in image1_file] 
            else:
                raise NotImplementedError('Image format not supported')
            image2_file = self.raw_data[i]['image2']
           
            if type(image2_file) == str:
                image2 = self.vis_processor(image2_file) # ImageProcessorHD
            elif type(image2_file) == list:
                image2 = [self.vis_processor(i) for i in image2_file] 
            else:
                raise NotImplementedError('Image format not supported')
            sample['image1'] = image1
            sample['image2'] = image2
            if self.has_mask:
                assert isinstance(image1_file, str), "image_file must be a string" #need single image
                image1_g = Image.open(image1_file).convert("RGB") 
                image2_g = Image.open(image2_file).convert("RGB") 
                w, h = image1_g.size
                ori_hw = (h, w)
                
                
                if 'label' in self.raw_data[i]:
                    label_path=self.raw_data[i]['label']
                    # masks = []
                    binary_image_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    mask = (binary_image_np > 0).astype(np.float32)
                    masks = mask[np.newaxis, :, :]
                    # masks.append(mask)
                    # mask = torch.from_numpy(masks_np)
                if 'polygons' in self.raw_data[i]:
                    polygons_file = self.raw_data[i]['polygons']
                    assert isinstance(polygons_file, str), "polygons_file must be a string"
                    with open(polygons_file, 'r') as file:
                        try:
                            data = json.load(file)  
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON file: {polygons_file}")
                    # Processing the polygons data
                    masks = []
                    for polygon in data["polygons"]:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for poly in polygon:
                            assert len(poly) > 0 and len(poly[0]) == 2, "invalid multiple polygons"
                            cv2.fillPoly(mask, np.array([poly], dtype=np.int32), color=1)
                        masks.append(mask)
                    assert len(masks) == conv_text.count('[SEG]') , f"number of grounding tokens are not equal to number of masks provided with image: {image1_file}"

                elif 'segmentation' in self.raw_data[i]: ####不执行

                    segm = self.raw_data[i]['segmentation']
                    assert len(segm) == conv_text.count('[SEG]') , f"number of grounding tokens are not equal to number of masks provided with image: {image1_file}"
                    masks = []
                    if segm is None:
                        raise ValueError(f"Failed to read mask")
                    for rle in segm:
                        binary_mask = M.decode(rle).astype(np.uint8)
                        masks.append(binary_mask)
                # else:
                #     print(f"No 'polygon' or 'segmentation' found in grounding data")
                
                sample['image1_g'] = None
                sample['image2_g'] = None
                sample['ori_hw'] = ori_hw
                sample['masks'] = masks 
            else: 
                sample['image1_g'] = None
                sample['image2_g'] = None
                sample['ori_hw'] = None
                sample['masks'] = None
        else:
            sample['image1'] = None
            sample['image2'] = None
        
        
        test_sample = test_process_single_sample(self.raw_data[i]['image1'], self.raw_data[i]['image2'], self.raw_data[i]['label'])
        sample['inputs'] = test_sample['inputs']
        sample['data_samples'] = test_sample['data_samples']
        
       
        return sample

    def get_item(self, ): 
        text_input, image1, image2, image1_g, image2_g, masks, ori_hw = [], [], [], [], [], [], []
        inputs_batch, data_samples_batch = [], []  

        for i in range(self.batch_size):
            idx = random.randrange(len(self.raw_data))
            sample = self.__get_item__(idx)
            text_input.append(sample['text_input'])
            
            
            inputs_batch.append(sample['inputs'])
            data_samples_batch.append(sample['data_samples'])

            if sample['image1'] is None:
                pass
            else:
                images1_batch = []       # list of 1xCxHxW
                images2_batch = []       # list of 1xCxHxW
                if type(sample['image1']) is list:
                    for im in sample['image1']:
                        images1_batch.append(im.unsqueeze(0))
                    for im in sample['image2']:
                        images2_batch.append(im.unsqueeze(0))
                else:
                    images1_batch.append(sample['image1'].unsqueeze(0))
                    images2_batch.append(sample['image2'].unsqueeze(0))
                    if sample['image1_g'] is None:
                        pass
                    else:
                        image1_g.append(sample['image1_g'].unsqueeze(0))
                        image2_g.append(sample['image2_g'].unsqueeze(0))
                        masks.append(sample['masks'])
                        ori_hw.append(sample['ori_hw'])
                image1.append(images1_batch)
                image2.append(images2_batch)
        if self.has_mask:
            data_type = 'grounding' 
        elif self.has_img : 
            data_type = 'multi' 
        else:
            data_type = 'text'
        sample = {
            'text_input': text_input,
            'data_type': data_type,
            'inputs': inputs_batch,  
            'data_samples': data_samples_batch  
        }
        if self.has_img:
            sample['image1'] = image1
            sample['image2'] = image2
        if self.has_mask:
            sample['image1_g'] = None
            sample['image2_g'] = None
            sample['ori_hw'] = ori_hw
            sample['masks'] = masks 
       
        return sample 


class Val_Mix_dataset(Dataset):

    def __init__(self,
                json_datas,
                batch_size=1,
                local_rank=0,
                resolution=560,
                resolution_gr = 1024,
                hd_num=18):
        """vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file."""
        super().__init__()
        print(f'initializing mix data at rank {local_rank}')
        self.datasets_text, self.datasets_multi, self.datasets_grounding = [], [], []
        self.data_num_text, self.data_num_multi, self.data_num_grounding = [], [], []

        self.batch_size = batch_size
        self.set_seed = False
        self.local_rank = local_rank
        
        for _, d in json_datas.items():# 只循环一次
            
            has_img =  'image1' in d[0].keys()
            has_mask = ('polygons' in d[0].keys()) or ('segmentation' in d[0].keys()) or ('label' in d[0].keys())

            self.sub_data_set = Val_Sample_dataset(
                d, 
                batch_size, 
                has_img=has_img,
                has_mask=has_mask,
                resolution=resolution, 
                resolution_gr=resolution_gr, 
                hd_num=hd_num
            )
            
            
            if has_img:
                if has_mask:
                    self.datasets_grounding.append(self.sub_data_set)
                    self.data_num_grounding.append(len(self.sub_data_set))
                else:
                    self.datasets_multi.append(self.sub_data_set)
                    self.data_num_multi.append(len(self.sub_data_set))
            else:
                self.datasets_text.append(self.sub_data_set)
                self.data_num_text.append(len(self.sub_data_set))
        
        self.data_ratio_grounding = [
            float(ratio) / sum(self.data_num_grounding)
            for ratio in self.data_num_grounding
        ]
        self.data_ratio_multi = [
            float(ratio) / sum(self.data_num_multi)
            for ratio in self.data_num_multi
        ]
        self.data_ratio_text = [
            float(ratio) / sum(self.data_num_text)
            for ratio in self.data_num_text
        ]
        self.data_num = np.sum(self.data_num_grounding) + np.sum(self.data_num_multi) + np.sum(self.data_num_text)
        self.num_of_ds =sum(1 for dataset in [self.datasets_text, self.datasets_multi, self.datasets_grounding] if dataset)
        self.use_grounding = 0
        self.use_multi = batch_size*(self.num_of_ds-1)  

    def __len__(self):
        return int(self.data_num / self.batch_size)

    def __getitem__(self, index):
        
        sample = self.sub_data_set.get_item(index)
        
        self.use_grounding += 1
        self.use_multi += 1
        if self.use_grounding == self.batch_size * self.num_of_ds:
            self.use_grounding = 0
        if self.use_multi == self.batch_size * self.num_of_ds:
            self.use_multi = 0
       
        return dict(samples=sample)
    
class Val_Sample_dataset(Dataset):

    def __init__(self,
                 raw_data,
                 batch_size,
                 has_img=False,
                 has_mask=False,
                 resolution=560,
                 resolution_gr = 1024,
                 hd_num=18):
        self.raw_data = raw_data
        print(f'initilized Sample_dataset with {len(self.raw_data)}')
        self.batch_size = batch_size
        self.vis_processor = ImageProcessorHD(
            resolution=resolution, hd_num=hd_num)
        # self.vis_processor_gr = SAM2Transforms(
        #     resolution=resolution_gr,mask_threshold=0.0,max_hole_area=0.0,max_sprinkle_area=0.0)
        self.text_processor = conv2text
        self.has_img = has_img
        self.has_mask = has_mask

    def __len__(self):
        return len(self.raw_data)

    def __get_item__(self, i): # 处理单个数据
        conv_text = conv2text(self.raw_data[i]['conversations'])
        label_file=self.raw_data[i]['label']
        sample = dict(text_input=conv_text, )
        sample['label']=label_file
        if self.has_img:
            image1_file = self.raw_data[i]['image1']
           
            if type(image1_file) == str:
                image1 = self.vis_processor(image1_file) # ImageProcessorHD
            elif type(image1_file) == list:
                image1 = [self.vis_processor(i) for i in image1_file] 
            else:
                raise NotImplementedError('Image format not supported')
            image2_file = self.raw_data[i]['image2']
           
            if type(image2_file) == str:
                image2 = self.vis_processor(image2_file) # ImageProcessorHD
            elif type(image2_file) == list:
                image2 = [self.vis_processor(i) for i in image2_file] 
            else:
                raise NotImplementedError('Image format not supported')
            sample['image1'] = image1
            sample['image2'] = image2
            if self.has_mask:
                assert isinstance(image1_file, str), "image_file must be a string" #need single image
                image1_g = Image.open(image1_file).convert("RGB") 
                image2_g = Image.open(image2_file).convert("RGB") 
                w, h = image1_g.size
                ori_hw = (h, w)
                
                
                if 'label' in self.raw_data[i]:
                    label_path=self.raw_data[i]['label']
                    # masks = []
                    binary_image_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    mask = (binary_image_np > 0).astype(np.float32)
                    masks = mask[np.newaxis, :, :]
                if 'polygons' in self.raw_data[i]:
                    polygons_file = self.raw_data[i]['polygons']
                    assert isinstance(polygons_file, str), "polygons_file must be a string"
                    with open(polygons_file, 'r') as file:
                        try:
                            data = json.load(file)  
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON file: {polygons_file}")
                    # Processing the polygons data
                    masks = []
                    for polygon in data["polygons"]:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for poly in polygon:
                            assert len(poly) > 0 and len(poly[0]) == 2, "invalid multiple polygons"
                            cv2.fillPoly(mask, np.array([poly], dtype=np.int32), color=1)
                        masks.append(mask)
                        # masks.append(torch.from_numpy(mask).to(dtype=torch.float32))
                   
                    assert len(masks) == conv_text.count('[SEG]') , f"number of grounding tokens are not equal to number of masks provided with image: {image_file}"

                elif 'segmentation' in self.raw_data[i]: ####不执行

                    segm = self.raw_data[i]['segmentation']
                    assert len(segm) == conv_text.count('[SEG]') , f"number of grounding tokens are not equal to number of masks provided with image: {image_file}"
                    masks = []
                    if segm is None:
                        raise ValueError(f"Failed to read mask")
                    for rle in segm:
                        binary_mask = M.decode(rle).astype(np.uint8)
                        masks.append(binary_mask)
                        # masks.append(torch.from_numpy(binary_mask).to(dtype=torch.float32))
                # else:
                #     print(f"No 'polygon' or 'segmentation' found in grounding data")
                sample['image1_g'] = None
                sample['image2_g'] = None
                sample['ori_hw'] = ori_hw
                sample['masks'] = masks 
            else: 
                sample['image1_g'] = None
                sample['image2_g'] = None
                sample['ori_hw'] = None
                sample['masks'] = None
        else:
            sample['image1'] = None
            sample['image2'] = None
        test_sample = test_process_single_sample(self.raw_data[i]['image1'], self.raw_data[i]['image2'], self.raw_data[i]['label'])
        sample['inputs'] = test_sample['inputs']
        sample['data_samples'] = test_sample['data_samples']
        return sample

    def get_item(self, index): 
        text_input, label, image1, image2, image1_g, image2_g, masks, ori_hw = [], [], [], [], [], [], [], []
        inputs_batch, data_samples_batch = [], []  

        for i in range(self.batch_size):
            sample = self.__get_item__(index)
            text_input.append(sample['text_input'])
            label.append(sample['label'])
            
            
            inputs_batch.append(sample['inputs'])
            data_samples_batch.append(sample['data_samples'])

            if sample['image1'] is None:
                pass
            else:
                images1_batch = []       # list of 1xCxHxW
                images2_batch = []       # list of 1xCxHxW
                if type(sample['image1']) is list:
                    for im in sample['image1']:
                        images1_batch.append(im.unsqueeze(0))
                    for im in sample['image2']:
                        images2_batch.append(im.unsqueeze(0))
                else:
                    images1_batch.append(sample['image1'].unsqueeze(0))
                    images2_batch.append(sample['image2'].unsqueeze(0))
                    if sample['image1_g'] is None:
                        pass
                    else:
                        image1_g.append(sample['image1_g'].unsqueeze(0))
                        image2_g.append(sample['image2_g'].unsqueeze(0))
                        masks.append(sample['masks'])
                        ori_hw.append(sample['ori_hw'])
                image1.append(images1_batch)
                image2.append(images2_batch)
        if self.has_mask:
            data_type = 'grounding' 
        elif self.has_img : 
            data_type = 'multi' 
        else:
            data_type = 'text'
        sample = {
            'text_input': text_input,
            'data_type': data_type,
            'label': label,
            'inputs': inputs_batch,  
            'data_samples': data_samples_batch  
        }
        if self.has_img:
            sample['image1'] = image1
            sample['image2'] = image2
        if self.has_mask:
            sample['image1_g'] = None
            sample['image2_g'] = None
            sample['ori_hw'] = ori_hw
            sample['masks'] = masks 
       
        return sample 