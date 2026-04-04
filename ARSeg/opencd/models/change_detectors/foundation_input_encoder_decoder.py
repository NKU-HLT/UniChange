# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmengine.structures import PixelData
from torch import Tensor


from opencd.models.backbones.vit_sam_normal import ViTSAM_Normal

from opencd.models.decode_heads.foundation_decoder import Foundation_Decoder_v1

from opencd.models.data_preprocessor import FoundationInputSegDataPreProcessor

from opencd.models.utils.rewrite import resize, add_prefix



class FoundationEncoderDecoder(nn.Module):

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 finetune_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        
        super().__init__()
        
       
        if data_preprocessor is not None:
            if isinstance(data_preprocessor, dict) and data_preprocessor.get('type') == 'FoundationInputSegDataPreProcessor':
                
                data_preprocessor_config = data_preprocessor.copy()
                data_preprocessor_config.pop('type', None)  
                self.data_preprocessor = FoundationInputSegDataPreProcessor(**data_preprocessor_config)
            else:
                
                self.data_preprocessor = data_preprocessor
        else:
            
            self.data_preprocessor = None

        permitted_finetune_cfg = ['backbone', 'neck', 'decoder', 'ab mask head', 'cd mask head', 'ab query head', 'cd query head']
        if finetune_cfg is not None:
            for finetune_part in finetune_cfg:
                assert finetune_part in permitted_finetune_cfg
        self.finetune_cfg = finetune_cfg
        
        if isinstance(backbone, dict) and backbone.get('type') == 'ViTSAM_Normal':
           
            backbone_config = backbone.copy()
            backbone_config.pop('type', None)  
            self.backbone = ViTSAM_Normal(**backbone_config)
      
        
        if self.finetune_cfg:
            if 'backbone' not in self.finetune_cfg:
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = False
      
        if isinstance(decode_head, dict) and decode_head.get('type') == 'Foundation_Decoder_v1':
           
            decode_head_config = decode_head.copy()
            decode_head_config.pop('type', None)  
            decode_head_config['finetune_cfg'] = finetune_cfg
            self.decode_head = Foundation_Decoder_v1(**decode_head_config)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
       
        self.with_decode_head = True
        self.with_neck = neck is not None
        self.with_auxiliary_head = auxiliary_head is not None
        
        
        if neck is not None:
           
            self.neck = None  
        else:
            self.neck = None
            
       
        if auxiliary_head is not None:
            
            self.auxiliary_head = None  
        else:
            self.auxiliary_head = None

        assert self.with_decode_head

    def to(self, device):
        """Move model to device"""
        super().to(device)
        if self.data_preprocessor is not None:
            self.data_preprocessor = self.data_preprocessor.to(device)
        return self
    
    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)
        if self.data_preprocessor is not None:
            self.data_preprocessor.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        if self.data_preprocessor is not None:
            self.data_preprocessor.eval()
        return self

    # def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
    #     """Initialize ``auxiliary_head``"""
    #     if auxiliary_head is not None:
    #         if isinstance(auxiliary_head, list):
    #             self.auxiliary_head = nn.ModuleList()
    #             for head_cfg in auxiliary_head:
    #                 self.auxiliary_head.append(MODELS.build(head_cfg))
    #         else:
    #             self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        if inputs.shape[1] == 6:
            inputs_1, inputs_2 = torch.split(inputs, inputs.shape[1]//2, dim=1) 
        else:
            inputs_1, inputs_2 = inputs, inputs
        inputs = torch.cat((inputs_1, inputs_2), dim=0)
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
          
        return x

    def encode_decode(self, inputs: Tensor,
                      data_samples: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, data_samples,
                                              self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg, self.finetune_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs, data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self,
                inputs,
                data_samples=None):

        seg_logits = self.inference(inputs, data_samples)

        return self.postprocess_result(seg_logits, data_samples)

    def test_step(self, data):
        """Test step method to replace BaseSegmentor's test_step.
        
        Args:
            data (dict): A dict containing:
                - inputs (Tensor): Input images with shape (N, C, H, W)
                - data_samples (List[SegDataSample]): Data samples containing metainfo and gt_sem_seg
        
        Returns:
            List[SegDataSample]: Prediction results
        """
        # 先进行数据预处理，就像原项目的test_step一样
        if self.data_preprocessor is not None:
            data = self.data_preprocessor(data, False)
        
        inputs = data['inputs']
        data_samples = data['data_samples']
        
        # 使用现有的predict方法进行推理
        return self.predict(inputs, data_samples)

    def _forward(self,
                 inputs,
                 data_samples=None):
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x, data_samples)

    def inference(self, inputs: Tensor, data_samples: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logit = self.encode_decode(inputs, data_samples)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def postprocess_result(self,
                           seg_logits,
                           data_samples=None):
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                if isinstance(data_samples[i], dict):
                    img_meta = data_samples[i]['metainfo']
                else:
                    img_meta = data_samples[i].metainfo
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=None,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]


            i_seg_logits = i_seg_logits.sigmoid()
            i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)

            if isinstance(data_samples[i], dict):
                data_samples[i]['seg_logits'] = {'data': i_seg_logits}
                data_samples[i]['pred_sem_seg'] = {'data': i_seg_pred}
            # else:
            #     # SegDataSample格式：使用set_data方法
            #     data_samples[i].set_data({
            #         'seg_logits':
            #         PixelData(**{'data': i_seg_logits}),
            #         'pred_sem_seg':
            #         PixelData(**{'data': i_seg_pred})
            #     })

        return data_samples
