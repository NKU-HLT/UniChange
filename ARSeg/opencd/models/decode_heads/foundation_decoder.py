# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor
import einops

from mmseg.utils import ConfigType, SampleList
from mmseg.models.utils import resize

# from opencd.registry import MODELS
from mmengine.model import BaseModule, ModuleList, caffe2_xavier_init
from .upsample_decoder import UpsampleDecoder
from opencd.evaluation.metrics import compute_metrics_tools as CMT
from PIL import Image
import numpy as np
from mmdet.models.losses import CrossEntropyLoss
from .transformer import *
from opencd.models.losses import EdgeLoss

# 直接导入类，不使用注册方式
from opencd.models.necks.simple_fpn import SimpleFPN_det
from opencd.models.decode_heads.fusion_head import FusionHead
from mmdet.models.necks import FPN
import pdb



class Foundation_Decoder_v1(BaseModule):
   
    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 drop=0.0,
                 loss_type = 'BCELoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
                 fake_loss_weight=0,  # default 0, only use when pretraining
                 init_cfg=None,
                 finetune_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.loss_weight = loss_weight
        self.fake_loss_weight = fake_loss_weight
        self.align_corners=None
        self.channels = out_channels

        # position emb
        self.pe_emb = PositionEmbeddingRandom(out_channels // 2)
        # query emb
        self.building_a_embed = nn.Embedding(1, out_channels)
        self.building_b_embed = nn.Embedding(1, out_channels)
        self.cd_embed = nn.Embedding(1, out_channels)

        # decoder
        self.transformer_decoder = TwoWayTransformer_fpn(
                depth=4,
                embedding_dim=[out_channels] * 4,
                mlp_dim=2048,
                num_heads=8,
            )


        self.simple_fpn = SimpleFPN_det(
            backbone_channel=in_channels,
            in_channels=[in_channels//4, in_channels//2, in_channels, in_channels],
            out_channels=256,
            num_outs=4,
            norm_cfg=dict(type='LN2d', requires_grad=True)
        )

        
        self.fusion_ab = FusionHead(
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=[out_channels]*4,
        )
        self.fusion_cd = FusionHead(
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=[out_channels]*4,
        )

        # query head
        self.pred_head_ab = MLP(
            out_channels, out_channels, out_channels//4, 3, drop
        )
        self.pred_head_cd = MLP(
            out_channels, out_channels, out_channels//4, 3, drop
        )

        # upsample
        self.output_upscaling_ab = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )
        self.output_upscaling_cd = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss()
        elif loss_type == 'BCELoss':
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        self.finetune_cfg = finetune_cfg
        if self.finetune_cfg:
            if 'neck' not in self.finetune_cfg:
                for name, param in self.simple_fpn.named_parameters():
                    param.requires_grad = False
            if 'decoder' not in self.finetune_cfg:
                for name, param in self.transformer_decoder.named_parameters():
                    param.requires_grad = False
                for name, param in self.building_a_embed.named_parameters():
                    param.requires_grad = False       
                for name, param in self.building_b_embed.named_parameters():
                    param.requires_grad = False  
                for name, param in self.cd_embed.named_parameters():
                    param.requires_grad = False  
                for name, param in self.pe_emb.named_parameters():
                    param.requires_grad = False  
            if 'ab mask head' not in self.finetune_cfg:
                for name, param in self.fusion_ab.named_parameters():
                    param.requires_grad = False 
                for name, param in self.output_upscaling_ab.named_parameters():
                    param.requires_grad = False 
            if 'cd mask head' not in self.finetune_cfg:
                for name, param in self.fusion_cd.named_parameters():
                    param.requires_grad = False 
                for name, param in self.output_upscaling_cd.named_parameters():
                    param.requires_grad = False 
            if 'ab query head' not in self.finetune_cfg:
                for name, param in self.pred_head_ab.named_parameters():
                    param.requires_grad = False 
            if 'cd query head' not in self.finetune_cfg:
                for name, param in self.pred_head_cd.named_parameters():
                    param.requires_grad = False 

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head_cd, bias=0)
        caffe2_xavier_init(self.pred_head_ab, bias=0)
        caffe2_xavier_init(self.output_upscaling_cd, bias=0)
        caffe2_xavier_init(self.output_upscaling_ab, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_query_embed(self,seg_query,seg_query1,seg_query2,flag,b):
        
        # if self.type == 'only cd':
        #     # global_embed_cd = self.cd_embed.weight
        #     global_embed_cd = seg_query
        #     global_embed_a = global_embed_cd.new_zeros((1, self.channels))
        #     global_embed_b = global_embed_cd.new_zeros((1, self.channels))
        # elif self.type == 'only bx':
        #     global_embed_a = self.building_a_embed.weight
        #     global_embed_b = self.building_b_embed.weight
        #     global_embed_cd = global_embed_a.new_zeros((1, self.channels))
        # else:
        #     global_embed_a = self.building_a_embed.weight
        #     global_embed_b = self.building_b_embed.weight
        #     global_embed_cd = self.cd_embed.weight
        if flag=="CD":    
            global_embed_cd = seg_query
            global_embed_a = global_embed_cd.new_zeros((1, self.channels))
            global_embed_b = global_embed_cd.new_zeros((1, self.channels))
        elif flag=="SEG":
            global_embed_a = seg_query1
            global_embed_b = seg_query2
            global_embed_cd = global_embed_a.new_zeros((1, self.channels))
        query_embed = torch.cat((global_embed_a, global_embed_b, global_embed_cd), dim=0)
        return query_embed.unsqueeze(0).repeat(b, 1, 1)

    def forward(self, x, batch_data_samples, seg_query, seg_query1, seg_query2, flag, finetune_cfg):

        x = x[-1]
        x = self.simple_fpn(x)

        encoder_features = []
        pos_embed = []
        for i in range(len(x)):
            x_i = x[i]
            encoder_feature_1, encoder_feature_2 = torch.split(x_i, x_i.shape[0]//2, dim=0) 
            b, c, h, w = encoder_feature_1.shape
            
            # 沿batch切开，按token拼起来，[real_bs, token(2048), embed_num]
            encoder_feature = torch.cat((encoder_feature_1.flatten(2).permute(0,2,1), encoder_feature_2.flatten(2).permute(0,2,1)), dim=1)        
            encoder_features.append(encoder_feature)  

            pos_embed_i = self.pe_emb((h, w)).to(x_i.device)
            pos_embed_i = einops.rearrange(pos_embed_i, 'c h w -> (h w) c')
            pos_embed_i = einops.repeat(pos_embed_i, 'n c -> b n c', b=b)
            pos_embed_i = torch.cat((pos_embed_i, pos_embed_i), dim=1)
            pos_embed.append(pos_embed_i)
            # pos_embed[:, :pos_embed.shape[1]//2, :] = pos_embed[:, :pos_embed.shape[1]//2, :] + self.decoder_picture_embed[0]
            # pos_embed[:, pos_embed.shape[1]//2:, :] = pos_embed[:, pos_embed.shape[1]//2:, :] + self.decoder_picture_embed[1]

       
       
        query_embed = self.generate_query_embed(seg_query, seg_query1, seg_query2, flag, b)
        
        # encoder_features=encoder_features.to(torch.bfloat16)
        # pos_embed=pos_embed.to(torch.bfloat16)
        # query_embed=query_embed.to(torch.bfloat16)
        
        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features[::-1],
            image_pe=pos_embed[::-1],
            point_embedding=query_embed
        )
        a_embed, b_embed, cd_embed = torch.split(query_embed, [1,1,1], dim=1)

        a_embed = self.pred_head_ab(a_embed)
        b_embed = self.pred_head_ab(b_embed)
        cd_embed = self.pred_head_cd(cd_embed)

        img_a, img_b, img_cd = [], [], []
        for image_embedding in image_embeddings:
            img_a_i, img_b_i = torch.split(image_embedding, image_embedding.shape[1]//2, dim=1)
            b, n, c = img_a_i.shape
            img_a_i = img_a_i.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
            img_b_i = img_b_i.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
            img_cd_i = img_a_i - img_b_i
            img_a.append(img_a_i)
            img_b.append(img_b_i)
            img_cd.append(img_cd_i)

        img_a_fusion = self.fusion_ab(img_a[::-1])
        img_b_fusion = self.fusion_ab(img_b[::-1])
        mask_a = self.output_upscaling_ab(img_a_fusion)
        mask_b = self.output_upscaling_ab(img_b_fusion)
       
        img_cd_fusion = self.fusion_cd(img_cd[::-1])
        mask_cd = self.output_upscaling_cd(img_cd_fusion)    


        mask_a_logits = torch.einsum('bqc,bchw->bqhw',a_embed, mask_a)
        mask_b_logits = torch.einsum('bqc,bchw->bqhw',b_embed, mask_b)
        mask_cd_logits = torch.einsum('bqc,bchw->bqhw',cd_embed, mask_cd)
            

        logits = torch.cat((mask_a_logits, mask_b_logits, mask_cd_logits), dim=1)
        
        if isinstance(batch_data_samples[0], dict):
            target_size = batch_data_samples[0]['metainfo']['img_shape']
        else:
            target_size = batch_data_samples[0].metainfo['img_shape']
        logits = resize(
            input=logits,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners
            )

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType, finetune_cfg) -> dict:
        
        seg_logits = self.forward(inputs, batch_data_samples, finetune_cfg)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:

        seg_logits = self.forward(inputs, data_samples, None)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
       
        if isinstance(batch_data_samples[0], dict):
            
            sample0 = batch_data_samples[0]
            if 'type' in sample0:
                data_type = sample0['type']
            else:
                has_cd = 'gt_sem_seg' in sample0
                has_a = 'gt_sem_seg_from' in sample0
                has_b = 'gt_sem_seg_to' in sample0
                if has_cd and not has_a and not has_b:
                    data_type = 'only_cd_label'
                elif has_a and not has_cd and not has_b:
                    data_type = 'only_building_label'
                else:
                    data_type = 'all_label'

            get_gt = lambda field: [s[field]['data'] for s in batch_data_samples if field in s]
        else:
            data_type = batch_data_samples[0].type
            get_gt = lambda field: [getattr(s, field).data for s in batch_data_samples]
        a_logits, b_logits, cd_logits = torch.split(seg_logits, [1, 1, 1], dim=1)
        loss_dict = dict()
        if data_type == 'only_cd_label':
            gt_semantic_segs_cd = get_gt('gt_sem_seg')
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
            
            if not torch.is_grad_enabled():
                cd_pred = cd_logits.sigmoid()
                cd_pred = (cd_pred > 0.5).float()
                gt_cd_bin = (gt_cd > 0).float()
                loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(cd_pred, gt_cd_bin))
            
            loss_dict['loss_build_a'] = self.loss_weight[0] * seg_logits.new_zeros([1])
            loss_dict['loss_build_b'] = self.loss_weight[1] * seg_logits.new_zeros([1])
            loss_dict['loss_cd']      = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float())  
            
            a_pred = a_logits.sigmoid()
            b_pred = b_logits.sigmoid()
            fake_cd_Pred = torch.abs(a_pred - b_pred)
            loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

        elif data_type == 'only_building_label':
            gt_semantic_segs_a = get_gt('gt_sem_seg_from')
            if len(gt_semantic_segs_a) == 0:
                
                data_type = 'all_label'
            else:
                gt_a = torch.stack(gt_semantic_segs_a, dim=0)
            gt_cd = seg_logits.new_zeros(cd_logits.shape)
            
            if not torch.is_grad_enabled():
                a_pred = a_logits.sigmoid()
                a_pred = (a_pred > 0.5).float()
                gt_a_bin = (gt_a > 0).float()
                
                loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a_bin))
            gt_b = gt_a

            loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
            loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b.float())
            loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float()) * 0.1

            a_pred = a_logits.sigmoid()
            b_pred = b_logits.sigmoid()
            fake_cd_Pred = torch.abs(a_pred - b_pred)
            loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

        else: 
            gt_semantic_segs_cd = get_gt('gt_sem_seg')
            gt_semantic_segs_a = get_gt('gt_sem_seg_from')
            gt_semantic_segs_b = get_gt('gt_sem_seg_to')
            if len(gt_semantic_segs_a) == 0 or len(gt_semantic_segs_b) == 0 or len(gt_semantic_segs_cd) == 0:
                
                B, _, H, W = seg_logits.shape
                if len(gt_semantic_segs_a) == 0:
                    gt_semantic_segs_a = [seg_logits.new_zeros(1, H, W) for _ in range(B)]
                if len(gt_semantic_segs_b) == 0:
                    gt_semantic_segs_b = [seg_logits.new_zeros(1, H, W) for _ in range(B)]
                if len(gt_semantic_segs_cd) == 0:
                    gt_semantic_segs_cd = [seg_logits.new_zeros(1, H, W) for _ in range(B)]
            gt_a = torch.stack(gt_semantic_segs_a, dim=0)
            gt_b = torch.stack(gt_semantic_segs_b, dim=0)
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
            if not torch.is_grad_enabled():
                _pred = seg_logits.sigmoid()
                _pred = (_pred > 0.5).float()
                _gt = torch.cat((gt_a.unsqueeze(0), gt_b.unsqueeze(0), gt_cd.unsqueeze(0)), dim=0)
                _gt = (_gt > 0).float()
                loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(_pred, _gt))
            
            loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
            loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b.float())
            loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float())

            a_pred = a_logits.sigmoid()
            b_pred = b_logits.sigmoid()
            fake_cd_Pred = torch.abs(a_pred - b_pred)
            loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

        return loss_dict
    # def loss_by_feat(self, seg_logits: Tensor,
    #                  batch_data_samples: SampleList) -> dict:
    #     data_type = batch_data_samples[0].type
    #     a_logits, b_logits, cd_logits = torch.split(seg_logits, [1, 1, 1], dim=1)
    #     loss_dict = dict()
    #     if data_type == 'only_cd_label':
    #         gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
    #         gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
    #         cd_pred = cd_logits.sigmoid()
    #         cd_pred = (cd_pred > 0.5).float()
    #         gt_cd = (gt_cd > 0).float()
    #         loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(cd_pred, gt_cd))
            
    #         loss_dict['loss_build_a'] = self.loss_weight[0] * seg_logits.new_zeros([1])
    #         loss_dict['loss_build_b'] = self.loss_weight[1] * seg_logits.new_zeros([1])
    #         loss_dict['loss_cd']      = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float())  
            
    #         a_pred = a_logits.sigmoid()
    #         b_pred = b_logits.sigmoid()
    #         fake_cd_Pred = torch.abs(a_pred - b_pred)
    #         loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

    #     elif data_type == 'only_building_label':
    #         gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
    #         gt_a = torch.stack(gt_semantic_segs_a, dim=0)
    #         gt_cd = seg_logits.new_zeros(cd_logits.shape)
            
    #         a_pred = a_logits.sigmoid()
    #         a_pred = (a_pred > 0.5).float()
    #         gt_a = (gt_a > 0).float()
    #         gt_b = gt_a
            
    #         loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))

    #         loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
    #         loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b.float())
    #         loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float()) * 0.1

    #         a_pred = a_logits.sigmoid()
    #         b_pred = b_logits.sigmoid()
    #         fake_cd_Pred = torch.abs(a_pred - b_pred)
    #         loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

    #     else: 
    #         gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
    #         gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
    #         gt_semantic_segs_b = [data_sample.gt_sem_seg_to.data for data_sample in batch_data_samples]
    #         gt_a = torch.stack(gt_semantic_segs_a, dim=0)
    #         gt_b = torch.stack(gt_semantic_segs_b, dim=0)
    #         gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
    #         _pred = seg_logits.sigmoid()
    #         _pred = (_pred > 0.5).float()
    #         _gt = torch.cat((gt_a.unsqueeze(0), gt_b.unsqueeze(0), gt_cd.unsqueeze(0)), dim=0)
    #         _gt = (_gt > 0).float()
    #         loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(_pred, _gt))
            
    #         loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
    #         loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b.float())
    #         loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float())

    #         a_pred = a_logits.sigmoid()
    #         b_pred = b_logits.sigmoid()
    #         fake_cd_Pred = torch.abs(a_pred - b_pred)
    #         loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

    #     return loss_dict
    
    def predict_by_feat(self, seg_logits: Tensor,
                        data_samples: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        target_size = data_samples[0]['metainfo']['img_shape'] if isinstance(data_samples[0], dict) else data_samples[0].metainfo['img_shape']
        seg_logits = resize(
            input=seg_logits,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners
            )
        return seg_logits


class Foundation_Decoder_swin_v1(BaseModule):
   
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=256,
                 drop=0.0,
                 loss_type = 'BCELoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
                 fake_loss_weight=0,
                 init_cfg=None,
                 finetune_cfg=None):
        super().__init__(init_cfg)
        self.loss_weight = loss_weight
        self.fake_loss_weight = fake_loss_weight
        self.align_corners=None
        self.channels = out_channels

        self.pe_emb = PositionEmbeddingRandom(out_channels // 2)
        self.building_a_embed = nn.Embedding(1, out_channels)
        self.building_b_embed = nn.Embedding(1, out_channels)
        self.cd_embed = nn.Embedding(1, out_channels)
        self.transformer_decoder = TwoWayTransformer_fpn(
                depth=4,
                embedding_dim=[out_channels] * 4,
                mlp_dim=2048,
                num_heads=8,
            )

        # neck - 直接实例化FPN，不使用注册方式
        self.simple_fpn = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=4,
        )

        
        self.fusion_ab = FusionHead(
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=[out_channels]*4,
        )
        self.fusion_cd = FusionHead(
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=[out_channels]*4,
        )

        self.pred_head_ab = MLP(
            out_channels, out_channels, out_channels//4, 3, drop
        )
        self.pred_head_cd = MLP(
            out_channels, out_channels, out_channels//4, 3, drop
        )

        self.output_upscaling_ab = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )
        self.output_upscaling_cd = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )


        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss()
        elif loss_type == 'BCELoss':
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        self.finetune_cfg = finetune_cfg
        if self.finetune_cfg:
            if 'neck' not in self.finetune_cfg:
                for name, param in self.simple_fpn.named_parameters():
                    param.requires_grad = False
            if 'decoder' not in self.finetune_cfg:
                for name, param in self.transformer_decoder.named_parameters():
                    param.requires_grad = False
                for name, param in self.building_a_embed.named_parameters():
                    param.requires_grad = False       
                for name, param in self.building_b_embed.named_parameters():
                    param.requires_grad = False  
                for name, param in self.cd_embed.named_parameters():
                    param.requires_grad = False  
                for name, param in self.pe_emb.named_parameters():
                    param.requires_grad = False  
            if 'ab mask head' not in self.finetune_cfg:
                for name, param in self.fusion_ab.named_parameters():
                    param.requires_grad = False 
                for name, param in self.output_upscaling_ab.named_parameters():
                    param.requires_grad = False 
            if 'cd mask head' not in self.finetune_cfg:
                for name, param in self.fusion_cd.named_parameters():
                    param.requires_grad = False 
                for name, param in self.output_upscaling_cd.named_parameters():
                    param.requires_grad = False 
            if 'ab query head' not in self.finetune_cfg:
                for name, param in self.pred_head_ab.named_parameters():
                    param.requires_grad = False 
            if 'cd query head' not in self.finetune_cfg:
                for name, param in self.pred_head_cd.named_parameters():
                    param.requires_grad = False 


    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head_cd, bias=0)
        caffe2_xavier_init(self.pred_head_ab, bias=0)
        caffe2_xavier_init(self.output_upscaling_cd, bias=0)
        caffe2_xavier_init(self.output_upscaling_ab, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
       

        x = self.simple_fpn(x)
        encoder_features = []
        pos_embed = []

        for i in range(len(x)):
            x_i = x[i]
            encoder_feature_1, encoder_feature_2 = torch.split(x_i, x_i.shape[0]//2, dim=0) 
            b, c, h, w = encoder_feature_1.shape
            
           
            encoder_feature = torch.cat((encoder_feature_1.flatten(2).permute(0,2,1), encoder_feature_2.flatten(2).permute(0,2,1)), dim=1)        
            encoder_features.append(encoder_feature)  

            pos_embed_i = self.pe_emb((h, w)).to(x_i.device)
            pos_embed_i = einops.rearrange(pos_embed_i, 'c h w -> (h w) c')
            pos_embed_i = einops.repeat(pos_embed_i, 'n c -> b n c', b=b)
            pos_embed_i = torch.cat((pos_embed_i, pos_embed_i), dim=1)
            pos_embed.append(pos_embed_i)
            # pos_embed[:, :pos_embed.shape[1]//2, :] = pos_embed[:, :pos_embed.shape[1]//2, :] + self.decoder_picture_embed[0]
            # pos_embed[:, pos_embed.shape[1]//2:, :] = pos_embed[:, pos_embed.shape[1]//2:, :] + self.decoder_picture_embed[1]

        
        if self.type == 'only cd':
            global_embed_cd = self.cd_embed.weight
            global_embed_a = global_embed_cd.new_zeros((1, self.channels))
            global_embed_b = global_embed_cd.new_zeros((1, self.channels))
        elif self.type == 'only bx':
            global_embed_a = self.building_a_embed.weight
            global_embed_b = self.building_b_embed.weight
            global_embed_cd = global_embed_a.new_zeros((1, self.channels))
        else:
            global_embed_a = self.building_a_embed.weight
            global_embed_b = self.building_b_embed.weight
            global_embed_cd = self.cd_embed.weight
        query_embed = torch.cat((global_embed_a, global_embed_b, global_embed_cd), dim=0)
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features[::-1],
            image_pe=pos_embed[::-1],
            point_embedding=query_embed
        )

        a_embed, b_embed, cd_embed = torch.split(query_embed, [1,1,1], dim=1)
        a_embed = self.pred_head_ab(a_embed)
        b_embed = self.pred_head_ab(b_embed)
        cd_embed = self.pred_head_cd(cd_embed)

        img_a, img_b, img_cd = [], [], []
        for image_embedding in image_embeddings:
            img_a_i, img_b_i = torch.split(image_embedding, image_embedding.shape[1]//2, dim=1)
            b, n, c = img_a_i.shape
            img_a_i = img_a_i.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
            img_b_i = img_b_i.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
            img_cd_i = img_a_i - img_b_i
            img_a.append(img_a_i)
            img_b.append(img_b_i)
            img_cd.append(img_cd_i)


        img_a_fusion = self.fusion_ab(img_a[::-1])
        img_b_fusion = self.fusion_ab(img_b[::-1])
        img_cd_fusion = self.fusion_cd(img_cd[::-1])
        mask_a = self.output_upscaling_ab(img_a_fusion)
        mask_b = self.output_upscaling_ab(img_b_fusion)
        mask_cd = self.output_upscaling_cd(img_cd_fusion)

        mask_a_logits = torch.einsum('bqc,bchw->bqhw',a_embed, mask_a)
        mask_b_logits = torch.einsum('bqc,bchw->bqhw',b_embed, mask_b)
        mask_cd_logits = torch.einsum('bqc,bchw->bqhw',cd_embed, mask_cd)
            

        logits = torch.cat((mask_a_logits, mask_b_logits, mask_cd_logits), dim=1)
        logits = resize(
            input=logits,
            size=batch_data_samples[0]['metainfo']['img_shape'] if isinstance(batch_data_samples[0], dict) else batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType, finetune_cfg) -> dict:
        
        seg_logits = self.forward(inputs, batch_data_samples)
        with torch.cuda.amp.autocast(enabled=False):
            losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:
        
        seg_logits = self.forward(inputs, data_samples)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        data_type = batch_data_samples[0].type
        a_logits, b_logits, cd_logits = torch.split(seg_logits, [1, 1, 1], dim=1)
        loss_dict = dict()
        if data_type == 'only_cd_label':
            gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
            cd_pred = cd_logits.sigmoid()
            cd_pred = (cd_pred > 0.5).float()
            gt_cd = (gt_cd > 0).float()
            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(cd_pred, gt_cd))
            
            loss_dict['loss_build_a'] = self.loss_weight[0] * seg_logits.new_zeros([1])
            loss_dict['loss_build_b'] = self.loss_weight[1] * seg_logits.new_zeros([1])
            loss_dict['loss_cd']      = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float())  

            a_pred = a_logits.sigmoid()
            b_pred = b_logits.sigmoid()
            fake_cd_Pred = torch.abs(a_pred - b_pred)
            loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')


        elif data_type == 'only_building_label':
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
            gt_a = torch.stack(gt_semantic_segs_a, dim=0)
            gt_cd = seg_logits.new_zeros(cd_logits.shape)
            
            a_pred = a_logits.sigmoid()
            a_pred = (a_pred > 0.5).float()
            gt_a = (gt_a > 0).float()
            gt_b = gt_a
            
            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))

            loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
            loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b.float())
            loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float()) * 0.1

            a_pred = a_logits.sigmoid()
            b_pred = b_logits.sigmoid()
            fake_cd_Pred = torch.abs(a_pred - b_pred)
            loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

        else: 
            gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
            gt_semantic_segs_b = [data_sample.gt_sem_seg_to.data for data_sample in batch_data_samples]
            gt_a = torch.stack(gt_semantic_segs_a, dim=0)
            gt_b = torch.stack(gt_semantic_segs_b, dim=0)
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
            _pred = seg_logits.sigmoid()
            _pred = (_pred > 0.5).float()
            _gt = torch.cat((gt_a.unsqueeze(0), gt_b.unsqueeze(0), gt_cd.unsqueeze(0)), dim=0)
            _gt = (_gt > 0).float()
            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(_pred, _gt))
            
            loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
            loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b.float())
            loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd.float())

            a_pred = a_logits.sigmoid()
            b_pred = b_logits.sigmoid()
            fake_cd_Pred = torch.abs(a_pred - b_pred)
            loss_dict['loss_cd'] += self.fake_loss_weight * F.binary_cross_entropy(fake_cd_Pred, gt_cd.float(), reduction='mean')

        return loss_dict
    
    def predict_by_feat(self, seg_logits: Tensor,
                        data_samples: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = resize(
            input=seg_logits,
            size=data_samples[0]['metainfo']['img_shape'] if isinstance(data_samples[0], dict) else data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )
        return seg_logits

