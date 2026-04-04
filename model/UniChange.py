from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h,build_cd_sam_vit_h
from .common import LayerNorm2d,FusionModule,AggregatorModel
from peft import get_peft_model, LoraConfig
import sys
import os
sys.path.append('./ARSeg') # Change the absolute path of the ARseg folder
from opencd.models import FoundationEncoderDecoder

sys.path.append('/UniChange') # Change the absolute path of the UniChange project
from scd_tools import color_label_to_index
import numpy as np
from skimage import io
from .seg_loss import create_loss_functions

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


# 计算二元交叉熵损失
def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def create_model(model_path: str):
    file_name = os.path.basename(model_path)
    if file_name == 'ViT_L.pth' or file_name == 'vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth':
        arch1='large'
        in_channels1=1024
    elif file_name == 'ViT_B.pth':
        arch1='base'
        in_channels1=768
    else:
        raise ValueError(f"Unsupported model file: {file_name}")
    
    model = FoundationEncoderDecoder(
        backbone=dict(
            type='ViTSAM_Normal',
            # arch='base', # VITB
            # arch='large', # VITL
            arch=arch1, 
            img_size=512,
            patch_size=16,
            in_channels=3,
            out_channels=-1,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=7,
            drop_path_rate=0.1,
            init_cfg=None
        ),
        decode_head=dict(
            type='Foundation_Decoder_v1',
            # in_channels=768, # VITB
            # in_channels=1024, # VITL
            in_channels=in_channels1, 
            out_channels=256,
            drop=0.0,
            loss_type='BCELoss',
            num_classes=2,
            loss_weight=[1, 1, 1],
            fake_loss_weight=0,
            init_cfg=None,
            finetune_cfg=None,
        ),
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=dict(
            type='FoundationInputSegDataPreProcessor',
            mean=[123.675, 116.28, 103.53] * 2,
            std=[58.395, 57.12, 57.375] * 2,
            bgr_to_rgb=True,
            rgb_to_bgr=False,
            size_divisor=32,
            pad_val=0,
            seg_pad_val=255,
            test_cfg=dict(size_divisor=32),
        )
    )
    
   
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
   
    return model
    

def extract_masks(seg_logits_list, data_samples, cls_num):
    """
    Extracts pred_masks and gt_masks from a list of segmentation logits.

    Args:
        seg_logits_list: A list of model predicted logits, each with shape [1, 3, H, W].
        data_samples: A list of data samples.

    Returns:
        pred_mask: tensor([B, H, W]) - Change detection predicted logits.
        gt_mask: array([B, H, W]) - Change detection ground truth labels.
    """
    
    
    pred_masks_list = []
    seg_logits_0 = seg_logits_list[0]
    a_logits, b_logits, cd_logits = torch.split(seg_logits_0, [1, 1, 1], dim=1)
   
    pred_masks_list.append(cd_logits)
    
    pred_masks_list_a = []
    pred_masks_list_b = []
    for i in range(1, 1 + cls_num):
        seg_logits_i = seg_logits_list[i]
        a_logits, b_logits, _ = torch.split(seg_logits_i, [1, 1, 1], dim=1) 
        
        pred_masks_list_a.append(a_logits)
        pred_masks_list_b.append(b_logits)
    
    
    pred_masks_list.extend(pred_masks_list_a)
    pred_masks_list.extend(pred_masks_list_b)

    
    pred_cd_logits = torch.cat(pred_masks_list, dim=0)
    
   
    pred_mask = torch.squeeze(pred_cd_logits, dim=1)
    
    
    gt_semantic_segs_cd = [s['gt_sem_seg']['data'] for s in data_samples]
    gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)


    gt_mask = gt_cd[0].detach().cpu().numpy()  # [bs, H, W] array
    
    return pred_mask, gt_mask

class UniChangeMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(UniChangeMetaModel, self).__init__(config)

        self.config = config
       
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_unichange_modules(self.config)

    def initialize_unichange_modules(self, config):
        self.visual_model=create_model(self.vision_pretrained)
        for param in self.visual_model.parameters():
           
            param.requires_grad = True
        if config.train_mask_decoder:
            self.visual_model.decode_head.train()
            for param in self.visual_model.decode_head.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class UniChangeModel(UniChangeMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.config=config
        super(UniChangeModel, self).__init__(config, **kwargs)
        self.initialize_unichange_modules(self.config)
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class UniChangeForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
        
        self.dataset = kwargs.pop("dataset")    
        
        self.t1_token_idx = kwargs.pop("t1_token_idx")
       
        self.t2_token_idx = kwargs.pop("t2_token_idx")
        
        self.change_token_idx = kwargs.pop("change_token_idx")

        super().__init__(config)

        self.model = UniChangeModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        elif "LEVIR-CD+" in self.dataset or "WHU-CD" in self.dataset or "S2Looking" in self.dataset or "SECOND" in self.dataset:
            return self.model_forward_levir_cd(**kwargs)
        else:
            return self.model_forward(**kwargs)

    def model_forward_levir_cd(
        self,
        rsinputs,
        rsdata_samples,
        imaget1: torch.FloatTensor,
        imaget1_clip: torch.FloatTensor,
        imaget2: torch.FloatTensor,
        imaget2_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        label1_list,
        label2_list,
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        
        # change_token_mask    
        change_token_mask=input_ids[:, 1:] == self.change_token_idx
        
        change_token_mask = torch.cat(
            [
                change_token_mask,
                torch.zeros((change_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        change_token_mask = torch.cat(
            [torch.zeros((change_token_mask.shape[0], 255*2)).bool().cuda(), change_token_mask],
            dim=1,
        )
        
        # t1_token_mask    
        t1_token_mask=input_ids[:, 1:] == self.t1_token_idx
        
        t1_token_mask = torch.cat(
            [
                t1_token_mask,
                torch.zeros((t1_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        t1_token_mask = torch.cat(
            [torch.zeros((t1_token_mask.shape[0], 255*2)).bool().cuda(), t1_token_mask],
            dim=1,
        )
        
        # t2_token_mask    
        t2_token_mask=input_ids[:, 1:] == self.t2_token_idx
        
        t2_token_mask = torch.cat(
            [
                t2_token_mask,
                torch.zeros((t2_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        t2_token_mask = torch.cat(
            [torch.zeros((t2_token_mask.shape[0], 255*2)).bool().cuda(), t2_token_mask],
            dim=1,
        )
        
        
        model_data = {
            'inputs': rsinputs,
            'data_samples': rsdata_samples
        }
            
        
        data = self.model.visual_model.data_preprocessor(model_data, False)
        
        inputs = data['inputs'].to(torch.bfloat16)
        data_samples_list = data['data_samples']
        
        
        inputs1, inputs2 = torch.chunk(inputs, 2, dim=1)
        
        # change encoder
        x = self.model.visual_model.extract_feat(inputs)
        b=int(len(x[0])/2)
        x_tensor = x[0]
        first_half = x_tensor[:b]  
        second_half = x_tensor[b:]  
        x_list = []
        for i in range(b):
            pair = torch.stack([first_half[i], second_half[i]], dim=0)  # shape: (2, C, H, W)
            x_list.append((pair,))
        
        
        if inference:
            n_batch = 1
            
            length = input_ids.shape[0]
            assert imaget1_clip.shape[0] == 1 
            
            
            imaget1_clip_extend = imaget1_clip.expand(length, -1, -1, -1).contiguous()
            imaget2_clip_extend = imaget2_clip.expand(length, -1, -1, -1).contiguous()
            

           
            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                
                output_i = super().forward(
                    imaget1=imaget1_clip_extend[:end_i - start_i],
                    imaget2=imaget2_clip_extend[:end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        # 训练阶段
        else:
            imaget1_clip_list = []
            imaget2_clip_list = []
            
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                imaget1_clip_i = (
                    imaget1_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                imaget1_clip_list.append(imaget1_clip_i)
                
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                imaget2_clip_i = (
                   
                    imaget2_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                imaget2_clip_list.append(imaget2_clip_i)
                
            imaget1_clip = torch.cat(imaget1_clip_list, dim=0)
            imaget2_clip = torch.cat(imaget2_clip_list, dim=0)

            output = super().forward(
                imaget1=imaget1_clip,
                imaget2=imaget2_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) 
        
        
       
        pred_embeddings = last_hidden_state[change_token_mask]
        change_token_counts = change_token_mask.int().sum(-1)  # [bs, ]
        change_token_offset = change_token_counts.cumsum(-1)
        change_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), change_token_offset], dim=0
        )
        change_token_offset = change_token_offset[offset]
        pred_embeddings_ = []
        for i in range(len(change_token_offset) - 1):
            start_i, end_i = change_token_offset[i], change_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        
        pred_embeddings = pred_embeddings_

       
        pred_embeddings1 = last_hidden_state[t1_token_mask]
        seg1_token_counts = t1_token_mask.int().sum(-1)  # [bs, ]
        seg1_token_offset = seg1_token_counts.cumsum(-1)
        seg1_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg1_token_offset], dim=0
        )
        seg1_token_offset = seg1_token_offset[offset]
        pred_embeddings1_ = []
        for i in range(len(seg1_token_offset) - 1):
            start_i, end_i = seg1_token_offset[i], seg1_token_offset[i + 1]
            pred_embeddings1_.append(pred_embeddings1[start_i:end_i])
        
        pred_embeddings1 = pred_embeddings1_
        
        
        pred_embeddings2 = last_hidden_state[t2_token_mask]
        seg2_token_counts = t2_token_mask.int().sum(-1)  # [bs, ]
        seg2_token_offset = seg2_token_counts.cumsum(-1)
        seg2_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg2_token_offset], dim=0
        )
        seg2_token_offset = seg2_token_offset[offset]
        pred_embeddings2_ = []
        for i in range(len(seg2_token_offset) - 1):
            start_i, end_i = seg2_token_offset[i], seg2_token_offset[i + 1]
            pred_embeddings2_.append(pred_embeddings2[start_i:end_i])
        
        pred_embeddings2 = pred_embeddings2_

       
        multimask_output = False
        all_gt_masks = []
        all_pred_masks = []
        
        # decoder
        for i in range(len(pred_embeddings)):
            seg_logits_list = []
            
            cls_num=pred_embeddings[i].shape[0]
            for j in range(cls_num):
                seg_query = pred_embeddings[i][j].unsqueeze(0)
                data_samples=[data_samples_list[i]]
                x=x_list[i]
                flag="CD"
                seg_logits = self.model.visual_model.decode_head.forward(x, data_samples, seg_query, None, None, flag, None)
                seg_logits_list.append(seg_logits)
                
            cls_num1=pred_embeddings1[i].shape[0]
            for j in range(cls_num1):
                seg_query1 = pred_embeddings1[i][j].unsqueeze(0)
                seg_query2 = pred_embeddings2[i][j].unsqueeze(0)
                data_samples=[data_samples_list[i]]
                x=x_list[i]
                flag="SEG"
                seg_logits1_2 = self.model.visual_model.decode_head.forward(x, data_samples,None,seg_query1, seg_query2, flag, None)
                seg_logits_list.append(seg_logits1_2)
            
                
            pred_mask,gt_mask = extract_masks(seg_logits_list, data_samples, cls_num1)
            all_gt_masks.append(gt_mask)
            all_pred_masks.append(pred_mask)
            
        
        model_output = output
        
        
        bn_gt_mask = [all_gt_masks[i][0:1] for i in range(len(all_gt_masks))]
        bn_pred_mask = [all_pred_masks[i][0:1] for i in range(len(all_pred_masks))]

       
        first_pred_mask = all_pred_masks[0]  
        is_correct_shape = (first_pred_mask.shape == torch.Size([15, 512, 512]))
        
        
        is_building_seg=(first_pred_mask.shape == torch.Size([3, 1024, 1024]))
        if is_building_seg:
            bn_gt_mask = [all_gt_masks[i][0:3] for i in range(len(all_gt_masks))]
            bn_pred_mask = [all_pred_masks[i][0:3] for i in range(len(all_pred_masks))]
        
        
        if is_correct_shape:
            outputA = []
            outputB = []
            for i in range(len(all_pred_masks)):
                layers_1_7 = all_pred_masks[i][1:8]  
                outputA.append(layers_1_7)
                
               
                layers_8_14 = all_pred_masks[i][8:15] 
                outputB.append(layers_8_14)
            
            outputA = torch.stack(outputA)  
            outputB = torch.stack(outputB)  
            
          
            label_A_list = []
            label_B_list = []
            for i in range(len(all_pred_masks)):
                label_A = color_label_to_index(io.imread(label1_list[i])).astype(np.uint8)
                label_B = color_label_to_index(io.imread(label2_list[i])).astype(np.uint8)
                label_A_list.append(label_A)
                label_B_list.append(label_B)
            
            label_A = torch.stack([torch.from_numpy(label).long() for label in label_A_list]).to(device=outputA.device)  # shape: (batch_size, 512, 512)
            label_B = torch.stack([torch.from_numpy(label).long() for label in label_B_list]).to(device=outputB.device)  # shape: (batch_size, 512, 512)
            
            
            criterion, criterion_sc = create_loss_functions()
           
            loss_seg = criterion(outputA, label_A) + criterion(outputB, label_B)
            
           
            if torch.isnan(loss_seg):
                loss_seg = torch.tensor(0.0, device=loss_seg.device, requires_grad=True)
                
            labels_bn = (label_A > 0).unsqueeze(1).cuda().float()  
            loss_sc = criterion_sc(outputA[:, 1:], outputB[:, 1:], labels_bn)
        else:
            loss_seg = torch.tensor(0.0, device=all_pred_masks[0].device, requires_grad=True)
            loss_sc = torch.tensor(0.0, device=all_pred_masks[0].device, requires_grad=True)
        
        Cls_Num = pred_embeddings1[0].shape[0] if len(pred_embeddings1) > 0 else 0
        
        if inference:
            return {
                "label1_list": label1_list,
                "label2_list": label2_list,
                "pred_masks": all_pred_masks,
                "gt_masks": all_gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * 1.0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(all_pred_masks)):
            gt_mask = torch.stack(
                    [
                        torch.from_numpy(gt_mask).to(dtype=bn_pred_mask[batch_idx].dtype, device=bn_pred_mask[batch_idx].device)
                        for gt_mask in bn_gt_mask[batch_idx]
                    ],
                    dim=0
                ) 
            pred_mask = bn_pred_mask[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            
            
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss + loss_seg*0.5 + loss_sc

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "seg_loss": loss_seg,
            "sc_loss": loss_sc,
        }
    