import argparse
import os
import shutil
import sys
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from scipy import stats
from metric import (
    preds_to_indices_batch,
    compute_confusion_matrix,
    overall_accuracy,
    kappa_score,
    cal_kappa,
    preprocess_gt_pair,
    compute_scd_metrics_gstm_style,
)
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.UniChange import UniChangeForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from skimage import io
from scd_tools import color_label_to_index,accuracy,SCDD_eval_all

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def parse_args(args):
    parser = argparse.ArgumentParser(description="UniChange Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="/UniChange_data/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="LEVIR-CD+", type=str
    )
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train1|train2", type=str)
    parser.add_argument("--val_dataset", default="LEVIR-CD+", type=str)
    parser.add_argument("--val_datasets", default="", type=str, help="eg: LEVIR-CD+,S2Looking")
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="unichange", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=100, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=16,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lovasz_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="facebook/sam2-hiera-large", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token 
    
    #Add [SEG1]token
    num_added_tokens_seg1 = tokenizer.add_tokens("[T1]") 
    args.t1_token_idx = tokenizer("[T1]", add_special_tokens=False).input_ids[0]


    # Add [SEG2]token
    num_added_tokens_seg2 = tokenizer.add_tokens("[T2]") 
    args.t2_token_idx = tokenizer("[T2]", add_special_tokens=False).input_ids[0]
    
    # Add [CHANGE]token
    num_added_tokens_change = tokenizer.add_tokens("[CHANGE]") 
    args.change_token_idx = tokenizer("[CHANGE]", add_special_tokens=False).input_ids[0]
    
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder, # bool
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "t1_token_idx": args.t1_token_idx,
        "t2_token_idx": args.t2_token_idx,
        "change_token_idx": args.change_token_idx,
        "vision_pretrained": args.vision_pretrained,# SAM
        "vision_tower": args.vision_tower,#clip
        "use_mm_start_end": args.use_mm_start_end,
        "dataset": args.dataset,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        
    
    model = UniChangeForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()


    
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_unichange_modules(model.get_model().config)

    for p in vision_tower.parameters():# clip
        p.requires_grad = True
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "decode_head", "text_hidden_fcs","backbone","vision_tower","mm_projector"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
        if args.val_datasets:
            val_datasets = {}
            val_loaders = {}
            val_dataset_names = [name.strip() for name in args.val_datasets.split(",")]
            for val_name in val_dataset_names:
                val_dataset = ValDataset(
                    args.dataset_dir,
                    tokenizer,
                    args.vision_tower,
                    val_name,
                    args.image_size,
                )
                val_datasets[val_name] = val_dataset
                print(f"Validation dataset {val_name}: {len(val_dataset)} examples")
        else:
            val_dataset = ValDataset(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.val_dataset,
                args.image_size,
            )
            val_datasets = {args.val_dataset: val_dataset}
            print(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.")
    else:
        val_datasets = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    
    
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            dataset=args.dataset,
        ),
        config=ds_config,
        )
    
        
    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_datasets is not None:
        assert args.val_batch_size == 1
        val_loaders = {}
        for val_name, val_dataset in val_datasets.items():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            )
            val_loaders[val_name] = val_loader

    train_iter = iter(train_loader)
    best_score, cur_giou = 0.0, 0.0

    if args.eval_only:
        if isinstance(val_loaders, dict):
            results = validate(val_loaders, model_engine, 0, writer, args)
        else:
            giou, ciou = validate(val_loaders, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            if isinstance(val_loaders, dict):
                results = validate(val_loaders, model_engine, epoch, writer, args)
                total_score = 0.0
                dataset_count = len(results)
                
                for dataset_name, result in results.items():
                    if dataset_name.lower() == "second":
                        total_score += result.get("Fscd", 0.0)
                    else:
                        total_score += result.get("iou1", 0.0)
                
                avg_score = total_score / dataset_count
                is_best = avg_score > best_score
                best_score = max(avg_score, best_score)
                cur_giou = list(results.values())[0]["giou"] if is_best else cur_giou
                display_iou1 = avg_score  
            else:
                detailed_result = validate(val_loaders, model_engine, epoch, writer, args)
                score = detailed_result.get("iou1", 0.0)
                
                is_best = score > best_score
                best_score = max(score, best_score)
                cur_giou = detailed_result["giou"] if is_best else cur_giou
                display_iou1 = score

        should_save_tensor = torch.tensor(
            1 if (args.no_eval or is_best) else 0,
            device=f"cuda:{args.local_rank}"
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(should_save_tensor, src=0)
        should_save = bool(should_save_tensor.item())

        if should_save:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                print(f"save checkpoint: no_eval={args.no_eval}, is_best={is_best}, save_dir={save_dir}")
                fscd_info = ""
                if isinstance(val_loaders, dict):
                    for dataset_name, result in results.items():
                        if dataset_name.lower() == "second":
                            fscd_value = result.get("Fscd", 0.0)
                            fscd_info = f"_Fscd{fscd_value:.3f}"
                            print(f"SECOND dataset Fscd: {fscd_value:.4f}")
                            break
                
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_avg{:.3f}{}.pth".format(
                            cur_giou, display_iou1, fscd_info
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)

            if args.local_rank == 0:
                print("Wait for all processes to reach the barrier...")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if args.local_rank == 0:
                print("All processes have been synchronized. Start saving the model....")
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print("Model saved successfully")


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    seg_losses = AverageMeter("SegLoss", ":.4f")
    sc_losses = AverageMeter("SCLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            seg_losses,
            sc_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):# 500
        for i in range(args.grad_accumulation_steps):# 10
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["imaget1"] = input_dict["imaget1"].half()
                input_dict["imaget1_clip"] = input_dict["imaget1_clip"].half()
                input_dict["imaget2"] = input_dict["imaget2"].half()
                input_dict["imaget2_clip"] = input_dict["imaget2_clip"].half()
            elif args.precision == "bf16":
                input_dict["imaget1"] = input_dict["imaget1"].bfloat16()
                input_dict["imaget1_clip"] = input_dict["imaget1_clip"].bfloat16()
                input_dict["imaget2"] = input_dict["imaget2"].bfloat16()
                input_dict["imaget2_clip"] = input_dict["imaget2_clip"].bfloat16()
            else:
                input_dict["imaget1"] = input_dict["imaget1"].float()
                input_dict["imaget1_clip"] = input_dict["imaget1_clip"].float()
                input_dict["imaget2"] = input_dict["imaget2"].float()
                input_dict["imaget2_clip"] = input_dict["imaget2_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            seg_loss = output_dict["seg_loss"]
            sc_loss = output_dict["sc_loss"]
            
            losses.update(loss.item(), input_dict["imaget1"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["imaget1"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["imaget1"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["imaget1"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["imaget1"].size(0))
            seg_losses.update(seg_loss.item(), input_dict["imaget1"].size(0))
            sc_losses.update(sc_loss.item(), input_dict["imaget1"].size(0))
            
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                seg_losses.all_reduce()
                sc_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar("train/seg_loss", seg_losses.avg, global_step)
                writer.add_scalar("train/sc_loss", sc_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            seg_losses.reset()
            sc_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter

def validate_single(val_loader, model_engine, epoch, writer, args, dataset_name=""):
    categories = ["all"]
    intersection_meters = {cat: AverageMeter("Intersec", ":6.3f", Summary.SUM) for cat in categories}
    union_meters = {cat: AverageMeter("Union", ":6.3f", Summary.SUM) for cat in categories}
    acc_iou_meters = {cat: AverageMeter("gIoU", ":6.3f", Summary.SUM) for cat in categories}

    precision_meters = {cat: AverageMeter("Precision", ":6.3f", Summary.SUM) for cat in categories}
    recall_meters = {cat: AverageMeter("Recall", ":6.3f", Summary.SUM) for cat in categories}
    f1_meters = {cat: AverageMeter("F1", ":6.3f", Summary.SUM) for cat in categories}
    miou_meters = {cat: AverageMeter("mIoU", ":6.3f", Summary.SUM) for cat in categories}
    acc_meters = {cat: AverageMeter("Acc", ":6.3f", Summary.SUM) for cat in categories}
    
    tp_meters = {cat: AverageMeter("TP", ":6.3f", Summary.SUM) for cat in categories}
    tn_meters = {cat: AverageMeter("TN", ":6.3f", Summary.SUM) for cat in categories}
    fp_meters = {cat: AverageMeter("FP", ":6.3f", Summary.SUM) for cat in categories}
    fn_meters = {cat: AverageMeter("FN", ":6.3f", Summary.SUM) for cat in categories}

    
    is_second_dataset = dataset_name.lower() == "second"
    display_name = dataset_name
    preds_all = []
    labels_all = []
    
    scd_acc_meter = AverageMeter("SCD_Acc", ":6.3f", Summary.SUM) if is_second_dataset else None
    

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader, desc=f"Validating {display_name}"):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["imaget1"] = input_dict["imaget1"].half()
            input_dict["imaget1_clip"] = input_dict["imaget1_clip"].half()
            input_dict["imaget2"] = input_dict["imaget2"].half()
            input_dict["imaget2_clip"] = input_dict["imaget2_clip"].half()
        elif args.precision == "bf16":
            input_dict["imaget1"] = input_dict["imaget1"].bfloat16()
            input_dict["imaget1_clip"] = input_dict["imaget1_clip"].bfloat16()
            input_dict["imaget2"] = input_dict["imaget2"].bfloat16()
            input_dict["imaget2_clip"] = input_dict["imaget2_clip"].bfloat16()
        else:
            input_dict["imaget1"] = input_dict["imaget1"].float()
            input_dict["imaget1_clip"] = input_dict["imaget1_clip"].float()
            input_dict["imaget2"] = input_dict["imaget2"].float()
            input_dict["imaget2_clip"] = input_dict["imaget2_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

       
        label1_list=output_dict["label1_list"]
        label2_list=output_dict["label2_list"]
        
        pred_masks = output_dict["pred_masks"][0]
        gt_masks_raw = output_dict["gt_masks"][0]
        
        def to_2d_mask(tensor_like, is_second=False):
            if isinstance(tensor_like, np.ndarray):
                tensor_like = torch.from_numpy(tensor_like)
            if isinstance(tensor_like, torch.Tensor):
                t = tensor_like
                if is_second:
                    if t.ndim == 4: 
                        t = t[0, 0]  
                    elif t.ndim == 3:  
                        t = t[0]  
                    elif t.ndim == 2:  
                        t = t
                    else:
                        raise ValueError(f"Unsupported SECOND mask dimension: {t.ndim}")
                else:
                    if t.ndim == 4:  
                        t = t[0, 0]
                    elif t.ndim == 3:  
                        t = t[0]
                    elif t.ndim == 2: 
                        t = t
                    else:
                        raise ValueError(f"Unsupported mask dimension: {t.ndim}")
                return t
            else:
                raise ValueError("Unsupported mask type for evaluation")

        pred_all_2d = to_2d_mask(pred_masks, is_second_dataset)
        gt_all_2d = to_2d_mask(gt_masks_raw, is_second_dataset)

        pred_all_2d = pred_all_2d.to(device=pred_all_2d.device)
        gt_all_2d = gt_all_2d.to(device=pred_all_2d.device)

        output_i = (pred_all_2d > 0).int()
        mask_i = (gt_all_2d > 0).int()

        intersection_i, union_i, _ = intersectionAndUnionGPU(
            output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
        )

        acc_iou = intersection_i / (union_i + 1e-5)
        acc_iou[union_i == 0] += 1.0

        pred_binary = (output_i.squeeze() > 0).bool()
        label_binary = (mask_i.squeeze() > 0).bool()

        tp = (pred_binary & label_binary).sum().item()
        fn = ((~pred_binary) & label_binary).sum().item()
        fp = (pred_binary & (~label_binary)).sum().item()
        tn = ((~pred_binary) & (~label_binary)).sum().item()

        intersection_meters["all"].update(intersection_i.cpu().numpy())
        union_meters["all"].update(union_i.cpu().numpy())
        acc_iou_meters["all"].update(acc_iou.cpu().numpy(), n=1)

        tp_meters["all"].update(tp, n=1)
        tn_meters["all"].update(tn, n=1)
        fp_meters["all"].update(fp, n=1)
        fn_meters["all"].update(fn, n=1)

        if is_second_dataset:
            label_A = color_label_to_index(io.imread(label1_list[0])).astype(np.uint8)
            label_B = color_label_to_index(io.imread(label2_list[0])).astype(np.uint8)
            
            out_change = pred_all_2d.unsqueeze(0).unsqueeze(0)  
            change_mask = out_change > 0 
            t1 = pred_masks[1:8].unsqueeze(0)   
            t2 = pred_masks[8:15].unsqueeze(0)  
            
            pred_A = torch.argmax(t1, dim=1) 
            pred_B = torch.argmax(t2, dim=1) 
            
            pred_A = (pred_A * change_mask.squeeze().long()).squeeze(0).detach().cpu().numpy()
            pred_B = (pred_B * change_mask.squeeze().long()).squeeze(0).detach().cpu().numpy()
            
            import os
            target_local_rank = 0 
            
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            scd_acc_meter.update(acc)

    for cat in categories:
        intersection_meters[cat].all_reduce()
        union_meters[cat].all_reduce()
        acc_iou_meters[cat].all_reduce()
        precision_meters[cat].all_reduce()
        recall_meters[cat].all_reduce()
        f1_meters[cat].all_reduce()
        miou_meters[cat].all_reduce()
        acc_meters[cat].all_reduce()
        tp_meters[cat].all_reduce()
        tn_meters[cat].all_reduce()
        fp_meters[cat].all_reduce()
        fn_meters[cat].all_reduce()
        if is_second_dataset and scd_acc_meter is not None:
            scd_acc_meter.all_reduce()

    results = {}
    cat = "all"
    iou_class = intersection_meters[cat].sum / (union_meters[cat].sum + 1e-10)
    iou0_old = iou_class[1]
    giou = acc_iou_meters[cat].avg[1]

    total_tp = tp_meters[cat].sum
    total_tn = tn_meters[cat].sum
    total_fp = fp_meters[cat].sum
    total_fn = fn_meters[cat].sum

    iou0_new = total_tn / (total_tn + total_fp + total_fn + 1e-10)
    iou1 = total_tp / (total_tp + total_fp + total_fn + 1e-10)

    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-10)
    miou = (iou0_new + iou1) / 2

    results[cat] = {
        "giou": giou,
        "iou0": iou0_old,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miou": miou,
        "acc": acc,
        "iou0_new": iou0_new,
        "iou1": iou1,
    }

    scd_metrics = {"Fscd": 0.0, "IoU_mean": 0.0, "Sek": 0.0, "Scd_acc": 0.0, "OA": 0.0}
    if is_second_dataset:
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_preds, preds_all)
            torch.distributed.all_gather_object(gathered_labels, labels_all)

            metrics_tensor = torch.zeros(
                4, device=f"cuda:{args.local_rank}", dtype=torch.float32
            )
            if args.local_rank == 0:
                merged_preds = []
                merged_labels = []
                for lst in gathered_preds:
                    if lst:
                        merged_preds.extend(lst)
                for lst in gathered_labels:
                    if lst:
                        merged_labels.extend(lst)
                Fscd, IoU_mean, Sek, OA = SCDD_eval_all(merged_preds, merged_labels, 7)
                metrics_tensor.copy_(
                    torch.tensor([Fscd, IoU_mean, Sek, OA], device=metrics_tensor.device)
                )
            torch.distributed.broadcast(metrics_tensor, src=0)
            Fscd, IoU_mean, Sek, OA = [float(x) for x in metrics_tensor.tolist()]
        else:
            Fscd, IoU_mean, Sek, OA = SCDD_eval_all(preds_all, labels_all, 7)

        scd_metrics["Fscd"] = Fscd
        scd_metrics["IoU_mean"] = IoU_mean
        scd_metrics["Sek"] = Sek
        scd_metrics["OA"] = OA
        scd_metrics["Scd_acc"] = scd_acc_meter.avg
    results[cat].update(scd_metrics)

    if args.local_rank == 0:
        prefix = f"val/{dataset_name}/" if dataset_name else "val/"
        writer.add_scalar(f"{prefix}all_precision", results["all"]["precision"], epoch)
        writer.add_scalar(f"{prefix}all_recall", results["all"]["recall"], epoch)
        writer.add_scalar(f"{prefix}all_f1", results["all"]["f1"], epoch)
        writer.add_scalar(f"{prefix}all_miou", results["all"]["miou"], epoch)
        writer.add_scalar(f"{prefix}all_acc", results["all"]["acc"], epoch)
        writer.add_scalar(f"{prefix}all_iou0_new", results["all"]["iou0_new"], epoch)
        writer.add_scalar(f"{prefix}all_iou1", results["all"]["iou1"], epoch)
        writer.add_scalar(f"{prefix}all_giou", results["all"]["giou"], epoch)
        writer.add_scalar(f"{prefix}all_iou0", results["all"]["iou0"], epoch)
        
        writer.add_scalar(f"{prefix}scd_acc", results["all"].get("Scd_acc", 0.0), epoch)
        writer.add_scalar(f"{prefix}scd_SeK", results["all"].get("Sek", 0.0), epoch)
        writer.add_scalar(f"{prefix}scd_Fscd", results["all"].get("Fscd", 0.0), epoch)
        writer.add_scalar(f"{prefix}scd_mIoU", results["all"].get("IoU_mean", 0.0), epoch)
        writer.add_scalar(f"{prefix}scd_OA", results["all"].get("OA", 0.0), epoch)

        if is_second_dataset:
            print(
                f"{display_name} - All: "
                f"P:{results['all']['precision']*100:.2f}% R:{results['all']['recall']*100:.2f}% F1:{results['all']['f1']*100:.2f}% "
                f"mIoU:{results['all']['miou']*100:.2f}% Acc:{results['all']['acc']*100:.2f}% "
                f"IoU0:{results['all']['iou0_new']*100:.2f}% IoU1:{results['all']['iou1']*100:.2f}% "
                f"Original_IoU0:{results['all']['iou0']*100:.2f}% GIoU:{results['all']['giou']*100:.2f}% | "
                f"SCD: Acc:{results['all'].get('Scd_acc',0.0)*100:.2f}% "
                f"SeK:{results['all'].get('Sek',0.0)*100:.2f}% "
                f"Fscd:{results['all'].get('Fscd',0.0)*100:.2f}% "
                f"mIoU:{results['all'].get('IoU_mean',0.0)*100:.2f}% "
                f"OA:{results['all'].get('OA',0.0)*100:.2f}%"
            )
        else:
            print(f"{display_name} - All: P:{results['all']['precision']*100:.2f}% R:{results['all']['recall']*100:.2f}% F1:{results['all']['f1']*100:.2f}% mIoU:{results['all']['miou']*100:.2f}% Acc:{results['all']['acc']*100:.2f}% IoU0:{results['all']['iou0_new']*100:.2f}% IoU1:{results['all']['iou1']*100:.2f}% Original_IoU0:{results['all']['iou0']*100:.2f}% GIoU:{results['all']['giou']*100:.2f}%")

    return results

def print_validation_summary(detailed_results, epoch, args):
    """打印验证结果汇总表格 - 仅'all'类别"""
    print("\n" + "="*140)
    print(f"Epoch {epoch} - Validation Results Summary (All Categories)")
    print("="*140)

    categories = ["all"]
    col_width = 35
    
    header1 = f"{'Eval_Value':<12} {'Dataset':<15} |"
    for cat in categories:
        header1 += f" {cat.capitalize():<{col_width-1}} |"
    print(header1)
    
    header2 = f"{'':<12} {'':<15} |"
    for cat in categories:
        has_second = any("second" in dataset.lower() for dataset in detailed_results.keys())
        if has_second:
            header2 += (
                f" {'IoU1':<6} {'IoU0':<6} {'mIoU':<7} {'P':<6} {'R':<6} {'F1':<6} {'Acc':<6} |"
                f" {'SCD_Acc':<8} {'SCD_SeK':<8} {'SCD_Fscd':<9} {'SCD_mIoU':<9} {'SCD_OA':<7} |"
            )
        else:
            header2 += (
                f" {'IoU1':<6} {'IoU0':<6} {'mIoU':<7} {'P':<6} {'R':<6} {'F1':<6} {'Acc':<6} |"
            )
    print(header2)
    print("-" * 140)
    
    total_metrics = {}
    for cat in categories:
        total_metrics[cat] = {
            "precision": 0, "recall": 0, "f1": 0, "miou": 0, "acc": 0, "iou0_new": 0, "iou1": 0,
            "Scd_acc": 0, "Sek": 0, "Fscd": 0, "IoU_mean": 0, "OA": 0,
        }
    
    num_datasets = len(detailed_results)
    
    for dataset_name, metrics in detailed_results.items():
        is_second = dataset_name.lower() == "second"
        if is_second:
            eval_value = f"{metrics['all'].get('Fscd', 0.0)*100:.2f}%"
        else:
            eval_value = f"{metrics['all'].get('iou1', 0.0)*100:.2f}%"
        display_name = dataset_name
        row = f"{eval_value:<12} {display_name:<15} |"
        for cat in categories:
            cat_metrics = metrics[cat]
            is_second = dataset_name.lower() == "second"
            if is_second:
                scd_acc = cat_metrics.get("Scd_acc", 0.0)
                scd_sek = cat_metrics.get("Sek", 0.0)
                scd_fscd = cat_metrics.get("Fscd", 0.0)
                scd_miou = cat_metrics.get("IoU_mean", 0.0)
                scd_oa = cat_metrics.get("OA", 0.0)
                row += (
                    f" {cat_metrics['iou1']*100:<6.2f}% {cat_metrics['iou0_new']*100:<6.2f}% {cat_metrics['miou']*100:<7.2f}% "
                    f"{cat_metrics['precision']*100:<6.2f}% {cat_metrics['recall']*100:<6.2f}% {cat_metrics['f1']*100:<6.2f}% {cat_metrics['acc']*100:<6.2f}% |"
                    f" {scd_acc*100:<8.2f}% {scd_sek*100:<8.2f}% {scd_fscd*100:<9.2f}% {scd_miou*100:<9.2f}% {scd_oa*100:<7.2f}% |"
                )
            else:
                row += (
                    f" {cat_metrics['iou1']*100:<6.2f}% {cat_metrics['iou0_new']*100:<6.2f}% {cat_metrics['miou']*100:<7.2f}% "
                    f"{cat_metrics['precision']*100:<6.2f}% {cat_metrics['recall']*100:<6.2f}% {cat_metrics['f1']*100:<6.2f}% {cat_metrics['acc']*100:<6.2f}% |"
                )
        print(row)
        
        for cat in categories:
            for metric in [
                "precision", "recall", "f1", "miou", "acc", "iou0_new", "iou1",
                "Scd_acc", "Sek", "Fscd", "IoU_mean", "OA",
            ]:
                total_metrics[cat][metric] += metrics[cat].get(metric, 0.0)
    
    print("-" * 150)
    total_eval_value = 0
    for dataset_name, metrics in detailed_results.items():
        is_second = dataset_name.lower() == "second"
        if is_second:
            total_eval_value += metrics['all'].get('Fscd', 0.0)
        else:
            total_eval_value += metrics['all'].get('iou1', 0.0)
    avg_eval_value = f"{total_eval_value/num_datasets*100:.2f}%"
    avg_row = f"{avg_eval_value:<12} {'Average':<15} |"
    for cat in categories:
        cat_metrics = total_metrics[cat]
        has_second = any("second" in dataset.lower() for dataset in detailed_results.keys())
        if has_second:
            avg_row += (
                f" {cat_metrics['iou1']/num_datasets*100:<6.2f}% {cat_metrics['iou0_new']/num_datasets*100:<6.2f}% {cat_metrics['miou']/num_datasets*100:<7.2f}% "
                f"{cat_metrics['precision']/num_datasets*100:<6.2f}% {cat_metrics['recall']/num_datasets*100:<6.2f}% {cat_metrics['f1']/num_datasets*100:<6.2f}% {cat_metrics['acc']/num_datasets*100:<6.2f}% |"
                f" {cat_metrics['Scd_acc']/num_datasets*100:<8.2f}% {cat_metrics['Sek']/num_datasets*100:<8.2f}% {cat_metrics['Fscd']/num_datasets*100:<9.2f}% {cat_metrics['IoU_mean']/num_datasets*100:<9.2f}% {cat_metrics['OA']/num_datasets*100:<7.2f}% |"
            )
        else:
            avg_row += (
                f" {cat_metrics['iou1']/num_datasets*100:<6.2f}% {cat_metrics['iou0_new']/num_datasets*100:<6.2f}% {cat_metrics['miou']/num_datasets*100:<7.2f}% "
                f"{cat_metrics['precision']/num_datasets*100:<6.2f}% {cat_metrics['recall']/num_datasets*100:<6.2f}% {cat_metrics['f1']/num_datasets*100:<6.2f}% {cat_metrics['acc']/num_datasets*100:<6.2f}% |"
            )
    print(avg_row)
    print("="*150)
    print()

def validate(val_loaders, model_engine, epoch, writer, args):
    if isinstance(val_loaders, dict):
        results = {}
        detailed_results = {} 
        
        for dataset_name, val_loader in val_loaders.items():
            detailed_result = validate_single(val_loader, model_engine, epoch, writer, args, dataset_name)
            
            detailed_results[dataset_name] = detailed_result
            
            results[dataset_name] = {
                "giou": detailed_result["all"]["giou"],  
                "iou0": detailed_result["all"]["iou0"], 
                "precision": detailed_result["all"]["precision"],
                "recall": detailed_result["all"]["recall"],
                "f1": detailed_result["all"]["f1"],
                "miou": detailed_result["all"]["miou"],
                "acc": detailed_result["all"]["acc"],
                "iou0_new": detailed_result["all"]["iou0_new"],  
                "iou1": detailed_result["all"]["iou1"],         
                "Fscd": detailed_result["all"].get("Fscd", 0.0),
                "Scd_acc": detailed_result["all"].get("Scd_acc", 0.0),
                "Sek": detailed_result["all"].get("Sek", 0.0),
                "IoU_mean": detailed_result["all"].get("IoU_mean", 0.0),
                "OA": detailed_result["all"].get("OA", 0.0),
            }
        
        if args.local_rank == 0:
            print_validation_summary(detailed_results, epoch, args)
        
        return results
    else:
        detailed_result = validate_single(val_loaders, model_engine, epoch, writer, args)
        return detailed_result["all"]


if __name__ == "__main__":
    main(sys.argv[1:])