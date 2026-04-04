export PATH=./CUDA/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=./CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_SHOW_CPP_STACKTRACES=0

CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --master_port=22669 ./UniChange/train_UniChange.py \
    --version="./UniChange_data/LLaVA-7B-Lightening-v1-1" \
    --dataset_dir='./UniChange/dataset' \
    --vision_pretrained='./UniChange_data/ARSeg/pretrain/ViT_L.pth' \
    --dataset="LEVIR-CD+||S2Looking||WHU-CD||SECOND" \
    --sample_rates="13,77,12,57" \
    --val_datasets="LEVIR-CD+,S2Looking,WHU-CD,SECOND" \
    --exp_name="UniChange" \
    --steps_per_epoch="400" \
    --batch_size="1" \
    --grad_accumulation_steps="8" \
    --lr="0.00005" \
    --log_base_dir="checkpoints" &> log/UniChange.log
