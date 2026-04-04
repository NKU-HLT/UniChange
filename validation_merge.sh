export PATH=./CUDA/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=./CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_SHOW_CPP_STACKTRACES=0

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=22866 ./UniChange/train_UniChange.py \
    --version="./UniChange-7B" \
    --dataset_dir='./UniChange/dataset' \
    --vision_pretrained='./UniChange_data/ARSeg/pretrain/ViT_L.pth' \
    --dataset="LEVIR-CD+||S2Looking||WHU-CD||SECOND" \
    --sample_rates="13,77,12,57" \
    --val_datasets="LEVIR-CD+,S2Looking,WHU-CD,SECOND" \
    --exp_name="" \
    --batch_size="1" \
    --eval_only 