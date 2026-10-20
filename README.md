<h1 align="center"> UniChange: Unifying Change Detection with Multimodal Large Language Model </h1>

<h5 align="center"><em>Xu Zhang, Danyang Li, Xiaohang Dong, Tianhao Wu, Hualong Yu, Jianye Wang, Qicheng Li, and Xiang Li </em></h5>

# News 📢
- **`2026/10/20`** Release dataset and final_weight checkpoints.
- **`2026/04/04`** Release the training and inference code.
- **`2026/02/21`** UniChange paper accepted by CVPR 2026. 🎉🎉🎉🎉.
- **`2025/11/04`** UniChange [paper](https://arxiv.org/abs/2511.02607) is available on arXiv.

# Overview ✨
- [News 📢](#news-)
- [TODO List 📅](#todo-list-)
- [Introduction 📄](#introduction-)
- [Installation 🛠️](#installation-️)
- [Datasets 🗂️](#datasets-️)
- [Checkpoints ⬇️](#checkpoints-️)
- [Training 🏋️](#training-️)
- [Inference 🧪](#inference-️)
- [License 🔑](#license-)
- [Acknowledgement 🤝](#acknowledgement-)
- [Citation 📚](#citation-)

# TODO List 📅
We will release the model code and checkpoints soon!🌈

- [x] Release environment setup
- [x] Release the training and inference code
- [x] Release the dataset
- [x] Release checkpoints

# Introduction 📄 
Change detection (CD) is a fundamental task for monitoring and analysing land cover dynamics. While recent high performance models and high quality datasets have significantly advanced the field, a critical limitation persists. Current models typically acquire limited knowledge from single-type annotated data and cannot concurrently leverage diverse binary change detection (BCD) and semantic change detection (SCD) datasets. This constraint leads to poor generalisation and limited versatility. The recent advancements in Multimodal Large Language Models (MLLMs) introduce new possibilities for a unified CD framework. We leverage the language priors and unification capabilities of MLLMs to develop UniChange, the first MLLM-based unified change detection model. UniChange integrates generative language abilities with specialised CD functionalities. We introduce three special tokens: [T1], [T2], and [CHANGE], utilising their embeddings as the key to query variations. This approach successfully accommodates both BCD and SCD tasks. Furthermore, UniChange utilises text prompts to guide the identification of change categories, eliminating the reliance on predefined classification heads. This design allows UniChange to effectively acquire knowledge from multi-source datasets, even when their class definitions conflict. Experiments on four public benchmarks (WHU-CD, S2Looking, LEVIR-CD+, and SECOND) demonstrate SOTA performance, achieving IoU scores of 90.41, 53.04, 78.87, and 57.62, respectively, surpassing all previous methods. The code is available at [GitHub](https://github.com/NKU-HLT/UniChange/).

---

<p align="center">
  <img width="810" alt="image" src="./asserts/overview.png"/>
</p>

Figure 1: The overview structure of UniChange is shown in (a). UniChange generates text outputs based on text instructions and dual-temporal remote sensing images. The text outputs incorporate a series of special tokens ([T1], [T2], and [CHANGE]). Concurrently, the dual-temporal remote sensing images undergo feature extraction via the Vision Backbone. These features, alongside the embeddings corresponding to the special tokens, are fed into the Token Driven Decoder. Ultimately, UniChange generates the masks corresponding to the text instructions. The overall structure of the Token Driven Decoder is shown in (b). It receives dual-temporal remote sensing image features and special token embeddings, generating corresponding masks.

# Installation 🛠️

1. Clone this repository and navigate to the base folder
```bash
git clone https://github.com/NKU-HLT/UniChange.git
cd UniChange
```

2. Install packages
```bash
conda create -n UniChange python=3.10 -y
conda activate UniChange
pip install -r requirements.txt
```

3. Install Flash-Attention
```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

# Datasets 🗂️

| Dataset | Task | Download |
| --- | --- | --- |
| WHU-CD | Binary Change Detection | [WHU Building Dataset](https://gpcv.whu.edu.cn/data/building_dataset.html) |
| S2Looking | Binary Change Detection | [GitHub](https://github.com/S2Looking/) |
| LEVIR-CD+ | Binary Change Detection | [LEVIR-CD Project Page](https://justchenhao.github.io/LEVIR/) |
| SECOND | Semantic Change Detection | [SCD Project Page](https://captain-whu.github.io/SCD/) |

# Checkpoints ⬇️

final_weight: [百度网盘](https://pan.baidu.com/s/1LAjJYrRdPWC_CzllRWDDqQ?pwd=3kwr)（提取码: 3kwr）

# Training 🏋️

```bash
bash train_UniChange.sh
```

脚本内容参考 `train_UniChange.sh`，训练多数据集联合训练，4 卡 DeepSpeed：

```bash
export PATH=./CUDA/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=./CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH

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
```

# Inference 🧪

```bash
bash validation_merge.sh
```

脚本内容参考 `validation_merge.sh`，使用合并权重进行验证，4 卡 DeepSpeed：

```bash
export PATH=./CUDA/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=./CUDA/cuda-11.8/lib64:$LD_LIBRARY_PATH

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
```

# License 🔑
This code repository is licensed under [Apache 2.0](./LICENSE).

# Acknowledgement 🤝
We would like to thank the following projects for their contributions to this work:

- [LISA](https://github.com/JIA-Lab-research/LISA)
- [RSBuilding](https://github.com/Meize0729/RSBuilding)
- [SAM](https://github.com/facebookresearch/segment-anything)
  
# Citation 📚

If you find our project useful for your research, please consider citing our paper and codebase with the following BibTeX:

```bibtex
@article{zhang2025unichange,
  title={UniChange: Unifying Change Detection with Multimodal Large Language Model},
  author={Zhang, Xu and Li, Danyang and Dong, Xiaohang and Wu, Tianhao and Yu, Hualong and Wang, Jianye and Li, Qicheng and Li, Xiang},
  journal={arXiv preprint arXiv:2511.02607},
  year={2025}
}
```
