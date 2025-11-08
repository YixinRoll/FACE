# FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens

[![arXiv](https://img.shields.io/badge/arXiv-1706.03762-b31b1b.svg)](https://arxiv.org/abs/2510.15729)

## 1. Special Files

`encoder/models/general_cf/` all models, including base models (end without vq) and FACE enhanced models (end with vq)

`encoder/FACE.py` the implementation of "Vector-quantized Disentangled Representation Mapping"



## 2. Run

### 2.1. Environments

`Python >= 3.9`

```bash
pip install -r requirements.txt
```



### 2.2. Preparations

+ **datasets:** Download the dataset from https://drive.google.com/file/d/1PzePFsBcYofG1MV2FisFLBM2lMytbMdW/view?usp=sharing
+ **embedding model:** Download the pretrained MiniLM embedding model from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 
+ **generate summary embeddings:** Use `generation/generate_repre_miniLM.py`for generating summary embeddings for all users & items



### 2.4. Pretrain the Base Model

Let's start with the example LightGCN and LightGCN+FACE on Amazon dataset.

```
python encoder/train_encoder.py --model lightgcn --dataset amazon --seed 256 --cuda 0
```

Make sure the checkpoint is saved.



### 2.5. Train

After pretraining the base model, we run FACE on its basis. It includes two step for mapping and aligning respectively:

```
python encoder/train_encoder.py --model lightgcn_vq --dataset amazon --stage map --seed 256 --cuda 0
python encoder/train_encoder.py --model lightgcn_vq --dataset amazon --stage align --seed 256 --cuda 0
```
