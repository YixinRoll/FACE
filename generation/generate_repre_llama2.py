import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
dataset_name = 'amazon'
llm_name = "llama2"
llm_path = './LLMs/llama2-embedding'
trust_remote_code = True
embedding_dim = 1536 # for llama2

with torch.no_grad():

    with open(f'./data/{dataset_name}/usr_prf.pkl', 'rb') as f:
        usr_prf = pickle.load(f)

    with open(f'./data/{dataset_name}/itm_prf.pkl', 'rb') as f:
        itm_prf = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(llm_path, trust_remote_code=trust_remote_code)

    _ = model.cuda()

    usr_repre_np = np.zeros((len(usr_prf), embedding_dim))
    for i in range(len(usr_prf)):
        encoded_input = tokenizer([usr_prf[i]['profile']], return_tensors="pt").to(model.device)
        sentence_embeddings = model.encode(encoded_input)
        usr_repre_np[i] = sentence_embeddings.cpu().detach().numpy()

    pickle.dump(usr_repre_np, open(f'./data/{dataset_name}/usr_repre_np_{llm_name}.pkl', 'wb'))
    print(f'usr_repre_np shape: {usr_repre_np.shape}')

    itm_repre_np = np.zeros((len(itm_prf), embedding_dim))
    for i in range(len(itm_prf)):
        encoded_input = tokenizer([itm_prf[i]['profile']], return_tensors="pt").to(model.device)
        sentence_embeddings = model.encode(encoded_input)
        itm_repre_np[i] = sentence_embeddings.cpu().detach().numpy()

    pickle.dump(itm_repre_np, open(f'./data/{dataset_name}/itm_repre_np_{llm_name}.pkl', 'wb'))
    print(f'itm_repre_np shape: {itm_repre_np.shape}')
