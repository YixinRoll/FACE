import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
dataset_name = 'steam'
llm_name = "miniLM"
llm_path = './LLMs/all-MiniLM-L6-v2'
embedding_dim = 384 # for miniLM

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with torch.no_grad():

    with open(f'./data/{dataset_name}/usr_prf.pkl', 'rb') as f:
        usr_prf = pickle.load(f)

    with open(f'./data/{dataset_name}/itm_prf.pkl', 'rb') as f:
        itm_prf = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModel.from_pretrained(llm_path)

    _ = model.cuda()

    usr_repre_np = np.zeros((len(usr_prf), embedding_dim))
    for i in range(len(usr_prf)):
        encoded_input = tokenizer([usr_prf[i]['profile']], return_tensors="pt").to(model.device)
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        usr_repre_np[i] = sentence_embeddings.cpu().detach().numpy()

    pickle.dump(usr_repre_np, open(f'./data/{dataset_name}/usr_repre_np_{llm_name}.pkl', 'wb'))
    print(f'usr_repre_np shape: {usr_repre_np.shape}')

    itm_repre_np = np.zeros((len(itm_prf), embedding_dim))
    for i in range(len(itm_prf)):
        encoded_input = tokenizer([itm_prf[i]['profile']], return_tensors="pt").to(model.device)
        for k_ in encoded_input: # input_ids, attention_mask
            encoded_input[k_] = encoded_input[k_].cuda()
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        itm_repre_np[i] = sentence_embeddings.cpu().detach().numpy()

    pickle.dump(itm_repre_np, open(f'./data/{dataset_name}/itm_repre_np_{llm_name}.pkl', 'wb'))
    print(f'itm_repre_np shape: {itm_repre_np.shape}')
