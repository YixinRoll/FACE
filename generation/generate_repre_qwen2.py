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
llm_name = "qwen2"
llm_path = './LLMs/gte-Qwen2'
trust_remote_code = True
embedding_dim = 1536 # for Qwen2

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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
        model_output = model(**encoded_input)
        sentence_embeddings = last_token_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
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
        sentence_embeddings = last_token_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        itm_repre_np[i] = sentence_embeddings.cpu().detach().numpy()

    pickle.dump(itm_repre_np, open(f'./data/{dataset_name}/itm_repre_np_{llm_name}.pkl', 'wb'))
    print(f'itm_repre_np shape: {itm_repre_np.shape}')
