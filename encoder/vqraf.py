import re
import os
import random
import pickle
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pandas as pd
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def get_embedding_model(llm_name, dataset_name):
    module_path = '.'.join(['embedding_models', llm_name])
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == llm_name.lower():
            return getattr(module, attr)(dataset_name)


class Quantizer(nn.Module):
    def __init__(self, codebook_dim = 256, word_num = 8, dataset_name = 'amazon', llm_name="llama2"):
        super().__init__()
        self.word_num = word_num
        self.dataset_name = dataset_name

        self.embedding_model = get_embedding_model(llm_name, dataset_name)

        # filter vocabulary
        tokens_df = self.embedding_model.filter_and_get_vocabulary() # two columns: token_id, token
        tokens_df.to_csv(f"./data/vocabulary/vocabulary_{dataset_name}_{llm_name}.csv", index=False) # save the filtered vocabulary

        # get token embedding & construct linear mapping
        self.token_id = torch.tensor(tokens_df['token_id'].values)
        self.vocabulary: list[str] = tokens_df['token'].tolist()
        input_embeddings = self.embedding_model.get_token_embedding_matrix()
        codebook_tensor = input_embeddings(self.token_id).detach() # no grad
        self.register_buffer('codebook_tensor_pca', codebook_tensor)
        self.codebook_mapping = nn.Linear(codebook_tensor.shape[-1], codebook_dim)

        # TODO: use pca for dimensional reduction first
        # from torch_pca import PCA

        itm_dict = {'amazon': 'book', 'yelp': 'restaurant', 'steam': 'game'}

        # prompt
        prompt_usr = "The user and his likes can be described as the following words:"
        prompt_itm = f"The {itm_dict[dataset_name]} attracts those who can be described as the following words:"
        prompt_embedding = self.embedding_model.get_text_token_embeddings([prompt_usr, prompt_itm, prompt_itm])
        prompt_comma = self.embedding_model.get_text_token_embeddings(",")
        
        # ids_usr = self.tokenizer(prompt_usr, return_tensors="pt")["input_ids"]
        # ids_itm = self.tokenizer(prompt_itm, return_tensors="pt")["input_ids"]
        # prompt_embedding_usr = self.model.get_input_embeddings()(ids_usr).detach()
        # prompt_embedding_itm = self.model.get_input_embeddings()(ids_itm).detach()
        # prompt_embedding = torch.cat([prompt_embedding_usr, prompt_embedding_itm, prompt_embedding_itm], dim=0)
        # ids_comma = self.tokenizer(",", return_tensors="pt")["input_ids"][:,1:]
        # prompt_comma = self.model.get_input_embeddings()(ids_comma).detach()

        self.register_buffer('prompt_embedding', prompt_embedding)
        self.register_buffer('prompt_comma', prompt_comma)

    def reverse_codebook_mapping(self, x):
        reverse_weight = torch.pinverse(self.codebook_mapping.weight.detach()).t()
        x = x - self.codebook_mapping.bias.detach()
        x = torch.matmul(x, reverse_weight)
        return x
    
    def forward(self, z_e):
        if hasattr(self, 'codebook_mapping'):
            mapped_codebook = self.codebook_mapping(self.codebook_tensor_pca)
        else:
            mapped_codebook = self.codebook_tensor_pca

        dist_matrix = torch.sum(z_e**2, dim=1, keepdim=True) + \
                        torch.sum(mapped_codebook**2, dim=1) - 2 * \
                        torch.matmul(z_e, mapped_codebook.t())
        
        dist_matrix = dist_matrix.reshape(-1, self.word_num, dist_matrix.shape[-1])
        batch_indices = torch.arange(dist_matrix.shape[0], device=dist_matrix.device).unsqueeze(1)
        
        min_dist_indices = None
        for i in range(self.word_num):
            word_i_min_dist_indices = torch.argmin(dist_matrix[:,i,:], dim=1, keepdim=True)
            dist_matrix[batch_indices, :, word_i_min_dist_indices] = float('inf')
            if min_dist_indices is None:
                min_dist_indices = word_i_min_dist_indices
            else:
                min_dist_indices = torch.cat([min_dist_indices, word_i_min_dist_indices], dim=1)
        min_dist_indices = min_dist_indices.reshape(-1)

        z_q = mapped_codebook[min_dist_indices]

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach() 

        # ||e-sg[z_e]||^2
        vq_loss = torch.mean((z_q - z_e.detach())**2)
        # ||z_e-sg[e]||^2
        commitment_loss = torch.mean((z_e - z_q.detach())**2)

        return z_q_st, 0.75 * vq_loss + 0.25 * commitment_loss
    

    def explain(self, codebook_tensor_indices: list[int]):
        return [self.vocabulary[idx] for idx in codebook_tensor_indices]
    
    def forward_explain(self, z_e):
        if hasattr(self, 'codebook_mapping'):
            mapped_codebook = self.codebook_mapping(self.codebook_tensor_pca)
        else:
            mapped_codebook = self.codebook_tensor_pca

        dist_matrix = torch.sum(z_e**2, dim=1, keepdim=True) + \
                        torch.sum(mapped_codebook**2, dim=1) - 2 * \
                        torch.matmul(z_e, mapped_codebook.t())
        
        dist_matrix = dist_matrix.reshape(-1, self.word_num, dist_matrix.shape[-1])
        batch_indices = torch.arange(dist_matrix.shape[0], device=dist_matrix.device).unsqueeze(1)
        
        min_dist_indices = None
        for i in range(self.word_num):
            word_i_min_dist_indices = torch.argmin(dist_matrix[:,i,:], dim=1, keepdim=True)
            dist_matrix[batch_indices, :, word_i_min_dist_indices] = float('inf')
            if min_dist_indices is None:
                min_dist_indices = word_i_min_dist_indices
            else:
                min_dist_indices = torch.cat([min_dist_indices, word_i_min_dist_indices], dim=1)
        min_dist_indices = min_dist_indices.reshape(-1)

        # min_dist, min_dist_indices = torch.min(dist_matrix, dim=1)
        z_q = mapped_codebook[min_dist_indices]
        return z_q, self.explain(min_dist_indices.tolist())
    
    def get_collaborative_representations(self, z_e):
        z_e = z_e.reshape(-1, z_e.shape[-1]) # batch_size*word_num, word_dim
        words_embedding = self.reverse_codebook_mapping(z_e) # batch_size*word_num, token_dim
        # if words_embedding.requires_grad:
        #     words_embedding.register_hook(lambda grad: print("Gradient:", grad)) # register hook to print gradient
        words_embedding = words_embedding.reshape(-1, self.word_num, words_embedding.shape[-1]) # batch_size, word_num, token_dim

        words_embedding_comma = torch.zeros(words_embedding.shape[0], words_embedding.shape[1] * 2 - 1, words_embedding.shape[2], device=words_embedding.device)
        words_embedding_comma[:,::2,:] = words_embedding
        words_embedding_comma[:,1::2,:] = self.prompt_comma.squeeze()

        batch_prompt_embedding = self.prompt_embedding.repeat(words_embedding_comma.shape[0] // 3, 1, 1)
        combined_embedding = torch.cat([batch_prompt_embedding, words_embedding_comma], dim=1)

        combined_embedding = self.embedding_model.add_special_tokens_for_embeddings(combined_embedding)
        collaborative_representations = self.embedding_model.encode_embeddings(token_embeddings=combined_embedding)

        # attention_mask = torch.ones(combined_embedding.shape[0], combined_embedding.shape[1], dtype=torch.long, device=combined_embedding.device)
        # collaborative_representations = self.model.encode(features={"inputs_embeds": combined_embedding, "attention_mask": attention_mask})

        return collaborative_representations
    
        
class LinearLayer(nn.Module):
    def __init__(self, input_dim, mlp_num, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(mlp_num, input_dim, output_dim))
        nn.init.orthogonal_(self.weights)
        self.bias = nn.Parameter(torch.randn(mlp_num, output_dim))

        self.reverse_linear = nn.Linear(mlp_num*output_dim, input_dim)
        
    def forward(self, x):
        x = torch.einsum('bi,cio->bco', x, self.weights)
        x = x + self.bias
        return x
    
    def compute_pinv_weights(self):
        weights_pinv = []
        for i in range(self.weights.size(0)):
            weight = self.weights[i].detach() 
            weight_pinv = torch.pinverse(weight)
            weights_pinv.append(weight_pinv)
        weights_pinv = torch.stack(weights_pinv, dim=0)
        return weights_pinv
    
    def reverse_pinv(self, x):
        weights_pinv = self.compute_pinv_weights()
        x = x - self.bias.detach()
        x = torch.einsum('bco,coi->bci', x, weights_pinv)
        x = x.mean(dim=1)
        return x
    
    def reverse(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.reverse_linear(x)
        return x
    

class VQRAF(nn.Module):
    def __init__(self, input_dim, word_num, word_dim, dataset_name, llm_name):
        super().__init__()
        self.input_dim = input_dim
        self.word_num = word_num
        self.word_dim = word_dim
        nhead = 1
        num_layers = 1

        self.linear_encoder = LinearLayer(input_dim=self.input_dim, mlp_num=self.word_num, output_dim=self.word_dim)
        self.transformer_layer = TransformerEncoderLayer(d_model=self.word_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.transformer_decoder = TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.quantizer = Quantizer(codebook_dim=self.word_dim, word_num = self.word_num, dataset_name=dataset_name, llm_name=llm_name)

    def forward(self, x, stage):
        # Encoder
        linear_out = self.linear_encoder(x) # batch_size, word_num, word_dim

        # # Transformer
        z_e = self.transformer_encoder(linear_out) # batch_size, word_num, word_dim

        # VQ
        z_e_reshape = z_e.reshape(-1, self.word_dim) # batch_size*word_num, word_dim

        z_q_reshape, vq_loss = self.quantizer(z_e_reshape) # do vq

        z_q = z_q_reshape.reshape(x.shape[0], self.word_num, self.word_dim) # batch_size, word_num, word_dim

        if stage == "map":
            collaborative_representations_2 = None
        else:
            collaborative_representations_2 = self.quantizer.get_collaborative_representations(z_q)

        # Decoder
        trans_decoder_out = self.transformer_decoder(z_q) # batch_size, word_num, word_dim

        decoded = self.linear_encoder.reverse(trans_decoder_out)

        recons_loss = F.mse_loss(decoded, x.detach())
        
        return decoded, vq_loss, recons_loss, collaborative_representations_2
    

        