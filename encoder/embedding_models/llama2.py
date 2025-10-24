from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
import pickle
from collections import Counter



class LLaMA2(nn.Module):
    def __init__(self, dataset_name):
        super(LLaMA2, self).__init__()
        self.dataset_name = dataset_name
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('./LLMs/llama2-embedding', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('./LLMs/llama2-embedding', trust_remote_code=True)
        # freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _coca60000_vocabulary(self, path = './data/vocabulary/word_frequency_list_60000_English.xlsx', voc_word_num = 20000, pos = ['N','J']):
        coca_df = pd.read_excel(path)
        coca_df = coca_df[['PoS','word']]
        coca_df = coca_df.map(lambda x: x.strip() if type(x) == str else x)
        coca_df = coca_df[coca_df['word'].apply(lambda x: len(x) > 0)] # remove the words with length 0
        coca_df['word'] = coca_df['word'].apply(lambda x: x[1:-1] if x[0] == '(' and x[-1] == ')' else x) # remove the brackets of the word
        coca_df.drop_duplicates(subset=['word'], inplace=True, keep='first') # keep the most frequent PoS of each word
        coca_df = coca_df[coca_df['PoS'].isin(pos)] # only keep the words with pos in pos
        coca_df = coca_df.iloc[:voc_word_num] # only keep the first word_num words
        return coca_df['word'].tolist()
    
    def _profile_vocabulary(self):
        usrprf_path = f"./data/{self.dataset_name}/usr_prf.pkl"
        itmprf_path = f"./data/{self.dataset_name}/itm_prf.pkl"
        with open(usrprf_path, 'rb') as f:
            usrprf = pickle.load(f)
            usrprf = [v['profile'] for v in usrprf.values()]
        with open(itmprf_path, 'rb') as f:
            itmprf = pickle.load(f)
            itmprf = [v['profile'] for v in itmprf.values()]
        prfs = usrprf + itmprf
        prfs = "\n".join(prfs)
        prfs = prfs.lower()
        prfs_split = re.split(r'[^a-zA-Z]+', prfs)
        word_freq = Counter(prfs_split)
        common_words = [word for word, freq in word_freq.items() if freq > 100]
        return common_words

    def filter_and_get_vocabulary(self):
        vocal_dict = self.tokenizer.get_vocab()
        tokens = vocal_dict.keys()
        tokens_list = list(tokens)

        tokens_df = pd.DataFrame()
        tokens_df['token'] = tokens_list
        # the id of token
        tokens_df['token_id'] = tokens_df['token'].apply(self.tokenizer.convert_tokens_to_ids)
        # remove the non-suffix token
        tokens_df = tokens_df[tokens_df['token'].apply(lambda x: x[0] == '▁')]
        # to string
        tokens_df['token'] = tokens_df['token'].apply(lambda x: self.tokenizer.convert_tokens_to_string([x]))
        tokens_df['token'] = tokens_df['token'].apply(lambda x: x.strip())
        # keep the English words
        tokens_df = tokens_df[tokens_df['token'].str.fullmatch(r'^[a-z]+$')]
        # load the vocabulary of COCA60000
        vocabulary_coca60000 = self._coca60000_vocabulary()
        # vocabulary_profile = self._profile_vocabulary()

        # find the intersection of the vocabulary of COCA60000 and the tokens
        tokens_df = tokens_df[tokens_df['token'].isin(vocabulary_coca60000)]
        # tokens_df = tokens_df[tokens_df['token'].isin(vocabulary_profile)]

        # sort the tokens by token_id
        tokens_df.sort_values(by='token_id', inplace=True)
        return tokens_df


    def get_token_embedding_matrix(self):
        return self.model.get_input_embeddings()

    def encode_text(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        sentence_embeddings = self.model.encode({'input_ids': encoded_input['input_ids'], 'attention_mask': encoded_input['attention_mask']})
        return sentence_embeddings
    
    def encode_embeddings(self, token_embeddings, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(token_embeddings.shape[:-1], dtype=torch.long).to(token_embeddings.device)
        sentence_embeddings = self.model.encode({"inputs_embeds": token_embeddings, "attention_mask": attention_mask})
        return sentence_embeddings
    
    def get_text_token_embeddings(self, sentences):
        ids = self.tokenizer(sentences, return_tensors='pt', add_special_tokens=False)['input_ids']
        token_embeddings = self.model.get_input_embeddings()(ids)
        return token_embeddings
    
    def add_special_tokens_for_embeddings(self, token_embeddings):
        bos_token = self.model.get_input_embeddings().weight[self.tokenizer.bos_token_id].reshape(1, 1, -1).repeat(token_embeddings.shape[0], 1, 1)
        token_embeddings = torch.cat([bos_token, token_embeddings], dim=1)
        return token_embeddings

if __name__ == "__main__":

    llama2 = LLaMA2('amazon')
    ids = llama2.tokenizer('hello world', return_tensors='pt', add_special_tokens=False)['input_ids']
    ids_2 = llama2.tokenizer('hello world', return_tensors='pt')['input_ids']
    print(llama2.tokenizer.cls_token_id)
    print(llama2.tokenizer.sep_token_id)
    print(llama2.tokenizer.pad_token_id)
    print(llama2.tokenizer.bos_token_id)
    print(llama2.tokenizer.eos_token_id)

    print(ids)
    print(ids_2)