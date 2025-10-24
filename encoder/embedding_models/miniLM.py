from transformers import AutoTokenizer, AutoModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
import pickle
from collections import Counter



class MiniLM(nn.Module):
    def __init__(self, dataset_name):
        super(MiniLM, self).__init__()
        self.dataset_name = dataset_name
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('./LLMs/all-MiniLM-L6-v2')
        self.model = BertModel.from_pretrained('./LLMs/all-MiniLM-L6-v2')
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
        tokens_df = tokens_df[~tokens_df['token'].apply(lambda x: x[0:2] == '##')]
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

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_token_embedding_matrix(self):
        return self.model.get_input_embeddings()

    def encode_text(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def encode_embeddings(self, token_embeddings, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(token_embeddings.shape[:-1], dtype=torch.long).to(token_embeddings.device)
        model_output = self.model(inputs_embeds=token_embeddings, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def get_text_token_embeddings(self, sentences):
        ids = self.tokenizer(sentences, return_tensors='pt', add_special_tokens=False)['input_ids']
        token_embeddings = self.model.get_input_embeddings()(ids)
        return token_embeddings
    
    def add_special_tokens_for_embeddings(self, token_embeddings):
        cls_token = self.model.get_input_embeddings().weight[self.tokenizer.cls_token_id].reshape(1, 1, -1).repeat(token_embeddings.shape[0], 1, 1)
        sep_token = self.model.get_input_embeddings().weight[self.tokenizer.sep_token_id].reshape(1, 1, -1).repeat(token_embeddings.shape[0], 1, 1)
        token_embeddings = torch.cat([cls_token, token_embeddings, sep_token], dim=1)
        return token_embeddings

if __name__ == "__main__":

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    sentences = ["Middle school boys who enjoy mysteries would enjoy reading Theodore Boone: The Accused. The book is suspenseful and engaging, with endearing characters and a good dose of humor. The book also serves as a lesson in the practice of law, which could be appealing to those who enjoy legal thrillers. The story's protagonist, Theo, is a relatable and charismatic young detective, and listeners will cheer for his efforts to solve the crimes. Although part of a series, this title can stand alone as a satisfying mystery for young adult readers.",\
                "The book attracts those who can be described as the following words:'mystery','race','humor','behavior','facing','heavy','driven','developed'."]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('./LLMs/all-MiniLM-L6-v2')
    model = BertModel.from_pretrained('./LLMs/all-MiniLM-L6-v2')

    # freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    print(model.get_input_embeddings().weight.shape)
    # print(model.get_input_embeddings().weight[tokenizer.cls_token_id])
    # print(model.get_input_embeddings()(torch.tensor(tokenizer.cls_token_id)))

    # # cls 101, sep 102
    # # print(tokenizer.get_vocab())
    # # print(tokenizer.convert_tokens_to_ids("[CLS]"))
    # # print(tokenizer.convert_tokens_to_ids("[SEP]"))
    # # print(tokenizer.convert_tokens_to_string(["[unused944]"]))

    # Tokenize sentences
    encoded_input = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

    # input_ids = encoded_input['input_ids']

    # print(input_ids)

    # attention_mask = encoded_input['attention_mask']

    # print(attention_mask)

    # embedding_tensor = model.get_input_embeddings()

    # input_embeddings = embedding_tensor(input_ids)

    # # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print(sentence_embeddings.shape)

    # Compute cosine similarity
    # print(F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0), dim=-1))
