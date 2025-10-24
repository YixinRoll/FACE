from transformers import AutoTokenizer, AutoModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
import pickle
from collections import Counter



class Qwen2(nn.Module):
    def __init__(self, dataset_name):
        super(Qwen2, self).__init__()
        self.dataset_name = dataset_name
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('./LLMs/gte-Qwen2', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('./LLMs/gte-Qwen2', trust_remote_code=True)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        # freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Qwen Instruct
        task_description = 'Given a query composed of 8 descriptors, retrieve the original user/item profile'
        task_prompt =  f'Instruct: {task_description}\nQuery:' 
        prompt_task = self.get_text_token_embeddings(task_prompt)
        self.register_buffer('prompt_task', prompt_task)

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
        tokens_df = tokens_df[tokens_df['token'].apply(lambda x: x[0] == 'Ġ')]
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

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_token_embedding_matrix(self):
        return self.model.get_input_embeddings()

    def encode_text(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.last_token_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def encode_embeddings(self, token_embeddings, attention_mask = None):
        if attention_mask is None:
            attention_mask = torch.ones(token_embeddings.shape[:-1], dtype=torch.long).to(token_embeddings.device)
        model_output = self.model(inputs_embeds=token_embeddings, attention_mask=attention_mask)
        sentence_embeddings = self.last_token_pool(model_output.last_hidden_state, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def get_text_token_embeddings(self, sentences):
        ids = self.tokenizer(sentences, return_tensors='pt', add_special_tokens=False)['input_ids']
        token_embeddings = self.model.get_input_embeddings()(ids)
        return token_embeddings
    
    def add_special_tokens_for_embeddings(self, token_embeddings):
        batch_task_embedding = self.prompt_task.repeat(token_embeddings.shape[0], 1, 1)
        pad_token = self.model.get_input_embeddings().weight[self.tokenizer.pad_token_id].reshape(1, 1, -1).repeat(token_embeddings.shape[0], 1, 1)
        token_embeddings = torch.cat([batch_task_embedding, token_embeddings, pad_token], dim=1)
        return token_embeddings

if __name__ == "__main__":

    # embedding_model = EmbeddingModel('./LLMs/gte-Qwen2', 'amazon')
    # filter_and_get_vocabulary = embedding_model.filter_and_get_vocabulary()

    def last_token_pool(last_hidden_states,
                    attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'


    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, 'how much protein should a female eat'),
        get_detailed_instruct(task, 'summit define')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    ]
    input_texts = queries + documents

    tokenizer = AutoTokenizer.from_pretrained('./LLMs/gte-Qwen2', trust_remote_code=True)
    model = AutoModel.from_pretrained('./LLMs/gte-Qwen2', trust_remote_code=True)


    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)

    # print(batch_dict['input_ids'])
    # print(batch_dict['attention_mask'])
    print(model.get_input_embeddings())
    # print(tokenizer.get_vocab())


    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    print(embeddings.shape)

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores.tolist())