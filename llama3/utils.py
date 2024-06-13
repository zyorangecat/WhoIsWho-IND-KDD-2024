import json
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from imblearn.under_sampling import RandomUnderSampler

class INDDataSetForMistral(Dataset):
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSetForMistral, self).__init__()
        self.author, self.pub = dataset
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()
        train_keys = []
        labels = []
        for key in author_keys:
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                })
                labels.append(0)
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)

        keys_ids = list(range(0, len(train_keys)))
        keys_ids = [[x, labels[x]] for x in keys_ids]
        keys_ids_0 = [i  for i in keys_ids if i[1] == 0]
        keys_ids_1 = [i  for i in keys_ids if i[1] == 1]


        rus = RandomUnderSampler(random_state=0)
        keys_ids = list(range(0, len(train_keys)))
        keys_ids = [[x, 0] for x in keys_ids]
        sampled_keys, _ = rus.fit_resample(keys_ids, labels)
        self.train_keys = [train_keys[i[0]] for i in sampled_keys]
        random.shuffle(self.train_keys)
        # self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper information: \n ### {} \n ### Does the paper ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)

        # 4是因为有inst_begin_tokens， inst_end_tokens， bos_token_id， eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_len = self.max_source_length + self.max_target_length + 4

    def __len__(self):
        return len(self.train_keys)

    def process(self, title='', authors='', abstract='', keywords=''):
        if len(keywords) == 0:
            keywords = 'unknown'
        else:
            keywords = ','.join(keywords[:5])

        if len(abstract) == 0:
            abstract = 'unknown'

        title = title.lower().strip()
        keywords = keywords.lower().strip()
        abstract = abstract.lower().strip()
        abstract = abstract.replace("\n\n", ' ')

        title = title if len(self.tokenizer.tokenize(title)) < 200 else ' '.join(title.split(' ')[:100])
        abstract = abstract if len(self.tokenizer.tokenize(abstract)) < 1000 else ' '.join(abstract.split(' ')[:500])

        if len(title) > 0:
            text = f"Paper Title: {title} # Keywords: {keywords} # Abstract: {abstract}"
        else:
            return ''
        return text

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] + self.author[self.train_keys[index]['author']]['outliers']
        profile_process = []
        for p in profile:
            if p != self.train_keys[index]['pub'] and p in self.pub:
                title = self.pub[p].get('title', '')
                authors = self.pub[p].get('authors', '')
                abstract = self.pub[p].get('abstract', '')
                keywords = self.pub[p].get('keywords', '')

                text = self.process(title=title, authors=authors, abstract=abstract, keywords=keywords)
                if text:
                    profile_process.append(text)
        random.shuffle(profile_process)

        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile_process]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len > self.max_source_length - 500:
            total_len = 0
            p = 0
            while total_len < self.max_source_length - 500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile_process = profile_process[:p - 1]

        profile_text = '\n\n'.join(profile_process)

        title = self.pub[self.train_keys[index]['pub']].get('title', '')
        authors = self.pub[self.train_keys[index]['pub']].get('authors', '')
        abstract = self.pub[self.train_keys[index]['pub']].get('abstract', '')
        keywords = self.pub[self.train_keys[index]['pub']].get('keywords', '')
        pub_text = self.process(title=title, authors=authors, abstract=abstract, keywords=keywords)
        context = self.instruct.format(profile_text, pub_text)


        human_tokens = self.tokenizer.encode(text=context, add_special_tokens=False, truncation=True,
                                             max_length=self.max_source_length)
        assistant_tokens = self.yes_token if self.train_keys[index]['label'] else self.no_token

        tmp_input_ids = [
                            self.bos_token_id] + self.inst_begin_tokens + human_tokens + self.inst_end_tokens + assistant_tokens + [
                            self.eos_token_id]

        if len(tmp_input_ids) > self.max_seq_len:
            ll = len(human_tokens) - (len(tmp_input_ids) - self.max_seq_len)
            human_tokens = human_tokens[:ll]

        del tmp_input_ids

        input_tokens = self.inst_begin_tokens + human_tokens + self.inst_end_tokens
        output_tokens = assistant_tokens + [self.eos_token_id]
        input_ids = [self.bos_token_id] + input_tokens + output_tokens

        target_mask = [0] + [0] * len(input_tokens) + [1] * len(output_tokens)
        attention_mask = [1] * len(input_ids)


        assert len(input_ids) == len(target_mask) == len(attention_mask)

        return {
            "input_ids": input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }


class INDDataSetForLLama(Dataset):
    '''
        iteratively return the profile of each author
    '''

    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSetForLLama, self).__init__()
        self.author, self.pub = dataset
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()
        train_keys = []
        labels = []
        for key in author_keys:
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                })
                labels.append(0)
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)
        rus = RandomUnderSampler(random_state=0)
        keys_ids = list(range(0, len(train_keys)))
        keys_ids = [[x, 0] for x in keys_ids]
        sampled_keys, _ = rus.fit_resample(keys_ids, labels)
        self.train_keys = [train_keys[i[0]] for i in sampled_keys]
        random.shuffle(self.train_keys)
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

        self.eot_token_id = tokenizer.encode('<|eot_id|>', add_special_tokens=False)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        system = 'You are a helpful assistant.'
        system_text = f'<|start_header_id|>system<|end_header_id|>\n\n{system}'
        self.system_token = self.tokenizer.encode(system_text, add_special_tokens=False)

        user_text = f"<|start_header_id|>user<|end_header_id|>\n\n"
        self.user_token = self.tokenizer.encode(user_text, add_special_tokens=False)

        assistant_text = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.assistant_token = self.tokenizer.encode(assistant_text, add_special_tokens=False)

        self.max_seq_len = self.max_source_length + self.max_target_length + len(self.system_token) + len(
            self.user_token) + len(self.assistant_token) + 4

    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):

        profile = self.author[self.train_keys[index]['author']]['normal_data'] + \
                  self.author[self.train_keys[index]['author']]['outliers']
        profile = [self.pub[p]['title'] for p in profile if
                   p != self.train_keys[index]['pub']]
        random.shuffle(profile)

        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len > self.max_source_length - 500:
            total_len = 0
            p = 0
            while total_len < self.max_source_length - 500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p - 1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title)) < 200 else ' '.join(
            title.split(' ')[:100])
        context = self.instruct.format(profile_text, title)

        input_tokens = self.tokenizer.encode(text=context, add_special_tokens=False, truncation=True,
                                             max_length=self.max_source_length)
        label_tokens = self.yes_token if self.train_keys[index]['label'] else self.no_token

        input_ids = [
                        self.bos_token_id] + self.system_token + self.eot_token_id + self.user_token + input_tokens + self.eot_token_id + self.assistant_token + label_tokens + self.eot_token_id

        target_mask = [0] + [0] * len(self.system_token) + [0] + [0] * len(self.user_token + input_tokens) + [0] + [
            0] * len(self.assistant_token) + [1] * len(label_tokens) + [1]
        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[:self.max_seq_len]
        target_mask = target_mask[:self.max_seq_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        return {
            "input_ids": input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }

class DataCollatorForLLama(object):
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 取出batch中的最大长度，如果超过max_seq_len，则取max_seq_len
        batch_max_len = min(max(lengths), self.max_seq_len)
        # batch_max_len = self.max_seq_len
        # print(">>>>>> batch", batch[0])
        # print(">>>>>> lengths", lengths)
        # print(">>>>>> batch_max_len", batch_max_len)
        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
            target_mask = target_mask[:self.max_seq_len]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        # print(">>>>>> input_ids_batch", input_ids_batch)
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        label = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': label
        }
        return inputs

class IND4EVALLlma3(Dataset):
    def __init__(self, dataset, tokenizer, max_source_length):
        super(IND4EVALLlma3, self).__init__()
        self.author, self.pub = dataset
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        author_keys = self.author.keys()

        self.val_set = []
        if 'normal_data' in self.author[list(author_keys)[0]]:
            for key in author_keys:
                for pub_key in self.author[key]['normal_data']:
                    self.val_set.append({
                        'pub': pub_key,
                        'author': key,
                        'label': 1
                    })
                for pub_key in self.author[key]['outliers']:
                    self.val_set.append({
                        'pub': pub_key,
                        'author': key,
                        'label': 0
                    })
        elif 'papers' in self.author[list(author_keys)[0]]:
            for key in author_keys:
                for pub_key in self.author[key]['papers']:
                    self.val_set.append({
                        'pub': pub_key,
                        'author': key,
                    })
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

        self.eot_token_id = tokenizer.encode('<|eot_id|>', add_special_tokens=False)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        system = 'You are a helpful assistant.'
        system_text = f'<|start_header_id|>system<|end_header_id|>\n\n{system}'
        self.system_token = self.tokenizer.encode(system_text, add_special_tokens=False)

        user_text = f"<|start_header_id|>user<|end_header_id|>\n\n"
        self.user_token = self.tokenizer.encode(user_text, add_special_tokens=False)

        assistant_text = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.assistant_token = self.tokenizer.encode(assistant_text, add_special_tokens=False)

    def __len__(self):
        return len(self.val_set)

    def __getitem__(self, index):
        if "normal_data" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['normal_data'] + \
                      self.author[self.val_set[index]['author']]['outliers']
        elif "papers" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['papers']
        else:
            raise ("No profile found")
        profile = [self.pub[p]['title'] for p in profile if
                   p != self.val_set[index]['pub']]  # delete disambiguate paper
        random.shuffle(profile)

        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len > self.max_source_length - 500:
            total_len = 0
            p = 0
            while total_len < self.max_source_length - 500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p - 1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.val_set[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title)) < 200 else ' '.join(title.split(' ')[:100])
        context = self.instruct.format(profile_text, title)

        input_tokens = self.tokenizer.encode(text=context, add_special_tokens=False, truncation=False)
        input_ids = [
                        self.bos_token_id] + self.system_token + self.eot_token_id + self.user_token + input_tokens + self.eot_token_id + self.assistant_token
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "author": self.val_set[index]['author'],
            "pub": self.val_set[index]['pub'],
        }