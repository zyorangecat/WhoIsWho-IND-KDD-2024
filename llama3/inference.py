# -*- coding: utf-8 -*-

import os
from peft import PeftModel,get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from utils import IND4EVAL, IND4EVALLlma3
import json
from accelerate import Accelerator
from tqdm import tqdm
from metric import compute_metric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', help='The path to the lora file',default="saved_dir/checkpoint-100")
parser.add_argument('--model_path',default='ZhipuAI/chatglm3-6b-32k')
parser.add_argument('--pub_path', help='The path to the pub file',default='test_pub.json')
parser.add_argument('--eval_path',default='eval_data.json')
parser.add_argument('--saved_dir',default='eval_result')
args = parser.parse_args()

checkpoint = args.lora_path.split('/')[-1]

accelerator = Accelerator()
device = torch.device(0)

batch_size = 1

model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=False, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
lora_model = PeftModel.from_pretrained(model, args.lora_path).half()
print('done loading peft model')
YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with open(args.pub_path, "r" , encoding = "utf-8") as f:
    pub_data = json.load(f)
with open(args.eval_path, "r", encoding="utf-8") as f: 
    eval_data = json.load(f)
eval_dataset = IND4EVALLlma3(
    (eval_data,pub_data),
    tokenizer,
    max_source_length = 8196,
) 
print('done reading dataset')


def collate_fn(batch):
    batch = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
    batch_max_len = max(batch)

    input_ids_batch, attention_mask_batch = [], []
    for x in batch:
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        padding_len = batch_max_len - len(input_ids)

        attention_mask = [0] * padding_len + attention_mask
        input_ids = [tokenizer.pad_token_id] * padding_len + input_ids

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)

    input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
    attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)

    batch = {k: [item[k] for item in batch] for k in ('author', 'pub')}
    batch_input = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch
        }
    # batch_input = tokenizer(
    #     batch['input_ids'],
    #     padding='longest',
    #     truncation=False,
    #     return_tensors="pt",
    #     add_special_tokens=False,
    # )
    return batch_input, batch['author'], batch['pub']

dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size ,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)
model.eval()
result = []


YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with torch.no_grad():
    for index,batch in tqdm(enumerate(val_data)):
        batch_input, author, pub = batch

        response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 16, return_dict_in_generate=True, output_scores=True)

        yes_prob, no_prob = response.scores[0][:,YES_TOKEN_IDS],response.scores[0][:,NO_TOKEN_IDS]
        logit = yes_prob/(yes_prob+no_prob)
        node_result = [(author[i],pub[i],logit[i].item()) for i in range(batch_size)]
        batch_result = accelerator.gather_for_metrics(node_result)
        if accelerator.is_main_process:
            result.extend(batch_result)

if accelerator.is_main_process: 
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    res_list = {}
    for i in result:
        [aid,pid,logit] = i
        if aid not in res_list.keys():
            res_list[aid] = {}
        res_list[aid][pid] = logit
    with open(f'{args.saved_dir}/result-{checkpoint}.json', 'w') as f:
        json.dump(res_list, f)
