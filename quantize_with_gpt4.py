import sys
import os
import json
from enum import Enum
from itertools import chain
import fire

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


class MODEL_TYPE(str, Enum):
    Baichuan = "baichuan"
    Qwen = "qwen"


def generate_prompt(model_type, instruction, response=None):
    if model_type in [MODEL_TYPE.Baichuan]:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        if response is not None:
            prompt += response + '</s>'
        return prompt
    elif model_type in [MODEL_TYPE.Qwen]:
        prompt = f"<|im_start|>### Instruction:\n{instruction}\n\n### Response:\n"
        if response is not None:
            prompt += response + '<|endoftext|>'
        return prompt
    else:
        prompt = instruction
        if response is not None:
            prompt += response
        return prompt
        

def load_gpt4_examples(file, shuffle=False, max_example=-1):
    with open(file) as f:
        gpt4 = json.load(f)
    
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(gpt4)
    if max_example != -1:
        gpt4 = gpt4[:max_example]
        
    examples = []
    for example in gpt4:
        assert example['items'][0]['from'] == 'human'
        assert example['items'][1]['from'] == 'gpt'
        examples.append({
            'instruction': example['items'][0]['value'],
            'output': example['items'][1]['value']
        })
    return examples


def main(
    pretrained_model_dir: str,
    quantized_model_dir: str,
    model_type: str = "baichuan",
    quant_batch_size: int = 1,
    text_cutoff_length: int = 1024 # 2048 for qwen
):
    model_type = MODEL_TYPE(model_type)  
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=False, trust_remote_code=True)
    
    gpt4_zh = load_gpt4_examples('calibration_data/gpt4/Chinese_cleaned_filtered.json')
    gpt4_en = load_gpt4_examples('calibration_data/gpt4/Chinese_English_cleaned_filtered.json', True, len(gpt4_zh))
    gpt4_zh = Dataset.from_list(gpt4_zh)
    gpt4_en = Dataset.from_list(gpt4_en)
    gpt4 = concatenate_datasets([gpt4_zh, gpt4_en]).shuffle(seed=42)
    
    # It seems that there might be an issue with identity recognition after the quantization of qwen, so add some hardcoded data"
    if model_type == MODEL_TYPE.Qwen:
        with open('calibration_data/identity/hardcoded.json') as f:
            np.random.seed(42)
            hardcoded = []
            for item in np.random.choice(json.load(f), 50, replace=False).tolist():
                hardcoded.append({
                    'instruction': item['conversations'][0]['value'],
                    'output': item['conversations'][1]['value']
                })
            hardcoded = Dataset.from_list(hardcoded)
        gpt4 = concatenate_datasets([gpt4, hardcoded]).shuffle(seed=42)
        
    
    def tokenize(data_point):
        prompt = generate_prompt(model_type, data_point['instruction'], data_point['output'])
        if model_type in [MODEL_TYPE.Baichuan]:
            return tokenizer(prompt)
        else:
            return tokenizer(prompt, truncation=True, max_length=text_cutoff_length)
    
    
    def group_texts(examples):
        concatenate_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenate_examples[list(examples.keys())[0]])
        if total_length > text_cutoff_length:
            total_length = (total_length // (text_cutoff_length * quant_batch_size)) * text_cutoff_length * quant_batch_size
        result = {
            k: [v[i: i + text_cutoff_length] for i in range(0, total_length, text_cutoff_length)]
            for k, v in concatenate_examples.items()
        }
        return result
    
    
    gpt4, unpadded = gpt4.map(tokenize).remove_columns(list(gpt4.features)), False
    if model_type in [MODEL_TYPE.Baichuan]:
        gpt4, unpadded = Dataset.from_dict(group_texts(gpt4.to_dict())), True
    
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model into the 1st GPU
    max_memory = {0: torch.cuda.get_device_properties(0).total_memory}
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_dir, quantize_config, trust_remote_code=True, torch_dtype=torch.float16, max_memory=max_memory
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(gpt4, batch_size=quant_batch_size, unpadded=unpadded)

    # save quantized model
    model.save_quantized(quantized_model_dir)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    fire.Fire(main)
