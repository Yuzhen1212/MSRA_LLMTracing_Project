# import requests
# response = requests.get('https://huggingface.co/meta-llama/Llama-2-13b-chat-hf')
# print(response.status_code)



# import os
# import requests
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # 设置 token
# token = 'your_hugging_face_token_here'
# headers = {"Authorization": f"Bearer {token}"}

# # 尝试访问模型的配置文件
# model_name_or_path = 'meta-llama/Llama-2-13b-chat-hf'
# config_url = f"https://huggingface.co/{model_name_or_path}/resolve/main/config.json"

# response = requests.get(config_url, headers=headers)

# if response.status_code == 200:
#     print("Token has access to the model.")
# else:
#     print(f"Failed to access model. Status code: {response.status_code}")
#     print(response.text)

import os
import sys
import json
import math
import nltk
import torch
import argparse
import tiktoken
import numpy as np
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer



class LLaMA2Estimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=False).half().to(device)

    def get_prompt(self, message, chat_history, system_prompt):
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n[/INST]\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message}')
        return ''.join(texts)

    def estimate(self, context, n, temp, max_token):
        system_prompt = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
        prompt = self.get_prompt(context, [], system_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt







model_path = 'meta-llama/Llama-2-13b-chat-hf'
device = torch.device('cuda:0')  

llama2_estimator = LLaMA2Estimator(model_name_or_path=model_path, device=device)


context = "Doing LLM Detection Project"
n = 5  
temp = 0.7  
max_token = 50  

results = llama2_estimator.estimate(context, n, temp, max_token)
print(results)
