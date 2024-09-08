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




# def get_root_word(word):
#     from nltk.stem import SnowballStemmer
#     stemmer = SnowballStemmer("english")
#     return stemmer.stem(word)




# class LLaMA2Estimator:

#     def __init__(self, model_name_or_path, device):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=False).half().to(device)

#     def get_prompt(self, message, chat_history, system_prompt):
#         texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n[/INST]\n']
#         # The first user input is _not_ stripped
#         do_strip = False
#         for user_input, response in chat_history:
#             user_input = user_input.strip() if do_strip else user_input
#             do_strip = True
#             texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
#         message = message.strip() if do_strip else message
#         texts.append(f'{message}')
#         return ''.join(texts)

#     def estimate(self, context, n, temp, max_token):
#         system_prompt = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
#         prompt = self.get_prompt(context, [], system_prompt)
#         input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
#         generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
#         output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

#         cnt = {"OTHER": 0}
#         for item in output:
#             try:
#                 word_list = nltk.word_tokenize(item)
#                 first_word = [word for word in word_list if word.isalnum()][0]
#                 root_word = get_root_word(first_word)
#                 if root_word in cnt:
#                     cnt[root_word] += 1
#                 else:
#                     cnt[root_word] = 1
#             except:
#                 cnt["OTHER"] += 1
#         cnt
#         return cnt


# estimator = LLaMA2Estimator(model_name_or_path='../Llama-2-13b-chat-hf', device=torch.device('cuda:0'))

# roberta_tknz = RobertaTokenizer.from_pretrained("roberta-base")

# label2id = {
#     'human': 0,
#     'gpt2-xl': 1,
#     'gpt-j-6b': 2,
#     'Llama-2-13b-chat-hf': 3,
#     'vicuna-13b-v1.5': 4,
#     'alpaca-7b': 5,
#     'gpt-3.5-turbo': 6,
#     'gpt-4-1106-preview': 7
# }



# class LLaMA2Estimator:
#     def __init__(self, model_name_or_path, device):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

#     def get_next_word_prob(self, prefix, target_word):
#         # Tokenize the input prefix
#         input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)

#         # Get model output without gradient calculation
#         with torch.no_grad():
#             outputs = self.model(input_ids)
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Convert logits to probabilities
#             next_token_probs = torch.softmax(next_token_logits, dim=-1)

#         # Get the token ID of the target word
#         target_word_id = self.tokenizer.convert_tokens_to_ids(target_word)

#         # Get the probability of the target word
#         target_word_prob = next_token_probs[target_word_id].item()

#         return target_word_prob

# # Define the model path and device
# model_name_or_path = '../Llama-2-13b-chat-hf'  # Replace with your actual path
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Instantiate the estimator
# estimator = LLaMA2Estimator(model_name_or_path, device)

# # Define the prefix and target word
# prefix = "What's the weather"
# target_word = "like"

# # Get the probability of the target word being the next word
# probability = estimator.get_next_word_prob(prefix, target_word)
# print(f"The probability of the next word being '{target_word}' is {probability}")


# class LLaMA2Estimator:
#     def __init__(self, model_name_or_path, device):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

#     def get_next_word_probabilities(self, prefix, top_k=10):
#         # Tokenize the input prefix
#         input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)

#         # Get model output without gradient calculation
#         with torch.no_grad():
#             outputs = self.model(input_ids)
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Convert logits to probabilities
#             next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
#         # Get top k probabilities and their corresponding token ids
#         top_k_probs, top_k_ids = torch.topk(next_token_probs, top_k)
        
#         # Convert token ids to words
#         top_k_words = [self.tokenizer.decode([token_id]) for token_id in top_k_ids]
        
#         # Return the top k words and their probabilities
#         return list(zip(top_k_words, top_k_probs.tolist()))

# # Define the model path and device
# model_name_or_path = '../Llama-2-13b-chat-hf'  # Replace with your actual path
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Instantiate the estimator
# estimator = LLaMA2Estimator(model_name_or_path, device)

# # Define the prefix
# prefix = "What's the weather"

# # Get the top 10 probabilities for the next word
# top_k_probabilities = estimator.get_next_word_probabilities(prefix, top_k=10)
# for word, prob in top_k_probabilities:
#     print(f"Word: {word}, Probability: {prob}")









# # This code can calculate the probability of specific word given contex


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# class LLaMA2Estimator:
#     def __init__(self, model_name_or_path, device):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

#     def get_next_word_prob(self, prefix, target_word):
#         # Tokenize the input prefix
#         input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)

#         # Get model output without gradient calculation
#         with torch.no_grad():
#             outputs = self.model(input_ids)
#             next_token_logits = outputs.logits[0, -1, :]
            
#             # Convert logits to probabilities
#             next_token_probs = torch.softmax(next_token_logits, dim=-1)

#         # Get the token IDs of the target word
#         target_word_ids = self.tokenizer.encode(target_word, add_special_tokens=False)

#         # Calculate the probability of the target word
#         target_word_prob = 1.0
#         for target_word_id in target_word_ids:
#             target_word_prob *= next_token_probs[target_word_id].item()
#             input_ids = torch.cat([input_ids, torch.tensor([[target_word_id]], device=self.device)], dim=1)
#             with torch.no_grad():
#                 outputs = self.model(input_ids)
#                 next_token_logits = outputs.logits[0, -1, :]
#                 next_token_probs = torch.softmax(next_token_logits, dim=-1)

#         return target_word_prob

# # Define the model path and device
# model_name_or_path = '../Llama-2-13b-chat-hf'  # Replace with your actual path
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Instantiate the estimator
# estimator = LLaMA2Estimator(model_name_or_path, device)

# # Define the prefix and target word
# prefix = "What's the weather"
# target_word = "like"

# # Get the probability of the target word being the next word
# probability = estimator.get_next_word_prob(prefix, target_word)
# print(f"The probability of the next word being '{target_word}' is {probability}")

# # Define function to get root word
# def get_root_word(word):
#     from nltk.stem import SnowballStemmer
#     stemmer = SnowballStemmer("english")
#     return stemmer.stem(word)

# # Define the LLaMA2Estimator class
# class LLaMA2Estimator:
#     def __init__(self, model_name_or_path, device):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=False).half().to(device)

#     def get_prompt(self, message, chat_history, system_prompt):
#         texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n[/INST]\n']
#         do_strip = False
#         for user_input, response in chat_history:
#             user_input = user_input.strip() if do_strip else user_input
#             do_strip = True
#             texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
#         message = message.strip() if do_strip else message
#         texts.append(f'{message}')
#         return ''.join(texts)

#     def estimate(self, context, target_word):
#         system_prompt = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
#         prompt = self.get_prompt(context, [], system_prompt)
#         input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

#         # Get model output logits
#         with torch.no_grad():
#             outputs = self.model(input_ids, labels=input_ids)
#         logits = outputs.logits

#         # Extract the logits for the last token
#         logits = logits[:, -1, :]

#         # Compute probabilities using softmax
#         probs = softmax(logits, dim=-1).cpu().numpy()

#         # Get the ID for the target word
#         target_id = self.tokenizer.convert_tokens_to_ids(target_word)
#         prob = probs[0, target_id] if target_id in self.tokenizer.get_vocab().values() else 0

#         return -math.log(prob) if prob > 0 else float('inf')  # Use `float('inf')` for zero probability

# # Initialize the estimator
# estimator = LLaMA2Estimator('../Llama-2-13b-chat-hf', device=torch.device(0))

# # Initialize tokenizer
# roberta_tknz = RobertaTokenizer.from_pretrained("roberta-base")

# # Define label to ID mapping
# label2id = {
#     'human': 0,
#     'gpt2-xl': 1,
#     'gpt-j-6b': 2,
#     'Llama-2-13b-chat-hf': 3,
#     'vicuna-13b-v1.5': 4,
#     'alpaca-7b': 5,
#     'gpt-3.5-turbo': 6,
#     'gpt-4-1106-preview': 7
# }

# # Define argument parser
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n', type=int, default=100)
#     parser.add_argument('--delta', type=float, default=1.2)
#     parser.add_argument('--k', type=int, default=10)
#     parser.add_argument('--input', type=str)
#     parser.add_argument('--output', type=str)
#     return parser.parse_args()

# # Define function to get estimated probability
# def get_estimate_prob(context, target_word):
#     return estimator.estimate(context, target_word)

# # Define function to get word list from text
# def get_word_list(text):
#     word_list = nltk.word_tokenize(text)

#     for i, word in enumerate(word_list):
#         if len(word) < 2 and word not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
#             word_list[i] = "[SKIP]"
#         if word in ["``", "''"]:
#             word_list[i] = "[SKIP]"

#     pos = []
#     word_start = 0
#     for word in word_list:
#         if word == "[SKIP]":
#             pos.append(word_start)
#             continue

#         while text[word_start] != word[0]:
#             word_start += 1
#         pos.append(word_start)
#         word_start += len(word)

#     return word_list, pos

# # Define function to calculate maximum log probability
# def calc_max_logprob(n, delta):
#     min_p = 1 / (1 + n / (1.96/delta)**2)
#     max_logp = -math.log(min_p)
#     return max_logp

# # Define main function
# def main(args):
#     max_loss = calc_max_logprob(args.n, args.delta)
#     enc = tiktoken.get_encoding("cl100k_base")

#     with open(args.input, 'r') as f:
#         data = [json.loads(line) for line in f.readlines()]

#     for item in tqdm(data):
#         item['label_int'] = label2id[item['label']]
#         word_list, pos = get_word_list(item['text'])
#         proxy_ll = list(zip(range(len(word_list)), item['proxy_prob'], word_list, pos))

#         skip_prefix = 20
#         proxy_ll = proxy_ll[skip_prefix:]
#         proxy_ll = [item for item in proxy_ll if item[2] != "[SKIP]"]

#         # Sort by position
#         proxy_ll = sorted(proxy_ll, key=lambda x: x[0])
        
#         # Estimate probabilities
#         est_prob = []
#         for proxy_ll_item in proxy_ll:
#             context = item['text'][:proxy_ll_item[3]]
#             context = enc.decode(enc.encode(context)[-20:])
#             est_prob.append(get_estimate_prob(context, proxy_ll_item[2]))
        
#         est_prob = np.array(est_prob)
#         est_prob = est_prob.T
#         item['est_prob_list'] = est_prob.tolist()
        
#         prefix_text_list = [item['text'][:proxy_ll_item[3]] for proxy_ll_item in proxy_ll]
#         target_roberta_idx_list = [len(ids)-2 for ids in roberta_tknz(prefix_text_list).input_ids]
#         item['target_roberta_idx'] = target_roberta_idx_list

#         item['target_prob_idx'] = [proxy_ll_item[3] for proxy_ll_item in proxy_ll]
#         del item['proxy_prob']
#         del item['source']

#         with open(args.output, 'a') as f:
#             f.write(json.dumps(item) + '\n')

# # Run the main function if this script is executed
# if __name__ == '__main__':
#     args = parse_args()
#     main(args)






import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class LLaMA2Estimator:
    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    def get_word_probabilities(self, text):
        # Tokenize the input text
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # Get model outputs without gradient calculation
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # Get logits for the sequence

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Get tokens and their probabilities
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_probs = probabilities[torch.arange(len(tokens)), torch.arange(len(tokens))]

        return tokens, token_probs

    def get_probabilities_from_word(self, text, start_word_index):
        tokens, token_probs = self.get_word_probabilities(text)

        # We need to find the token for the start word index
        start_word_token = tokens[start_word_index]

        # Get the probabilities for all tokens from the start_word_index
        probabilities = token_probs[start_word_index + 1:]

        return tokens[start_word_index + 1:], probabilities

# Define the model path and device
model_name_or_path = '../Llama-2-13b-chat-hf'  # Replace with your actual model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the estimator
estimator = LLaMA2Estimator(model_name_or_path, device)

# Define the text
text = "Yes By leaving them behind. It's not about standing up to them. Unless you are being physically threatened as your life is on the line. If they are a sociopath or psychopath you standing up against them verbally will put you further into their web of abuse. I can guarantee they will use this against you. They absolutely love it when this happens. This is the only love they can feel. Which is your destruction. DON'T DO IT In some cases your life will be put into danger and they will stalk and harass you for YEARS after you leave. If you have children or large assets get the best lawyer possible and only communicate through them. Standing up to a low level narcissist is a different story. They are extremely WEAK and confronting them will sometimes work. But, my guidance is that you are playing with your life if you do this. Time to start not thinking anything about this person and put all of your energy into yourself. Drop everything about them from your life. What you win is your life back"

# Get the probabilities for words from the 20th index
tokens_after_20, probs_after_20 = estimator.get_probabilities_from_word(text, start_word_index=20)

# Print the results
for token, prob in zip(tokens_after_20, probs_after_20):
    print(f"Word: {token}, Probability: {prob.item()}")

