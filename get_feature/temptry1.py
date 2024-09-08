# import json
# import re
# import nltk

# # 文件路径
# file_path = '../data/train.jsonl'

# # 读取 JSONL 文件
# with open(file_path, 'r', encoding='utf-8') as f:
#     # 读取第一行数据
#     first_line = f.readline().strip()
#     # 解析为 JSON 对象
#     first_data = json.loads(first_line)

# # 获取 text 和 proxy_prob 字段
# text = first_data['text']
# proxy_prob = first_data['proxy_prob']

# # 使用正则表达式匹配所有非空格字符
# text_tokens = re.findall(r'\S', text)
# text_length = len(text_tokens)

# # 计算 proxy_prob 的长度
# proxy_prob_length = len(proxy_prob)

# # 打印长度
# print(text_tokens)
# print(f"Text length: {text_length}")
# print(proxy_prob)
# print(f"Proxy_prob length: {proxy_prob_length}")


# import json
# from transformers import RobertaTokenizer

# # 示例数据
# item = {
#     "text": "Hello world! This is a test.",
#     "proxy_prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# }
# proxy_ll = [
#     (0, 0.1, 'Hello', 0),
#     (1, 0.2, 'world', 6),
#     (2, 0.3, '!', 11),
#     (3, 0.4, 'This', 13),
#     (4, 0.5, 'is', 18),
#     (5, 0.6, 'a', 21),
#     (6, 0.7, 'test', 23),
#     (7, 0.8, '.', 28)
# ]

# # 使用 RoBERTa tokenizer
# roberta_tknz = RobertaTokenizer.from_pretrained("roberta-base")

# # 生成 prefix_text_list
# prefix_text_list = [item['text'][:proxy_ll_item[3]] for proxy_ll_item in proxy_ll]

# # 计算 target_roberta_idx_list
# encoded_inputs = roberta_tknz(prefix_text_list, padding=True, return_tensors='pt')
# target_roberta_idx_list = [len(ids) - 2 for ids in encoded_inputs['input_ids']]

# # 存储在 item 字典中
# item['target_roberta_idx'] = target_roberta_idx_list

# # 打印结果
# print("Prefix text list:", prefix_text_list)
# print("Target RoBERTa idx list:", target_roberta_idx_list)


# est_prob_list=[1.7719568419318752, 4.605170185988091, 3.912023005428146, 3.506557897319982, 0.6733445532637656, 4.605170185988091, 1.1086626245216111, 0.6539264674066639, 2.4079456086518722, 4.605170185988091, 0.9675840262617056, 4.605170185988091, 4.605170185988091, 2.8134107167600364, 4.605170185988091, 2.5257286443082556, 4.605170185988091, 4.605170185988091, 4.605170185988091, 0.9942522733438669, 4.605170185988091, 1.4696759700589417, 4.605170185988091, 0.4307829160924542, 4.605170185988091, 3.506557897319982, 2.4079456086518722, 4.605170185988091, 1.1086626245216111, 3.506557897319982, 4.605170185988091, 4.605170185988091, 4.605170185988091, 2.995732273553991, 0.8915981192837836, 2.120263536200091, 3.912023005428146, 3.2188758248682006, 0.08338160893905101, 2.5257286443082556, 4.605170185988091, 3.506557897319982, 4.605170185988091, -0.0, 4.605170185988091, 1.0788096613719298, 4.605170185988091, 1.3093333199837622, 0.8675005677047231, 3.912023005428146, 4.605170185988091, 0.527632742082372, 4.605170185988091, 2.995732273553991, 4.605170185988091, 0.9675840262617056, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 0.5978370007556204, 4.605170185988091, 4.605170185988091, 2.2072749131897207, 3.506557897319982, 4.605170185988091, 2.5257286443082556, 1.5141277326297755, 4.605170185988091, 4.605170185988091, 1.1394342831883648, 1.6607312068216509, 4.605170185988091, 1.1086626245216111, 4.605170185988091, 2.659260036932778, 2.0402208285265546, 0.5447271754416722, 4.605170185988091, 2.4079456086518722, 4.605170185988091, 1.3862943611198906, 2.659260036932778, 3.506557897319982, 0.6733445532637656, 3.2188758248682006, 2.120263536200091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 3.506557897319982, 2.659260036932778, 4.605170185988091, 1.8325814637483102, 4.605170185988091, 3.912023005428146, 0.6931471805599453, 1.5141277326297755, 4.605170185988091, 4.605170185988091, 1.1394342831883648, 2.8134107167600364, 4.605170185988091, 2.8134107167600364, 4.605170185988091, 2.995732273553991, 3.2188758248682006, 4.605170185988091, 2.2072749131897207, 3.506557897319982, 3.506557897319982, 4.605170185988091, 4.605170185988091, 1.171182981502945, 4.605170185988091, 1.5606477482646683, 2.659260036932778, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 1.3862943611198906, 3.912023005428146, 1.7147984280919266, 4.605170185988091, 4.605170185988091, 2.3025850929940455, 4.605170185988091, 1.6607312068216509, 1.9661128563728327, 0.06187540371808753, 2.4079456086518722, 2.995732273553991, 4.605170185988091, 4.605170185988091, 4.605170185988091, 4.605170185988091, 3.2188758248682006, 4.605170185988091, 2.4079456086518722, 3.506557897319982, 4.605170185988091, 1.0498221244986778, 4.605170185988091, 3.506557897319982, 3.912023005428146, 0.07257069283483537, 1.6094379124341003, 0.6348782724359695, 4.605170185988091, 4.605170185988091, 3.912023005428146, 3.506557897319982, 0.9675840262617056, 0.916290731874155, 0.020202707317519466, 2.120263536200091, 4.605170185988091, 2.120263536200091, 4.605170185988091, 2.3025850929940455, 3.506557897319982, 3.912023005428146, 2.995732273553991] 
# target_roberta_idx=[21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 86, 86, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 201, 202, 203, 204, 205, 206, 207] 
# target_prob_idx=[96, 107, 110, 115, 120, 123, 126, 130, 136, 139, 144, 148, 150, 160, 163, 174, 178, 187, 190, 198, 203, 212, 217, 221, 225, 233, 238, 244, 248, 251, 258, 260, 264, 274, 279, 284, 288, 293, 301, 306, 311, 322, 327, 330, 335, 340, 349, 354, 357, 361, 366, 371, 376, 380, 386, 392, 395, 400, 413, 415, 419, 422, 425, 428, 433, 439, 444, 449, 454, 457, 461, 466, 473, 477, 482, 487, 493, 497, 504, 508, 512, 518, 524, 528, 535, 538, 542, 547, 556, 559, 565, 572, 576, 580, 585, 592, 601, 605, 610, 622, 630, 636, 645, 648, 651, 653, 657, 663, 674, 677, 679, 689, 696, 701, 705, 715, 720, 724, 736, 741, 746, 756, 762, 767, 770, 779, 782, 787, 791, 795, 803, 808, 813, 818, 821, 825, 828, 834, 839, 842, 848, 852, 861, 870, 876, 881, 888, 892, 896, 900, 903, 908, 915, 920, 930, 935, 946, 952, 957, 962, 967, 973, 978, 982, 986, 989, 994, 999]
# train_prob=[0.0, 9.564894676208496, 9.386779648917061, 7.009927272796631, 4.1703667640686035, 3.6779727935791016, 3.6779727935791016, 3.606122136116028, 3.9350740909576416, 1.762415361404419, 3.8492648899555206, 7.412138938903809, 7.412138938903809, 7.412138938903809, 1.864798665046692, 1.8785201907157898, 1.8720585107803345, 3.978859066963196, 8.951211929321289, 2.6674694299697874, 4.0936994552612305, 3.5280542373657227, 3.5280542373657227, 2.6547160148620605, 2.6547160148620605, 2.6547160148620605, 2.6547160148620605, 6.145983457565308, 5.942564010620117, 5.942564010620117, 3.483971118927002, 1.890977144241333, 0.3440511226654053, 0.9479171788682126, 0.003031303873285651, 1.9432489495258778, 1.6884050369262695, 4.467850983142853, 6.418478012084961, 1.8647500211372972, 1.002744436264038, 1.2617118060588837, 5.2237478494644165, 11.671541213989258, 11.671541213989258, 4.365618259779045, 3.4673385620117188, 1.4277923569083213, 2.913009747862816, 8.602839469909668, 8.602839469909668, 8.602839469909668, 1.8697948455810547, 1.8697948455810547, 3.0465097427368164, 4.341111660003662, 4.341111660003662, 3.2586326599121094, 2.243083953857422, 2.3144667829786028, 1.7479804754257202, 0.7401162981987, 5.253793716430664, 3.522689151763916, 3.5712645053863525, 3.5712645053863525, 3.525444805622101, 3.3879857063293457, 1.1526090162140983, 2.8467371463775635, 2.9582177996635437, 2.995378017425537, 1.1596238613128662, 1.1596238613128662, 0.3122854274697602, 1.442670316901058, 2.360045909881592, 7.701056003570557, 7.701056003570557, 7.701056003570557, 5.132345676422119, 2.5636353492736816, 2.8229568654840644, 5.98236608505249, 3.10128715634346, 0.22020822763442993, 0.22020822763442993, 0.22020822763442993, 2.299648880958557, 4.138236999511719, 1.3635878801345824, 2.6617212295532227, 4.6527756452560425, 10.165550231933594, 0.3836600184440613, 0.3836600184440613, 3.3723349571228027, 1.4297828674316406, 7.300997734069824, 5.605512648820877, 4.0947375893592834, 7.121445083618164, 6.297986030578613, 6.297986030578613, 5.532270054022471, 9.82837200164795, 1.4608311999007129, 3.388227644562721, 5.6409257253011065, 2.9001397132873534, 0.72243332862854, 0.72243332862854, 3.029548724492391, 4.012454092502594, 2.6814850568771362, 1.1092568784952164, 1.6662479043006897, 2.429927388827006, 1.6794567108154297, 1.6794567108154297, 2.6793874502182007, 2.7795588970184326, 2.087624553591013, 4.9832763671875, 2.500723123550415, 0.8496814356608824, 4.430686133248465, 4.306017755530775, 4.226325511932373, 2.7681447863578796, 0.6507620811462402, 0.6507620811462402, 0.6507620811462402, 1.4204298655192058, 1.4659645915031434, 1.930378371477127, 1.4559454917907715, 1.4559454917907715, 4.898074997795953, 6.463319778442383, 5.32328987121582, 5.849815607070923, 7.4293928146362305, 3.0847070746951633, 1.213552713394165, 1.213552713394165, 2.7772432457317007, 4.463581919670105, 6.96809184551239, 6.295356273651123, 4.2457664012908936, 2.196176528930664, 2.196176528930664, 2.196176528930664, 2.1304097175598145, 4.918900340795517, 10.28073501586914, 10.28073501586914, 0.049187857657670975, 0.6180639266967773, 7.251639332090106, 3.333008050918579, 3.907975137233734, 4.099630832672119, 2.0509838508442044, 0.3780734601120154, 1.1295466423034668, 4.213895082473755, 6.3962860107421875, 6.3962860107421875, 6.3962860107421875, 2.1519563674926756, 0.8668946921825409, 1.9639082103967667, 4.200367406010628, 6.9793318748474125, 3.0620814710855484, 7.237410187721252, 13.667749404907227, 13.667749404907227, 13.667749404907227, 0.21847109496593475, 0.630000326782465, 3.6473897298177085, 7.212993144989014, 6.151360750198364, 5.089728355407715, 1.8486323207616806, 7.525181674957276, 8.001488769054413, 1.9451169669628143, 2.805250257253647, 1.5543477237224579, 3.4373276829719543, 6.0218987464904785, 6.0218987464904785, 4.312111854553223, 0.8925380706787109, 2.479269027709961, 4.065999984741211, 0.20990687608718872, 3.3389284163713455]
# # {"text": "Yes By leaving them behind. It's not about standing up to them. Unless you are being physically threatened as your life is on the line. If they are a sociopath or psychopath you standing up against them verbally will put you further into their web of abuse. I can guarantee they will use this against you. They absolutely love it when this happens. This is the only love they can feel. Which is your destruction. DON'T DO IT In some cases your life will be put into danger and they will stalk and harass you for YEARS after you leave. If you have children or large assets get the best lawyer possible and only communicate through them. Standing up to a low level narcissist is a different story. They are extremely WEAK and confronting them will sometimes work. But, my guidance is that you are playing with your life if you do this. Time to start not thinking anything about this person and put all of your energy into yourself. Drop everything about them from your life. What you win is your life back", "label": "human", "source": "Quora-SAID", "proxy_prob": [0.0, 9.564894676208496, 9.386779648917061, 7.009927272796631, 4.1703667640686035, 3.6779727935791016, 3.6779727935791016, 3.606122136116028, 3.9350740909576416, 1.762415361404419, 3.8492648899555206, 7.412138938903809, 7.412138938903809, 7.412138938903809, 1.864798665046692, 1.8785201907157898, 1.8720585107803345, 3.978859066963196, 8.951211929321289, 2.6674694299697874, 4.0936994552612305, 3.5280542373657227, 3.5280542373657227, 2.6547160148620605, 2.6547160148620605, 2.6547160148620605, 2.6547160148620605, 6.145983457565308, 5.942564010620117, 5.942564010620117, 3.483971118927002, 1.890977144241333, 0.3440511226654053, 0.9479171788682126, 0.003031303873285651, 1.9432489495258778, 1.6884050369262695, 4.467850983142853, 6.418478012084961, 1.8647500211372972, 1.002744436264038, 1.2617118060588837, 5.2237478494644165, 11.671541213989258, 11.671541213989258, 4.365618259779045, 3.4673385620117188, 1.4277923569083213, 2.913009747862816, 8.602839469909668, 8.602839469909668, 8.602839469909668, 1.8697948455810547, 1.8697948455810547, 3.0465097427368164, 4.341111660003662, 4.341111660003662, 3.2586326599121094, 2.243083953857422, 2.3144667829786028, 1.7479804754257202, 0.7401162981987, 5.253793716430664, 3.522689151763916, 3.5712645053863525, 3.5712645053863525, 3.525444805622101, 3.3879857063293457, 1.1526090162140983, 2.8467371463775635, 2.9582177996635437, 2.995378017425537, 1.1596238613128662, 1.1596238613128662, 0.3122854274697602, 1.442670316901058, 2.360045909881592, 7.701056003570557, 7.701056003570557, 7.701056003570557, 5.132345676422119, 2.5636353492736816, 2.8229568654840644, 5.98236608505249, 3.10128715634346, 0.22020822763442993, 0.22020822763442993, 0.22020822763442993, 2.299648880958557, 4.138236999511719, 1.3635878801345824, 2.6617212295532227, 4.6527756452560425, 10.165550231933594, 0.3836600184440613, 0.3836600184440613, 3.3723349571228027, 1.4297828674316406, 7.300997734069824, 5.605512648820877, 4.0947375893592834, 7.121445083618164, 6.297986030578613, 6.297986030578613, 5.532270054022471, 9.82837200164795, 1.4608311999007129, 3.388227644562721, 5.6409257253011065, 2.9001397132873534, 0.72243332862854, 0.72243332862854, 3.029548724492391, 4.012454092502594, 2.6814850568771362, 1.1092568784952164, 1.6662479043006897, 2.429927388827006, 1.6794567108154297, 1.6794567108154297, 2.6793874502182007, 2.7795588970184326, 2.087624553591013, 4.9832763671875, 2.500723123550415, 0.8496814356608824, 4.430686133248465, 4.306017755530775, 4.226325511932373, 2.7681447863578796, 0.6507620811462402, 0.6507620811462402, 0.6507620811462402, 1.4204298655192058, 1.4659645915031434, 1.930378371477127, 1.4559454917907715, 1.4559454917907715, 4.898074997795953, 6.463319778442383, 5.32328987121582, 5.849815607070923, 7.4293928146362305, 3.0847070746951633, 1.213552713394165, 1.213552713394165, 2.7772432457317007, 4.463581919670105, 6.96809184551239, 6.295356273651123, 4.2457664012908936, 2.196176528930664, 2.196176528930664, 2.196176528930664, 2.1304097175598145, 4.918900340795517, 10.28073501586914, 10.28073501586914, 0.049187857657670975, 0.6180639266967773, 7.251639332090106, 3.333008050918579, 3.907975137233734, 4.099630832672119, 2.0509838508442044, 0.3780734601120154, 1.1295466423034668, 4.213895082473755, 6.3962860107421875, 6.3962860107421875, 6.3962860107421875, 2.1519563674926756, 0.8668946921825409, 1.9639082103967667, 4.200367406010628, 6.9793318748474125, 3.0620814710855484, 7.237410187721252, 13.667749404907227, 13.667749404907227, 13.667749404907227, 0.21847109496593475, 0.630000326782465, 3.6473897298177085, 7.212993144989014, 6.151360750198364, 5.089728355407715, 1.8486323207616806, 7.525181674957276, 8.001488769054413, 1.9451169669628143, 2.805250257253647, 1.5543477237224579, 3.4373276829719543, 6.0218987464904785, 6.0218987464904785, 4.312111854553223, 0.8925380706787109, 2.479269027709961, 4.065999984741211, 0.20990687608718872, 3.3389284163713455]}


# print(len(est_prob_list))
# print(len(target_roberta_idx))
# print(len(target_prob_idx))
# print(len(train_prob))

# world_list=nltk.word_tokenize("Yes By leaving them behind. It's not about standing up to them. Unless you are being physically threatened as your life is on the line. If they are a sociopath or psychopath you standing up against them verbally will put you further into their web of abuse. I can guarantee they will use this against you. They absolutely love it when this happens. This is the only love they can feel. Which is your destruction. DON'T DO IT In some cases your life will be put into danger and they will stalk and harass you for YEARS after you leave. If you have children or large assets get the best lawyer possible and only communicate through them. Standing up to a low level narcissist is a different story. They are extremely WEAK and confronting them will sometimes work. But, my guidance is that you are playing with your life if you do this. Time to start not thinking anything about this person and put all of your energy into yourself. Drop everything about them from your life. What you win is your life back")
# print(world_list)
# print(len(world_list))

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




def get_root_word(word):
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    return stemmer.stem(word)


class LLaMA2Estimator:
    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    def estimate(self, context):
        # Tokenize the context
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for the last token in the sequence
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities

        # Create a dictionary with token probabilities
        token_probs = {}
        for token_id in input_ids[0]:
            token_id = token_id.item()
            token = self.tokenizer.decode([token_id]).strip()
            if token and token not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']:  # Ignore special tokens
                token_probs[token] = probs[token_id].item()

        return token_probs

def get_estimate_prob(context, target_word):
    target_word = get_root_word(target_word)
    
    # Use the estimator instance to estimate probabilities
    token_probs = estimator.estimate(context)
    
    # Find the probability of the target word
    if target_word in token_probs:
        return -1 * math.log(token_probs[target_word])
    else:
        return -1 * math.log(1 / len(token_probs))  # Assuming uniform probability for unknown words

def get_word_list(text):
    word_list = nltk.word_tokenize(text)

    for i, word in enumerate(word_list):
        if len(word) < 2 and word not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            word_list[i] = "[SKIP]"
        if word in ["``", "''"]:
            word_list[i] = "[SKIP]"

    pos = []
    word_start = 0
    for word in word_list:
        if word == "[SKIP]":
            pos.append(word_start)
            continue

        while text[word_start] != word[0]:
            word_start += 1
        pos.append(word_start)
        word_start += len(word)

    return word_list, pos

def calc_max_logprob(n, delta):
    min_p = 1 / (1 + n / (1.96 / delta) ** 2)
    max_logp = -math.log(min_p)
    return max_logp

def main(args):
    max_loss = calc_max_logprob(args.n, args.delta)
    enc = tiktoken.get_encoding("cl100k_base")

    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    for item in tqdm(data):
        item['label_int'] = label2id[item['label']]
        word_list, pos = get_word_list(item['text'])
        proxy_ll = list(zip(range(len(word_list)), item['proxy_prob'], word_list, pos))

        skip_prefix = 20
        proxy_ll = proxy_ll[skip_prefix:]
        proxy_ll = [item for item in proxy_ll if item[2] != "[SKIP]"]

        proxy_ll = sorted(proxy_ll, key=lambda x: x[0])
        
        est_prob = []
        for proxy_ll_item in proxy_ll:
            context = item['text'][:proxy_ll_item[3]]
            context = enc.decode(enc.encode(context)[-20:])
            est_prob.append(get_estimate_prob(context, proxy_ll_item[2]))
        est_prob = np.array(est_prob)
        est_prob = est_prob.T
        item['est_prob_list'] = est_prob.tolist()
        
        prefix_text_list = [item['text'][:proxy_ll_item[3]] for proxy_ll_item in proxy_ll]
        target_roberta_idx_list = [len(ids) - 2 for ids in roberta_tknz(prefix_text_list).input_ids]
        item['target_roberta_idx'] = target_roberta_idx_list

        item['target_prob_idx'] = [proxy_ll_item[3] for proxy_ll_item in proxy_ll]
        del item['proxy_prob']
        del item['source']

        with open(args.output, 'a') as f:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))