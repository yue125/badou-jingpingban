# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.data = []  # 先初始化 self.data 为空列表
        self.load_data()  # 然后调用 self.load_data() 方法

    def load_data(self):
        with open(self.path, "r", encoding="utf8") as f:
            segments = f.read().strip().split("\n\n")
            self.sentences = []  # 添加这行代码来初始化句子列表
            for segment in segments:
                sentence, labels = [], []
                for line in segment.split("\n"):
                    if line:
                        char, label = line.split()
                        sentence.append(char)
                        labels.append(self.schema[label])
                self.sentences.append(''.join(sentence))  # 保存句子
                encoded_inputs = self.tokenizer(sentence,
                                                is_split_into_words=True,
                                                max_length=self.config["max_length"],
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors="pt")
                input_ids = encoded_inputs["input_ids"].squeeze()
                attention_mask = encoded_inputs["attention_mask"].squeeze()
                token_type_ids = encoded_inputs["token_type_ids"].squeeze()
                label_ids = self.encode_labels(labels)
                self.data.append((input_ids, attention_mask, token_type_ids, torch.LongTensor(label_ids)))

    def encode_labels(self, labels):
        # 保证至少有一个有效的标签
        label_ids = [self.schema.get(labels[0], 0)]  # 取第一个有效的标签而不是-100
        for label in labels[1:]:
            label_ids.append(self.schema.get(label, -100))  # -100 是忽略索引
        label_ids = label_ids[:self.config["max_length"]]
        padding_length = max(self.config["max_length"] - len(label_ids), 0)
        label_ids.extend([-100] * padding_length)  # 用忽略索引进行填充
        return label_ids

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

