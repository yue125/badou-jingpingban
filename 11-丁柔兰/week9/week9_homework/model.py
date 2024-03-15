# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
"""
建立网络模型结构
"""

from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classifier = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.use_crf:
                # 创建一个掩码，对于非忽略索引 (-100) 的位置为 True，否则为 False
                mask = labels.ne(-100)
                # 计算 CRF 损失，确保 labels 中没有忽略索引传递给 CRF 层
                loss = -self.crf_layer(logits, labels.where(mask, torch.tensor(0).to(labels.device)), mask=mask,
                                       reduction="mean")
                return loss
            else:
                return self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(logits)
            else:
                return logits



def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)