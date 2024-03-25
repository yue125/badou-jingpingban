# -*- coding: utf-8 -*-
import os
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        # 创建一个包含单行数据的字典
        data = {
            'Model': 'Logistic Regression',
            '预测正确条目': correct,
            '预测错误条目': wrong,
            '预测准确率': correct / (correct + wrong),
            '预测集合条目总量': correct + wrong
        }
        # 检查 CSV 文件是否存在
        csv_file_path = '../data/results.csv'
        if os.path.exists(csv_file_path):
            # 如果文件存在，读取现有数据
            # results_df = pd.read_csv(csv_file_path)
            # 追加新数据
            results_df = pd.read_csv(csv_file_path).append(data, ignore_index=True)
        else:
            # 如果文件不存在，创建一个新的 DataFrame
            results_df = pd.DataFrame([data])
        # 将 DataFrame 写入 CSV 文件
        results_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        return correct / (correct + wrong)
