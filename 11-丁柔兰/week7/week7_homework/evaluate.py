# -*- coding: utf-8 -*-
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
        # # 记录模型预测开始时间
        # start_time = time.time()
        # # 记录模型预测结束时间
        # end_time = time.time()
        # # 计算模型预测时间
        # prediction_time_lr = end_time - start_time
        # # 计算性能指标
        # # 准确率（Accuracy）：accuracy_score 函数计算的是模型正确预测的样本数与总样本数的比例。它是最直观的性能衡量指标。
        # accuracy_lr = accuracy_score(true_label, pred_label)
        # # 精确率（Precision）：precision_score 函数计算的是模型正确预测为正类的样本数与模型预测为正类的总样本数的比例。它衡量了模型预测正类别时的准确性。
        # precision_lr = precision_score(true_label, pred_label)
        # # 召回率（Recall）：recall_score 函数计算的是模型正确预测为正类的样本数与真实为正类的样本数的比例。也称为真正例率，它衡量了模型捕捉到的正类别样本的比例。
        # recall_lr = recall_score(true_label, pred_label)
        # # F1 分数（F1 Score）：f1_score 函数计算的是精确率和召回率的调和平均数。它是精确率和召回率的综合指标，特别适用于类别不平衡的情况。
        # f1_lr = f1_score(true_label, pred_label)
        # # 创建一个空的 DataFrame
        # results_df = pd.DataFrame(
        #     columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Prediction Time'])
        # # 添加逻辑回归模型的性能数据
        # results_df = results_df.append({
        #     'Model': 'Logistic Regression',
        #     'Accuracy': accuracy_lr,
        #     'Precision': precision_lr,
        #     'Recall': recall_lr,
        #     'F1 Score': f1_lr,
        #     'Prediction Time': prediction_time_lr
        # }, ignore_index=True)
        # print(results_df)
        return correct / (correct + wrong)
