# -*- coding: utf-8 -*-
# 设置文件编码，用于 Python 2.x 兼容性，确保文件中的字符串按 utf-8 编码处理。

import torch  # 导入 PyTorch 库
import re  # 导入正则表达式库，用于处理文本匹配和搜索
import numpy as np  # 导入 NumPy 库，用于科学计算
from collections import defaultdict  # 导入 defaultdict，它是一个字典类，支持快速初始化不存在的键
from loader import load_data  # 导入定义在其他文件中的 load_data 函数，用于加载数据

"""
模型效果测试
这段代码定义了一个评估器类 Evaluator，用于评估和测试命名实体识别（NER）模型的性能。
Evaluator 类通过加载验证数据集、执行预测、解码实体、统计正确识别的实体数，并计算准确率、召回率和 F1 分数。
"""


# 定义一个评估器类，用于评估模型性能
class Evaluator:
    # 初始化方法，接收配置对象、模型实例和日志记录器
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # 调用 load_data 函数加载验证数据集，设置 shuffle=False 表示不打乱数据顺序
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    # 定义一个评估模型性能的方法，接受当前的训练周期数作为参数
    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        # 初始化统计字典，用于记录不同实体类型的识别情况
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()  # 将模型设置为评估模式
        # 遍历验证数据集中的每个批次
        for index, batch_data in enumerate(self.valid_data):
            # 从数据集中获取句子数据
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            # 如果 CUDA 可用，则将数据转移到 GPU
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            # 解包批次数据，得到输入 ID、注意力掩码、令牌类型 ID 和标签
            input_ids, attention_mask, token_type_ids, labels = batch_data
            with torch.no_grad():  # 关闭梯度计算，因为我们不需要更新模型参数
                # 使用模型进行预测
                pred_results = self.model(input_ids, attention_mask, token_type_ids)
            # 调用 write_stats 方法更新统计字典
            self.write_stats(labels, pred_results, sentences)
        # 调用 show_stats 方法展示统计结果
        self.show_stats()
        return

    # 定义一个方法，用于更新统计字典的信息
    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)  # 确保标签、预测结果和句子数量相等
        # 如果不使用 CRF 层，则使用 argmax 获取预测类别
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        # 遍历每一个真实标签、预测标签和句子
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            # 如果不使用 CRF 层，则将预测标签转移到 CPU 并转换为列表
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            # 将真实标签转移到 CPU 并转换为列表
            true_label = true_label.cpu().detach().tolist()
            # 调用 decode 方法解码出实体信息
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)

            # 更新统计字典，计算正确识别的实体数、样本实体数和识别出的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    # 定义一个方法，用于展示统计结果
    def show_stats(self):
        F1_scores = []
        # 遍历每个实体类型，计算并打印准确率、召回率和 F1 分数
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        # 计算并打印宏平均 F1 分数
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        # 计算并打印微平均 F1 分数
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    # 定义一个解码方法，用于从标签序列中提取实体信息
    def decode(self, sentence, labels):
        # 将标签序列转换为字符串形式
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        # 使用正则表达式匹配不同类型的实体
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
