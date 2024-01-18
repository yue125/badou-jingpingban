import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        """

        :param vector_dim: 向量维度
        :param sentence_length: 句子长度
        :param vocab: 词汇表
        """
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = F.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_length) --> (batch_size, sen_length, vector_dim)
        x = x.transpose(1, 2)  # (batch_size, sen_length, vector_dim) --> (batch_size, vector_dim, sen_length)
        x = self.pool(x)  # (batch_size, vector_dim, sen_length) --> (batch_size, vector_dim, 1)
        x = x.squeeze()  # (batch_size, vector_dim, 1) --> (batch_size, vector_dim)
        x = self.classify(x)  # (batch_size, vector_dim) --> (batch_size, 1) 3*5 5*1 --> 3*1
        y_pred = self.activation(x)  # (batch_size, 1) --> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失

        return y_pred  # 输出预测结果


# 生成词汇表
# {"a":1, "b":2, "c":3...}
# abc --> [1, 2, 3]
def build_vocab():
    """
    生成词汇表
    :return: vocab
    """
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 27 当句子中出现不存在于字符集的字符时用 unk 表示

    return vocab


# 随机生成一个样本
# 从所有字符中选取 sentence_length 个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从词汇表中选取 sentence_length 个字，可能重复
    x = [np.random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定那些字出现时为正样本
    if set("abc") & set(x):
        y = 1
    # 指定字都未出现，则为负样本
    else:
        y = 0

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做 embedding

    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    """
    建立数据集
    :param sample_length: 样本数量
    :param vocab: 词汇表
    :param sentence_length: 句子长度
    :return: x数据集，y数据集
    """
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])

    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def buidl_model(vocab, char_dim, sentence_length):
    """
    建立模型
    :param vocab: 词汇表
    :param char_dim: 字的维度
    :param sentence_length:句子的长度
    :return: model
    """
    model = TorchModel(char_dim, sentence_length, vocab)

    return model


# 测试代码
# 用来测试每轮模型的准确率
def evalute(model, vocab, sentence_length):
    """
    测试函数
    :param model: 模型
    :param vocab: 词汇表
    :param sentence_length:句子长度
    :return: 正确率
    """
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立测试样本
    print(f"本次预测集中共有{int(sum(y))}个正样本，{int(200 - sum(y))}个负样本")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) > 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1

    print(f"正确预测个数：{correct}，正确率：{correct / (correct + wrong)}")

    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总训练样本数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = buidl_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构建一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=" * 20, f"第{epoch + 1}轮平均loss：{np.mean(watch_loss)}")

        acc = evalute(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.xlabel("epoch")

    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "../model/model.pth")
    # 保存词表
    writer = open("../model/vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))  # indent=2 缩进为2个空格
    writer.close()

    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))  # 加载字符集
    model = buidl_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测

    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}，预测类别：{round(float(result[i]))}，概率值：{float(result[i])}")  # 打印预测结果


if __name__ == '__main__':
    main()
    test_strings = ["fnrnnf", "epamfm", "reqnfs", "nadreb"]
    predict("../model/model.pth", "../model/vocab.json", test_strings)