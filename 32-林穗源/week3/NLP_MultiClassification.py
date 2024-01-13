import json
import random
import string
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

"""
任务：多分类任务，输入英文字符串，输出字母a在字符串中出现的位置
关键词：词表, embedding, RNN, 多分类
"""

def logger_config(log_path,logging_name):
    """
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    """
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

logger = logger_config(log_path='train.log',logging_name='nlp-multi-classification')

class NlpMultiClassificationModule(nn.Module):
    def __init__(self, vocab, vector_dim, out_features):
        super(NlpMultiClassificationModule, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)  # rnn层
        self.classify = nn.Linear(vector_dim, out_features)  # 全连接层
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # embedding
        # rnn将含有多个字符(多行)的矩阵纵向压缩成一行, 并包含序列信息, 相当于降维
        output, x = self.rnn(x)  # 获取最后一个时间步的隐藏层, 包含了所有的序列信息, 包括字符的位置信息
        x = x.squeeze()  # 去掉降维后值为1的维度, 使其形状与真实label形同, 用于计算损失
        y_pred = self.classify(x)  # 多分类

        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


def build_vocab():
    """
    构建词表
    :return:
    """
    chars = string.ascii_lowercase  # 获取26个英文字母
    vocab = {'pad': 0}
    for i, char in enumerate(chars):
        vocab[char] = i + 1
    vocab['unk'] = len(vocab)
    return vocab

def padding(max_sequence, sequence):
    """
    用于词表转换后, 截断或者填充句子
    :param max_sequence: 最大截断数
    :param sequence: 句子序列
    :return:
    """
    if len(sequence) >= max_sequence:
        return sequence[:max_sequence]
    else:
        return sequence + [0] * (max_sequence - len(sequence))

def build_sample(vocab, max_sequence):
    global spe_char  # 要定位的字符
    """
    构造一个样本
    :param vocab: 词表
    :param max_sequence: 最大截断数
    :return:
    """
    # 随机从词表选取sentence_length个字
    x = random.sample(list(vocab.keys())[1:-1], random.randint(1, max_sequence))

    if spe_char in x:
        # 使用自然顺序，从1开始
        char_index = x.index(spe_char)
        y = char_index + 1
    else:
        # 当样本没有这个字符时，标签为最后一个类别
        y = 0

    x = [vocab.get(word, vocab['unk']) for word in x]  # 转换为词表序号，为了做embedding
    x = padding(max_sequence, x)
    return x, y

def build_dataset(sample_num, vocab, max_sequence):
    """
    构造数据集
    :param sample_num: 样本数量
    :param vocab: 词表
    :param max_sequence: 最大截断数
    :return:
    """
    dataset_x = []
    dataset_y = []
    for i in range(sample_num):
        x, y = build_sample(vocab, max_sequence)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, max_sequence):
    """
    评估
    :param model: 模型
    :param vocab: 词表
    :param max_sequence: 最大句子截断数
    :return:
    """
    model.eval()
    sample_num = 200  # 构造200个测试样本
    x, y = build_dataset(sample_num, vocab, max_sequence)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
    accuracy = correct / sample_num
    logger.info(f'正确预测个数：{correct}，正确率：{accuracy}')
    return accuracy

def log_display(train_log):
    """
    训练日志画图
    :param train_log: 训练日志
    :return:
    """
    plt.xlabel('epoch')
    plt.plot(range(len(train_log)), [l[1] for l in train_log], label='loss')  # 画loss曲线
    plt.plot(range(len(train_log)), [l[0] for l in train_log], label='acc')  # 画acc曲线
    plt.legend()
    plt.savefig('train.jpg')
    plt.show()

def train(model, epoch_num: int, train_dataset_x, train_dataset_y, batch_size: int, optimizer, model_save_path: str, vocab, max_sequence: int, vocab_path: str):
    """
    训练模型
    :param model: 模型
    :param epoch_num: 总轮数
    :param train_dataset_x: 训练集样本
    :param train_dataset_y: 训练集真实标签
    :param batch_size: 批量大小
    :param optimizer: 优化器
    :param model_save_path: 模型保存路径
    :param vocab: 词表
    :param vocab_path: 词表保存路径
    :param max_sequence: 句子最大截断数
    :return:
    """
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        dataset_total_num = len(train_dataset_x)
        for batch_index in range(dataset_total_num // batch_size):
            # 最后一次可能不足一个batch
            # 取出样本
            x = train_dataset_x[batch_index * batch_size: min((batch_index + 1) * batch_size, dataset_total_num)]
            # 取出标签
            y = train_dataset_y[batch_index * batch_size: min((batch_index + 1) * batch_size, dataset_total_num)]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 求梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度清零

            watch_loss.append(loss.item())

        mean_loss = np.mean(watch_loss)  # 计算平均loss

        logger.info(f'=========\n第{epoch + 1}轮平均loss:{mean_loss}')
        acc = evaluate(model, vocab, max_sequence)  # 验证
        log.append([acc, mean_loss])  # 记录训练数据

    log_display(log)  # 画图

    torch.save(model.state_dict(), model_save_path)  # 保存模型

    # 保存词表，词表与模型绑定
    with open(vocab_path, 'w', encoding='utf8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)



def predict(model_path, vocab_path, max_sequence, input_strings:list, char_dim):
    global spe_char
    """
    预测
    :param model_path: 模型路径
    :param vocab_path: 词表路径
    :param max_sequence: 最大截断数
    :param input_strings: 输入字符串
    :param char_dim: 一个字符需要的维度, 此参数应与训练时所对应
    :return:
    """
    # 读取模型的词表
    vocab = json.load(open(vocab_path, 'r', encoding='utf8'))

    # 使得模型的分类类别数是最大截断数 + 1，多出的1个是不出现的类别
    model = NlpMultiClassificationModule(vocab, char_dim, out_features=max_sequence + 1)

    # 加载权重
    model.load_state_dict(torch.load(model_path))

    # 组成一个batch输入模型预测
    x = []
    for input_string in input_strings:
        x.append(padding(max_sequence, [vocab[char] for char in input_string]))

    model.eval()
    with torch.no_grad():
        res = model.forward(torch.LongTensor(x))

        # 当输入字符串长度n=1时，res的形状是 torch.Size([10]), 而n>1时，形状为torch.Size([n, 10]), 形状不一致
        if len(input_strings) == 1:
            res = res.unsqueeze(0)

        for y_p, input_str in zip(res, input_strings):
            # 预测类别如果为0，则不出现，否则使用自然数顺序从1开始计算第n个位置
            category = torch.argmax(y_p)

            # 显示截断后的字符串输入
            display_str = input_str[: max_sequence] if len(input_str) > max_sequence else input_str

            if category == 0:
                text = f'预测{spe_char}不在字符串里'
                true_category = get_str_true_category(display_str)
            else:
                text = f'预测{spe_char}在字符串第{category}个位置上'
                true_category = get_str_true_category(display_str)

            print(f"输入: {display_str}, {text}, 预测类别: {torch.argmax(y_p)}, 真实类别: {true_category}, 预测{'正确' if true_category == category else '错误'}")

def get_str_true_category(input_str):
    """
    评估字符串真实分类和预测分类
    :param input_str: 输入字符串
    :return: 真实类别
    """
    global spe_char
    if spe_char not in input_str:
        return 0
    return input_str.index(spe_char) + 1  # 使用1作为开始

def batch_predict(test_sample_num, model_path, vocab_path, max_sequence, char_dim):
    """
    批量预测
    :param test_sample_num: 测试样本数
    :param model_path: 模型路径
    :param vocab_path: 词表路径
    :param max_sequence: 最大截断数
    :param char_dim: 一个字符需要的维度, 此参数应与训练时所对应
    :return:
    """
    vocab = json.load(open(vocab_path, 'r', encoding='utf8'))

    test_strings = [''.join(random.sample(list(vocab.keys())[1:-1], random.randint(max_sequence // 2, max_sequence))) for _ in range(test_sample_num)]
    print(f'构造{test_sample_num}个测试样本: {test_strings}')
    predict(model_path, vocab_path, max_sequence, test_strings, char_dim)

def main():
    # 训练参数
    epoch_num = 20
    batch_size = 20
    train_sample_num = 5000
    learning_rate = 0.001
    char_dim = 10

    logger.info(f"训练参数: epoch_num: {epoch_num}, train_sample_num: {train_sample_num}, batch_size:{batch_size}, learning_rate:{learning_rate}, char_dim:{char_dim}, max_sequence:{max_sequence}")

    # 建立词表
    vocab = build_vocab()

    # 建模
    # 使得模型的分类类别数是最大截断数 + 1，多出的1个是不出现的类别
    model = NlpMultiClassificationModule(vocab, char_dim, out_features=max_sequence + 1)

    # 选择Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建数据集
    train_x, train_y = build_dataset(train_sample_num, vocab, max_sequence)

    # 训练模型
    train(model=model,
          epoch_num=epoch_num,
          train_dataset_x=train_x,
          train_dataset_y=train_y,
          batch_size=batch_size,
          optimizer=optimizer,
          model_save_path='model.pth',
          vocab_path='vocab.json',
          max_sequence=max_sequence,
          vocab=vocab)

if __name__ == '__main__':
    max_sequence = 9  # 最大句子长度，全局设置
    spe_char = 'a'  # 要定位的字符

    # main()  # 训练

    test_sample_num = 20
    batch_predict(test_sample_num, 'model.pth', 'vocab.json', max_sequence, 10)  # 批量预测

    # while True:  # 单个输入字符串的测试模式
    #     input_str = input('请输入字符串：')
    #     predict('model.pth', 'vocab.json', 9, [input_str], 10)
