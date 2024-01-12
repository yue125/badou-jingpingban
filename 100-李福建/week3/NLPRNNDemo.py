import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断字母出现的位置

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, sentence_length, batch_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(sentence_length, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy  # 均方差loss
        self.pool = nn.AvgPool1d(sentence_length)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, sentence_length)  # 线性层 5 6
        self.batch_size = batch_size

    def forward(self, x, y=None):  # y -> 20 1
        x = self.embedding(x)  # x 20 6 5
        ht = torch.zeros(1, self.batch_size, self.hidden_size)  # 1 20 5
        out, hidden_prev = self.rnn(x, ht)  # out -> 20 6 5, h -> 1 20 5
        out = out.transpose(1, 2)  # 20 5 6
        out = self.pool(out)  # 20 5 1
        out = out.squeeze()  # 20 5
        out = self.linear(out)  # 20 6
        ht = hidden_prev
        if y is not None:
            return self.loss(out, y.squeeze())
        else:
            return out


def build_vocab():
    chars = "abcdef"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index  # 每个字对应一个序号
    # vocab['unk'] = len(vocab)  # 7
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    x = []
    while len(x) < sentence_length:
        char = random.choice(list(vocab.keys()))
        if char not in x:
            x.append(char)

    if x.index("a") == 0:
        y = 0
    elif x.index("a") == 1:
        y = 1
    elif x.index("a") == 2:
        y = 2
    elif x.index("a") == 3:
        y = 3
    elif x.index("a") == 4:
        y = 4
    else:
        y = 5
    x = [vocab.get(word) for word in x]
    return x, y


# 构建数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(input_size, hidden_size, sentence_length, batch_size):
    return TorchModel(input_size, hidden_size, sentence_length, batch_size)


# 测试代码
def evaluate(model, batch_size, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(batch_size, vocab, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练逻辑
def main():
    input_size = 5  # 维度
    hidden_size = 5  # 隐藏维度
    sentence_length = 6  # 字段长度
    train_sample = 1000  # 每轮训练总共训练的样本总数
    batch_size = 20  # 每轮训练批量训练大小
    epoch_num = 20  # 训练轮数
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(input_size, hidden_size, sentence_length, batch_size)
    # 构建优化器
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    log = []
    for epoch in range(epoch_num):
        # 训练模式
        model.train()
        watch_loss = []
        for index in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            # x = x.view(1, batch_size, sentence_length)
            # y = y.view(1, batch_size, sentence_length)
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, batch_size, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "week3_job_model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, vocab_path, input_strings):
    input_size = 5
    hidden_size = 5
    sentence_length = 6
    batch_size = 20
    model = build_model(input_size, hidden_size, sentence_length, batch_size)
    model.load_state_dict(torch.load(model_path))
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    print(model.state_dict())

    x = []
    for i in range(5):
        for input_string in input_strings:
            x.append([vocab[char] for char in input_string])  # 将输入序列化

    model.eval()  # 测试模式
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for input_string, res in zip(input_strings, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["abcdef", "bcdaef", "badcef", "cdefba"]
    predict("week3_job_model.pt", "vocab.json", test_strings)
