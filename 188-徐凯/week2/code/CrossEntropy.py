import torch
import torch.nn as nn
import numpy as np


"""
手动实现交叉熵的计算
"""

# 使用 torch 计算交叉熵
ce_loss = nn.CrossEntropyLoss()
# 假设有 3 个样本，每个都在做 3 分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]])  # n * class_num
# 正确的类别分别是 1, 2, 0
target = torch.LongTensor([1, 2, 0])  # n
loss = ce_loss(pred, target)
print(f"torch输出交叉熵：{loss}")


# 手动实现交叉熵损失函数
print("=" * 20, "手动实现交叉熵损失函数", "=" * 20)

# 1. 实现 softmax 函数 ，因为交叉熵损失函数输入的预测值是一个概率分布，不能负值，并且和为1
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)  # keepdims 保留结构不变

# 验证 softmax 函数
# print(torch.softmax(pred, dim=1))
# print(softmax(pred.numpy()))

# 2. 将输入转化为 onehot 矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    # 实现 one-hot 编码
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1

    return one_hot_target

# 3. 实现交叉熵损失函数
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = -np.sum(target * np.log(pred), axis=1)
    # print(entropy)

    return sum(entropy) / batch_size

print(f"手动实现交叉熵：\n{cross_entropy(pred.numpy(), target.numpy())}")