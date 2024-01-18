import torch
import torch.nn as nn

"""
Pooling层的处理
"""

# pooling操作默认对于输入张量的最后一维进行
# 入参4，代表把四维池化为一维
layer = nn.AvgPool1d(4)  # 要对文本长度的维度进行池化
# 随机生成一个维度为3x4x5的张量
# 可以想象成3条文本长度为4向量长度为5的样本
x = torch.rand([3, 4, 5])
print(x)
print(x.shape)
# 为了进行池化操作，需要进行交换维度
x = x.transpose(1, 2)
print(x.shape, "交换后的形状")

# 经过pooling层
y = layer(x)
print(y)
print(y.shape)
# squeeze方法去掉值为1的维度
y = y.squeeze()
print(y)
print(y.shape)
