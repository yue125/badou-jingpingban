# 导入第三方库
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

"""
基于 pytorch 的网络编写
手动实现梯度计算和反向传播
加入激活函数
"""


class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        # 线性层
        # w.shape = (hidden_size, hidden_size)
        # 不添加偏置项 wx+b --> wx
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)
        # sigmoid 非线性激活函数
        self.activation = torch.sigmoid
        # mse 损失函数
        self.loss = F.mse_loss

    # 当输入真实标签，返回 loss 值；否则，返回预测值
    def forward(self, x, y=None):
        x = self.layer(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)

        return y_pred


# 自定义模型，接受一个参数矩阵作为入参
class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        # print("x shape：", x.shape)
        # print("w shape：", self.weight.shape)
        x = np.dot(x, self.weight.T)
        y_pred = self.diy_sigmoid(x)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)

        return y_pred

    # 自定义 sigmoid 函数
    @staticmethod
    def diy_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 自定义 mse 损失函数
    @staticmethod
    def diy_mse_loss(y_pred, y_true):
        return np.sum((y_pred - y_true) ** 2) / len(y_pred)

    # 手动实现梯度计算
    @staticmethod
    def calculate_grad(y_pred, y_true, x):
        # 反向过程
        # 1. 均方差导数
        # 均方差函数 (y_pred - y_true) ^ 2 / n 的导数是 2 * (y_pred - y_true) / n，结果为 2 维向量
        grad_mse = 2 * (y_pred - y_true) / len(x)
        # 2. sigmoid 导数
        # sigmoid 函数 y = 1 / (1 + e ^ (-x)) 的导数是 y * (1 - y)，结果为 2 维向量
        grad_sigmoid = y_pred * (1 - y_pred)

        # wx矩阵运算，wx = [w11*x0 + w21*x1, w12*x0, w22*x1]
        # 导数链式法则
        grad_w11 = grad_mse[0] * grad_sigmoid[0] * x[0]
        grad_w21 = grad_mse[0] * grad_sigmoid[0] * x[1]
        grad_w12 = grad_mse[1] * grad_sigmoid[1] * x[0]
        grad_w22 = grad_mse[1] * grad_sigmoid[1] * x[1]

        grad = np.array([[grad_w11, grad_w12],
                         [grad_w21, grad_w22]])
        # 由于 pytorch 存储做了转置，输出时也做转置处理
        return grad.T


# 自定义 sgd 优化器
def diy_sgd(grad, weight, learning_rate):
    return weight - learning_rate * grad

# 自定义 adam 优化器
def diy_adam(grad, weight):
    # 参数应当放在外面，此处为保持后方代码整洁简单实现一步
    alpha = 1e-3  # 学习率
    beta1 = 0.9   # 超参数
    beta2 = 0.999 # 超参数
    eps = 1e-8    # 超参数
    t = 0         # 初始化
    mt = 0        # 初始化
    vt = 0        # 初始化
    # 开始计算
    t += 1
    gt = grad
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * gt ** 2
    mth = mt / (1 - beta1 ** t)
    vth = vt / (1 - beta2 ** t)
    weight -= alpha * mth / (np.sqrt(vth) + eps)

    return weight


# 创建数据
x = np.array([-0.5, 0.1])  # 输入
y = np.array([0.1, 0.2])  # 预期输出

# torch实验
print("=" * 20, " torch 实验", "=" * 20)
torch_model = TorchModel(2)
torch_model_w = torch_model.state_dict()["layer.weight"]
print(f"torch 初始化权重:\n{torch_model_w}")
numpy_model_w = copy.deepcopy(torch_model_w.numpy())  # 复制一份权重参数，后续用于自定义模型使用

# numpy array --> torch tensor, unsqueeze 的目的是增加一个 batchsize 维度
torch_x = torch.from_numpy(x).float().unsqueeze(0)
torch_y = torch.from_numpy(y).float().unsqueeze(0)

# 1. torch 的前向计算，得到 loss
torch_loss = torch_model(torch_x, torch_y)
print(f"torch 模型计算 loss：\n{torch_loss}")
# 2. torch 的反向传播和梯度更新
# 定义优化器
learning_rate = 0.1  # 定义学习率
# optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(torch_model.parameters())
# 反向传播
torch_loss.backward()
print(f"torch 计算梯度：\n{torch_model.layer.weight.grad}")
# 更新权重参数
optimizer.step()
print(f"torch 更新后权重：\n{torch_model.state_dict()['layer.weight']}")
# 梯度清零
optimizer.zero_grad()

# 手动实验
print("=" * 20, "手动实验", "=" * 20)
# 1. 手动实现前向计算，得到 loss
diy_model = DiyModel(numpy_model_w)
diy_loss = diy_model.forward(x, y)
print(f"diy 模型计算 loss：\n{diy_loss}")
# 2. 手动实现反向传播和梯度更新
# 反向传播计算梯度
grad_diy = diy_model.calculate_grad(diy_model.forward(x), y, x)
print(f"diy 计算梯度：\n{grad_diy}")
# 更新权重参数
# print(f"diy 更新后权重：\n{diy_sgd(grad_diy, numpy_model_w, learning_rate)}")
print(f"diy 更新后权重：\n{diy_adam(grad_diy, numpy_model_w)}")
