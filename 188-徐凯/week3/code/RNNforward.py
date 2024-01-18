import torch
import torch.nn as nn
import numpy as np

"""
手动实现简单的神经网络
使用 pytorch 实现 RNN 
手动实现 RNN
对比
"""


class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.layer = nn.RNN(input_size,  # 输入维度
                            hidden_size,  # 隐单元个数
                            bias=False,  # 是否加入偏置项
                            batch_first=True)  # batch_first=True -> (batch_size, seq, features)

    def forward(self, x):
        return self.layer(x)


# 自定义RNN模型
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        ht = np.zeros((self.hidden_size))
        output = []
        # print(x.shape, "x")
        # print(self.w_ih.shape, "w_ih")
        # print(ht.shape, "ht")
        for xt in x:
            ux = np.dot(self.w_ih, xt)  # 4*3 3*1 --> 4*1
            wh = np.dot(self.w_hh, ht)  # 4*4 4*1 --> 4*1
            ht_next = np.tanh(ux + wh)  # 4*1 --> 1*4
            output.append(ht_next)  # 3*4
            ht = ht_next
            # print(xt.shape, "xt")
            # print(ux.shape, "ux")
            # print(wh.shape, "wh")
            # print(ht_next.shape, "ht_next")
            # break

        return np.array(output), ht


# 定义数据
# 网络输入
x = np.array([[1, 2, 3],
              [3, 4, 5],
              [5, 6, 7]])

# torch实验
hidden_size = 4
torch_model = TorchRNN(3, hidden_size)

# print(torch_model.state_dict())
w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
print(w_ih, w_ih.shape)
print(w_hh, w_hh.shape)

torch_x = torch.FloatTensor(x.reshape((1, 3, 3)))
output, h = torch_model.forward(torch_x)
# print(h)
print(output.detach().numpy(), "torch模型预测结果")
print(h.detach().numpy(), "torch模型预测隐含层结果")
print("-" * 30)

diy_model = DiyModel(w_ih, w_hh, hidden_size)
output, h = diy_model.forward(x)
print(output, "diy模型预测结果")
print(h, "diy模型预测隐含层结果")