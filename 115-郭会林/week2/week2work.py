import torch
import torch.nn as nn
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个6维向量
    如果第1个数最大，属于第1类
    如果第2个数最大，属于第2类
    如果第3个数最大，属于第3类
    如果第4个数最大，属于第4类
    如果第5个数最大，属于第5类
    如果第6个数最大，属于第6类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 6)
        self.activation = torch.softmax  #softmax
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x) # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x, 1)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(6)
    index_of_max = np.argmax(x)
    if index_of_max == 0:
        return x, 0
    elif index_of_max == 1:
        return x, 1
    elif index_of_max == 2:
        return x, 2
    elif index_of_max == 3:
        return x, 3
    elif index_of_max == 4:
        return x, 4
    elif index_of_max == 5:
        return x, 5

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    count_dict = Counter(y.tolist())
    # 输出结果
    print('本次预测集中共有:')
    for key, value in count_dict.items():
        print(f"{key} 类样本 {value} 个")
    correct, wrong = 0, 0
    with torch.no_grad(): # 不计算梯度
        y_pred = model(x) # 模型预测

        for y_p, y_t in zip(y_pred, y): #与真实标签进行对比
            index_of_max = np.argmax(y_p)
            if index_of_max == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():

    # 配置参数
    epoch_num = 500 #轮数
    batch_size = 10 #每次样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 6 #输入向量维度
    learning_rate = 0.01 #学习率

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 模型进入训练状态
        watch_loss = [] # 打印loss变化
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss
            loss.backward() # 计算梯度
            optim.step() #更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("============\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_my.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc") # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss") # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval() #测试模式
    with torch.no_grad(): #不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        index_of_max = np.argmax(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, index_of_max, res[index_of_max]))  # 打印结果

if __name__ == "__main__":
    main()
    # x, _  = build_dataset(10)
    # print(x.tolist())
    # test_vec = [[0.008492554537951946, 0.8390341997146606, 0.6726710200309753, 0.03270767629146576, 0.8565715551376343, 0.5021806955337524, 0.3565967381000519, 0.3313722610473633, 0.18774215877056122, 0.18569430708885193], [0.25283703207969666, 0.4021325409412384, 0.535735547542572, 0.17102143168449402, 0.27582836151123047, 0.5800834894180298, 0.4052554965019226, 0.33240148425102234, 0.21116173267364502, 0.23454982042312622], [0.6660555005073547, 0.03896741941571236, 0.3526579737663269, 0.09430383890867233, 0.34591904282569885, 0.8890732526779175, 0.7589467167854309, 0.39558637142181396, 0.5486478805541992, 0.9393573999404907], [0.07344057410955429, 0.9943010807037354, 0.3448270559310913, 0.15509502589702606, 0.6854188442230225, 0.6565245985984802, 0.36320918798446655, 0.03627978265285492, 0.8149006962776184, 0.24668273329734802], [0.42127299308776855, 0.876416027545929, 0.4468599259853363, 0.4330522418022156, 0.2995111048221588, 0.831609845161438, 0.8798548579216003, 0.9234006404876709, 0.9894086718559265, 0.7413456439971924], [0.8959604501724243, 0.29032063484191895, 0.7513151168823242, 0.4164835512638092, 0.24099181592464447, 0.3291573226451874, 0.494401752948761, 0.6629710793495178, 0.4535684883594513, 0.7568011283874512], [0.7200154662132263, 0.46178367733955383, 0.7913718223571777, 0.4682105481624603, 0.7468719482421875, 0.4402120113372803, 0.4407724440097809, 0.3259066045284271, 0.7599390149116516, 0.18274793028831482], [0.406130850315094, 0.17169837653636932, 0.6197201013565063, 0.7806094884872437, 0.8109942674636841, 0.6927044987678528, 0.12364725768566132, 0.28341877460479736, 0.7296603322029114, 0.44541141390800476], [0.3786452114582062, 0.9003918766975403, 0.9838286638259888, 0.5061833262443542, 0.33409544825553894, 0.5477629899978638, 0.43292611837387085, 0.7741659879684448, 0.300422340631485, 0.9132953882217407], [0.5635395646095276, 0.31165093183517456, 0.7243753671646118, 0.5707677006721497, 0.8036786913871765, 0.5419642925262451, 0.5316941738128662, 0.1845809370279312, 0.09598838537931442, 0.895081639289856]]
    # predict("model_my.pt", test_vec)