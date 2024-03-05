# 第七周作业

- config.py  
默认使用LSTM进行训练  
```python{.line-numbers}
# -*- coding: utf-8 -*-

Config = {
    "model_path": "model_output",
    "pretrain_model_path":r"D:\BaiduNetdiskDownload\bert-base-chinese",
    "train_data_path": "train.csv",
    "valid_data_path": "valid.csv",
    "model_type":"LSTM",
    "vocab_path":"chars.txt",
    "max_length": 463,
    "hidden_size": 256,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style":"max",
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987,
    "num_layers":1
}
```
- model.py  
```python{.line-numbers}
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "LSTM":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "RNN":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "CNN":
            self.encoder = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
            
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        
        if isinstance(x, tuple):
            x = x[0]
        
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict
```

- loader.py  
```python{.line-numbers}
def load_data(data_path, config, shuffle=True):
    data_generator = DataGenerator(data_path, config)
    data_loader = DataLoader(data_generator.data, batch_size=config["batch_size"], shuffle=shuffle)
    return data_loader
```
使用DataLoader进行数据封装  

- main.py
设置使用不同模型进行测试：  
```python{.line-numbers}
# train
    for modelType in optional_model:
        config["model_type"] = modelType
        for epoch in range(config["epoch"]):
            epoch += 1
            model.train()
            logger.info("epoch %d begin" % epoch)
            train_loss = []
            for index, batch_data in enumerate(train_data):
                if cuda_flag:
                    batch_data = [d.cuda() for d in batch_data]
                
                optimizer.zero_grad()
                input_ids, labels = batch_data
                loss = model(input_ids, labels)
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item()) 
                if index % int(len(train_data) / 2) == 0:
                    logger.info("batch loss: %f" % loss)
            logger.info("epoch average loss: %f" % np.mean(train_loss))
            acc = evaluator.eval(epoch)
        out_data = []
        
        keys = config.keys()
        for key in keys:
            value = config[key]
            out_data.append(value)
        out_data.extend((np.mean(train_loss), acc))
    return acc, out_data
```
