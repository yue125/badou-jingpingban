
提问：
**  1. 数据"vocab_path":"chars.txt" 有什么作用
**  2. 将下列性能指标加入表格
    # 计算性能指标
    accuracy_lr = accuracy_score(val_data['label'], val_pred_lr)
    precision_lr = precision_score(val_data['label'], val_pred_lr)
    recall_lr = recall_score(val_data['label'], val_pred_lr)
    f1_lr = f1_score(val_data['label'], val_pred_lr)
    数据分析：正负样本数，文本平均长度等
    实验对比3种以上模型结构的分类效果
    每种模型对比模型预测速度
