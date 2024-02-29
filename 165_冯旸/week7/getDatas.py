
from week7.config import Config

def get_datas(config):
    labels = []
    reviews = []
    with open('文本分类练习.csv', 'r', encoding='utf-8') as f:
        # 跳过第一行
        next(f)
        # 遍历每一行数据
        for line in f:
            # 处理每一行数据
            row = line.strip().split(',', 1)
            # print(row)
            if len(row[1]) > config["max_length"]:
                config["max_length"] = len(row[1]) - 2
            labels.append(row[0])
            reviews.append(row[1])
    return labels, reviews, config["max_length"]

if __name__ == "__main__":

    print(get_datas(Config)[2])