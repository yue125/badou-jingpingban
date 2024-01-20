word_p = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1,
}
sentence = "经常有意见分歧"


def cut_all(word_p, sentence, index, reverse=False):
    if len(sentence) - 1 == index:
        return [[sentence[index:]]]
    all_res = []
    for j in range(index + 1, len(sentence)):
        word = sentence[index:j]
        if reverse:
            word = word[::-1]
        if word in word_p:
            cut_child = cut_all(word_p, sentence, j)
            res = [[word] + child for child in cut_child]
            for r in res:
                if r not in all_res:
                    all_res.append(r)
    return all_res


cut_left = cut_all(word_p, sentence, 0)
cut_right = cut_all(word_p, sentence[::-1], 0, True)
cut_right = [r[::-1] for r in cut_right if r[::-1] not in cut_left]
for r in cut_left+cut_right:
    print(r)
