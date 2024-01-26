#Week4 HomeWork

dict = {'经常':1,
    '经':0,
    '常':0,
    '有':0,
    '意':0,
    '见':0,
    '分':0,
    '歧':0,
    '有意见':1,
    '意见':1,
    '分歧':1,
    '见分歧':1}


sentence = '经常有意见分歧'
max_length = len(max(dict, key = len))

def partition(sentence, dictionary):
    result = []
    max_length = len(max(dictionary, key=len))
    backtracking(sentence, 0, [], result, dictionary, max_length)
    return result

def backtracking(s, start_index, path, result, dictionary, max_length):
    if start_index == len(s):
        result.append(path)
        return

    for i in range(start_index, min(start_index + max_length, len(s))):
        prefix = s[start_index:i + 1]
        if prefix in dictionary:
            backtracking(s, i + 1, path + [prefix], result, dictionary, max_length)