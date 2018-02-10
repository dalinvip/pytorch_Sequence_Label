# @Author : bamtercelboo
# @Datetime : 2018/1/15 11:47
# @File : Load_Pretrained_Embed.py
# @Last Modify Time : 2018/1/15 11:47
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Load_Pretrained_Embed.py
    FUNCTION : loading pretrained word embedding
    REFERENCE : https://github.com/Joyce94/pytorch-embedding-package/blob/master/Embedding.py
"""
import torch
from collections import OrderedDict
import numpy as np
import tqdm
import hyperparams as hy
torch.manual_seed(hy.seed_num)
np.random.seed(hy.seed_num)


def load_pretrained_emb_zeros(path, text_field_words_dict, pad=None, set_padding=False):
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    if pad is not None:
        padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}\n'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings = np.zeros((int(word_count), int(embedding_dim)))
    iov_num = 0
    oov_num = 0
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        # lines = tqdm.tqdm(lines)
        for line in lines:
            values = line.strip().split(' ')
            if len(values) == 1 or len(values) == 2:
                continue
            index = text_field_words_dict.get(values[0])  # digit or None
            if index:
                iov_num += 1
                vector = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index] = vector
            else:
                oov_num += 1
    f.close()

    return torch.from_numpy(embeddings).float()


def calculate_oov(path, text_field_words_dict, pad=None, set_padding=False):
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    if pad is not None:
        padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}\n'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings = np.zeros((int(word_count), int(embedding_dim)))
    iov_num = 0
    oov_num = 0
    word_list = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        # lines = tqdm.tqdm(lines)
        for line in lines:
            values = line.strip().split(' ')
            if len(values) == 1 or len(values) == 2:
                continue
            word_list.append(values[0])
    f.close()

    for w in text_field_words_dict:
        # print(w)
        if w in word_list:
            iov_num += 1
        else:
            oov_num += 1

    print("\niov_num {} oov_num {} oov_radio {}".format(iov_num, oov_num, round((oov_num / (oov_num + iov_num)), 2)))


def load_pretrained_emb_avg(path, text_field_words_dict, pad=None, set_padding=False):
    if not isinstance(text_field_words_dict, dict):
        text_field_words_dict = convert_list2dict(text_field_words_dict)
    assert pad is not None, "pad not allow with None"
    padID = text_field_words_dict[pad]
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    f.close()
    word_count = len(text_field_words_dict)
    print('The number of wordsDict is {} \nThe dim of pretrained embedding is {}\n'.format(str(word_count),
                                                                                           str(embedding_dim)))
    embeddings = np.zeros((int(word_count), int(embedding_dim)))

    inword_list = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        lines = tqdm.tqdm(lines)
        for line in lines:
            lines.set_description("Processing")
            values = line.strip().split(" ")
            if len(values) == 1 or len(values) == 2:
                continue
            index = text_field_words_dict.get(values[0])  # digit or None
            if index:
                vector = np.array([float(i) for i in values[1:]], dtype='float32')
                embeddings[index] = vector
                inword_list.append(index)
    f.close()

    sum_col = np.sum(embeddings, axis=0) / len(inword_list)     # avg
    for i in range(len(text_field_words_dict)):
        if i not in inword_list and i != padID:
            embeddings[i] = sum_col

    return torch.from_numpy(embeddings).float()


def convert_list2dict(convert_list):
    list_dict = OrderedDict()
    for index, word in enumerate(convert_list):
        list_dict[word] = index
    return list_dict


