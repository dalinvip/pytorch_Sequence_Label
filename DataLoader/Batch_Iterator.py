# @Author : bamtercelboo
# @Datetime : 2018/2/10 17:38
# @File : Batch_Iterator.py
# @Last Modify Time : 2018/2/10 17:38
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Batch_Iterator.py
    FUNCTION : None
"""

import torch
from torch.autograd import Variable
import random
from DataLoader.Common import unkkey, paddingkey
from DataLoader.Instance import Instance
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class Batch_Features:
    def __init__(self):

        self.batch_length = 0
        self.inst = None
        self.word_features = 0
        self.label_features = 0

    def cuda(self, features):
        features.word_features = features.word_features.cuda()
        features.label_features = features.label_features.cuda()


class Iterators:
    def __init__(self):
        self.args = None
        self.batch_size = None
        self.data = None
        self.operator = None
        self.operator_static = None
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

    def createIterator(self, batch_size=None, data=None, operator=None, args=None):
        assert isinstance(data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(batch_size, list), "ERROR: batch_size must be in list [16,1,1]"
        self.args = args
        self.batch_size = batch_size
        self.data = data
        self.operator = operator
        for id_data in range(len(data)):
            print("*****************    create {} iterator    **************".format(id_data + 1))
            self.convert_word2id(self.data[id_data], self.operator)
            self.features = self.Create_Each_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                                       operator=self.operator)
            self.data_iter.append(self.features)
            self.features = []
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    def convert_word2id(self, insts, operator):
        # print(len(insts))
        # for index_inst, inst in enumerate(insts):
        for inst in insts:
            # copy with the word and pos
            for index in range(inst.words_size):
                word = inst.words[index]
                wordId = operator.word_alphabet.loadWord2idAndId2Word(word)
                # if wordID is None:
                if wordId == -1:
                    wordId = operator.word_unkId
                inst.words_index.append(wordId)

                label = inst.labels[index]
                labelId = operator.label_alphabet.loadWord2idAndId2Word(label)
                inst.label_index.append(labelId)

    def Create_Each_Iterator(self, insts, batch_size, operator):
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            # print(batch)
            if len(batch) == batch_size or count_inst == len(insts):
                # print("aaaa", len(batch))
                one_batch = self.Create_Each_Batch(insts=batch, batch_size=batch_size, operator=operator)
                self.features.append(one_batch)
                batch = []
        print("The all data has created iterator.")
        return self.features

    def Create_Each_Batch(self, insts, batch_size, operator):
        # print("create one batch......")
        batch_length = len(insts)
        # copy with the max length for padding
        max_word_size = -1
        max_label_size = -1
        for inst in insts:
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size

            if len(inst.labels) > max_label_size:
                max_label_size = len(inst.labels)

        # create with the Tensor/Variable
        # word features
        batch_word_features = Variable(torch.zeros(batch_length, max_word_size).type(torch.LongTensor))
        batch_label_features = Variable(torch.zeros(batch_length * max_word_size).type(torch.LongTensor))

        for id_inst in range(batch_length):
            inst = insts[id_inst]
            # copy with the word features
            for id_word_index in range(max_word_size):
                if id_word_index < inst.words_size:
                    batch_word_features.data[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features.data[id_inst][id_word_index] = operator.word_paddingId

                if id_word_index < len(inst.label_index):
                    batch_label_features.data[id_inst * max_word_size + id_word_index] = inst.label_index[id_word_index]
                else:
                    # batch_label_features.data[id_inst * max_word_size + id_word_index] = operator.label_unkId
                    # batch_label_features.data[id_inst * max_word_size + id_word_index] = 0
                    batch_label_features.data[id_inst * max_word_size + id_word_index] = operator.label_alphabet.loadWord2idAndId2Word("O")

        # batch
        features = Batch_Features()
        features.batch_length = batch_length
        features.inst = insts
        features.word_features = batch_word_features
        features.label_features = batch_label_features

        if self.args.use_cuda is True:
            features.cuda(features)
        return features
