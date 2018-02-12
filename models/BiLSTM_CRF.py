# @Author : bamtercelboo
# @Datetime : 2018/2/12 15:55
# @File : BiLSTM-CRF.py
# @Last Modify Time : 2018/2/12 15:55
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  BiLSTM-CRF.py
    FUNCTION : None
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as Variable
import random
import torch.nn.init as init
import numpy as np
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class BiLSTM_CRF(nn.Module):

    def __init__(self, BiLSTM, CRF, args):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM
        self.CRF = CRF
        self.args = args

    def forward(self, batch_features, train=True):
        words = batch_features.word_features
        labels = batch_features.label_features
        if train is True:
            # print("train")
            tag_scores = self.bilstm(words)
            tag_scores = tag_scores.view(tag_scores.size(0) * tag_scores.size(1), -1)
            loss = self.CRF.neg_log_likelihood(tag_scores, labels, self.args.batch_size)
            return loss
        else:
            # print("eval")
            tag_hiddens = self.bilstm(words)
            tag_hiddens = tag_hiddens.view(tag_hiddens.size(0) * tag_hiddens.size(1), -1)
            _, best_path = self.CRF.viterbi_decode(tag_hiddens)
            return best_path
