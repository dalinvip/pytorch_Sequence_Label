# @Author : bamtercelboo
# @Datetime : 2018/2/10 17:37
# @File : model_BiLSTM.py
# @Last Modify Time : 2018/2/10 17:37
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  model_PNC.py
    FUNCTION : Named Entity Recognition(NER) use BiLSTM Neural Networks Model
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


class BiLSTM(nn.Module):

    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.hidden_size = args.rnn_hidden_size
        self.input_size = args.rnn_input_size
        paddingId = args.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.dropout = nn.Dropout(args.dropout)

        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, dropout=args.dropout,
                              bidirectional=True, bias=True)

        # self.linear = nn.Linear(in_features=D * self.cat_size, out_features=C, bias=True)
        self.linear = nn.Linear(in_features=self.hidden_size * 2, out_features=C, bias=True)
        init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.uniform_(-np.sqrt(6 / (self.hidden_size + 1)), np.sqrt(6 / (self.hidden_size + 1)))

    def forward(self, batch_features):
        word = batch_features.word_features
        x = self.embed(word)  # (N,W,D)
        x = self.dropout_embed(x)
        cated_embed, _ = self.bilstm(x)
        cated_embed = F.tanh(cated_embed)
        logit = self.linear(cated_embed)
        return logit

