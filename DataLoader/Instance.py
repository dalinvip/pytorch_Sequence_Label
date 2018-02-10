# @Author : bamtercelboo
# @Datetime : 2018/2/10 17:39
# @File : Instance.py
# @Last Modify Time : 2018/2/10 17:39
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Instance.py
    FUNCTION : Data Instance
"""

import torch
import random
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


class Instance:
    def __init__(self):
        self.words = []
        self.labels = []
        self.words_size = 0

        self.words_index = []
        self.label_index = []