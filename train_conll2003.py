# @Author : bamtercelboo
# @Datetime : 2018/1/31 10:01
# @File : train.py
# @Last Modify Time : 2018/1/31 10:01
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  train.py
    FUNCTION : None
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import shutil
import random
from eval import Eval, EvalPRF
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


def train(train_iter, dev_iter, test_iter, model, args):
    if args.use_cuda:
        model.cuda()

    optimizer = None
    if args.Adam is True:
        print("Adam Training......")
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)

    file = open("./Test_Result.txt", encoding="UTF-8", mode="a", buffering=1)
    best_fscore = Best_Result()

    steps = 0
    model_count = 0
    model.train()
    max_dev_acc = -1
    train_eval = Eval()
    dev_eval = Eval()
    test_eval = Eval()
    for epoch in range(1, args.epochs+1):
        print("\n## The {} Epoch，All {} Epochs ！##".format(epoch, args.epochs))
        print("now lr is {}".format(optimizer.param_groups[0].get("lr")))
        random.shuffle(train_iter)
        model.train()
        for batch_count, batch_features in enumerate(train_iter):
            model.zero_grad()
            logit = model(batch_features)
            loss_logit = logit.view(logit.size(0) * logit.size(1), logit.size(2))
            loss = F.cross_entropy(loss_logit, batch_features.label_features)
            # print(loss)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f}".format(batch_count + 1, loss.data[0]))
        if steps is not 0:
            dev_eval.clear_PRF()
            eval(dev_iter, model, dev_eval, file, best_fscore, epoch, args, test=False)
        if steps is not 0:
            test_eval.clear_PRF()
            eval(test_iter, model, test_eval, file, best_fscore, epoch, args, test=True)


def eval(data_iter, model, eval_instance, file, best_fscore, epoch, args, test=False):
    # eval time
    eval_PRF = EvalPRF()
    gold_labels = []
    predict_labels = []
    for batch_features in data_iter:
        logit = model(batch_features)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            # eval_PRF = EvalPRF()
            predict_label = []
            for id_word in range(inst.words_size):
                maxId = getMaxindex(logit[id_batch][id_word], logit.size(2), args)
                # if maxId == args.create_alphabet.label_unkId:
                #     continue
                predict_label.append(args.create_alphabet.label_alphabet.from_id(maxId))
            gold_labels.append(inst.labels)
            predict_labels.append(predict_label)
            # print(inst.labels)
            # print(predict_labels)
            eval_PRF.evalPRF(predict_labels=predict_label, gold_labels=inst.labels, eval=eval_instance)
    # p, r, f = entity_evalPRF_exact(gold_labels=gold_labels, predict_labels=predict_labels)
    #
    # calculate the F-Score
    p, r, f = eval_instance.getFscore()
    # p = p * 100
    # f = f * 100
    # r = r * 100
    test_flag = "Test"
    if test is False:
        print("\n")
        test_flag = "Dev"
        if f > best_fscore.best_dev_fscore:
            best_fscore.best_dev_fscore = f
            best_fscore.best_epoch = epoch
            best_fscore.best_test = True
    if test is True and best_fscore.best_test is True:
        best_fscore.p = p
        best_fscore.r = r
        best_fscore.f = f
    print("{} eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(test_flag, p, r, f))
    if test is True:
        print("The Current Best Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_fscore.best_dev_fscore,
                                                                                 best_fscore.best_epoch))
    # if test is True and best_fscore.best_test is True:
        print("The Current Best Test Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(
            best_fscore.p, best_fscore.r, best_fscore.f))
    if test is False:
        file.write("The {} Epoch, All {} Epoch.\n".format(epoch, args.epochs))
    file.write("{} eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%\n".format(test_flag, p, r, f))
    if test is True:
        file.write("The Current Best Dev F-score: {:.6f}, Locate on {} Epoch.\n".format(best_fscore.best_dev_fscore, best_fscore.best_epoch))
        file.write("The Current Best Test Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%\n\n".format(
            best_fscore.p, best_fscore.r, best_fscore.f))
    if test is True:
        best_fscore.best_test = False
    # print("\neval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%\n".format(p * 100, r * 100, f * 100))


def getMaxindex(model_out, label_size, args):
    # print(model_out.size())
    # print(label_size)
    # print(model_out)
    max = model_out.data[0]
    maxIndex = 0
    for idx in range(1, label_size):
        if model_out.data[idx] > max:
            max = model_out.data[idx]
            maxIndex = idx
    # print(maxIndex)
    return maxIndex


class Best_Result:
    def __init__(self):
        self.best_dev_fscore = -1
        self.best_fscore = -1
        self.best_epoch = 1
        self.best_test = False
        self.p = -1
        self.r = -1
        self.f = -1


