# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 7:49 下午
# @Author  : Yushuo Wang
# @FileName: SVM.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.svm import SVC


PATH = 'Data/'
df_train = loadmat(PATH + 'data_train.mat')['data_train']
df_test = loadmat(PATH + 'data_test.mat')['data_test']
df_label = loadmat(PATH + 'label_train.mat')['label_train'].squeeze()
K = 4
kf = KFold(n_splits=K, shuffle=False)
# kf.split(df_label)
acc_train = 0
acc_valid = 0

for train_index, valid_index in kf.split(df_label):  # 4-fold

    model = SVC(C=1.0, kernel='rbf', gamma=0.1)
    model.fit(df_train[train_index], df_label[train_index])

    res_train = model.predict(df_train[train_index])
    res_valid = model.predict(df_train[valid_index])

    acc_train += float(sum(res_train == df_label[train_index]) / len(train_index))
    acc_valid += float(sum(res_valid == df_label[valid_index]) / len(valid_index))

    #print('ACC using SVM: ' + str(acc_train) + ' | ' + 'Valid: ' + str(acc_valid))

print('ACC using SVM: ' + str(acc_train / K) + ' | ' + 'Valid: ' + str(acc_valid / K))
