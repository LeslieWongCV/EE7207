# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 7:49 下午
# @Author  : Chen
# @FileName: RBF.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

from sklearn.cluster import KMeans
from scipy.linalg import norm
import numpy as np
import scipy.io as scio
import math
from sklearn.model_selection import KFold

PATH = 'Data/'
data_test = scio.loadmat(PATH + 'data_test.mat')['data_test']
data_train = scio.loadmat(PATH + 'data_train.mat')['data_train']
label_train = scio.loadmat(PATH + 'label_train.mat')['label_train']

center_num = 10
model = KMeans(n_clusters=center_num, max_iter=1500)
model.fit(data_train)
centers = model.cluster_centers_


class RBF(object):
    global center_num

    def __init__(self, centers, input_data, label, test_data):
        self.input_data = input_data
        self.label = label
        self.test_data = test_data
        self.centers = centers
        self.w = np.array([np.random.rand(center_num + 1)])
        self.sigma = self.getSigma(self.centers)
        self.threshold = 0

    def getSigma(self, centers):
        sigma_ = 0
        for i in centers:
            for j in centers:
                if abs(norm(i) - norm(j)) > sigma_:
                    sigma_ = abs(norm(i) - norm(j))
        return sigma_

    def basis(self, data, center):
        # sigma = 0.707
        return math.exp(-norm(data - center) ** 2 / (2 * self.sigma ** 2))

    def calcAct(self, input_data):
        act_val = np.zeros((input_data.shape[0], center_num))
        for xi, x in enumerate(input_data):
            for ci, c in enumerate(self.centers):
                act_val[xi][ci] = self.basis(x, c)
        return act_val

    def getThreshold(self, output, label):
        acc = 0
        threshold = 0
        th_list = np.arange(-0.5, 0.55, 0.05)
        for i in th_list:
            acc_ = self.acc(output, i, label)
            print('threshold=%.2f' % i + ' | acc = %.5f' % acc_)
            if acc_ > acc:
                acc = acc_
                threshold = i
        print('*' * 20 + '\n')
        print('Choosing %0.2f as threshold' % threshold + '\n')
        print('*' * 20 + '\n')
        self.threshold = threshold

    def acc(self, output, threshold, label, training=True):
        res = np.ones(output.shape)
        res[output > threshold] = 1
        res[output < threshold] = -1
        if training:
            return sum(res == label) / output.shape[0]
        else:
            return res

    def train(self):
        act_val = self.calcAct(self.input_data)
        w0_column = np.array([1 for _ in range(len(act_val))])
        act_val = np.insert(act_val, len(act_val[0]), values=w0_column, axis=1)
        self.w = np.linalg.inv(act_val.T.dot(act_val)).dot(act_val.T).dot(self.label)
        output = act_val.dot(self.w)
        self.getThreshold(output, label=self.label)
        return output

    def valid(self, data, label):
        act_val_test = self.calcAct(data)
        w0_column = np.array([1 for _ in range(len(act_val_test))])
        act_val_test = np.insert(act_val_test, len(act_val_test[0]), values=w0_column, axis=1)
        output = act_val_test.dot(self.w)
        self.getThreshold(output,label)
        return output, self.acc(output, self.threshold, label)

    def test(self):
        act_val_test = self.calcAct(self.test_data)
        w0_column = np.array([1 for _ in range(len(act_val_test))])
        act_val_test = np.insert(act_val_test, len(act_val_test[0]), values=w0_column, axis=1)
        output = act_val_test.dot(self.w)
        return output, self.acc(output, self.threshold, None, False)


K = 4
kf = KFold(n_splits=K, shuffle=False)
train_index, valid_index = next(kf.split(label_train))

data_valid = data_train[valid_index]
label_valid = label_train[valid_index]
data_train = data_train[train_index]
label_train = label_train[train_index]

network = RBF(centers, data_train, label_train, data_test)
output = network.train()

output_valid_raw, output_valid_acc = network.valid(data_valid, label_valid)
output_test_raw, output_test = network.test()
_ = 1
