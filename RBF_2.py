# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 7:49 下午
# @Author  : Chen
# @FileName: RBF_2.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

from sklearn.cluster import KMeans
from scipy.linalg import norm
import numpy as np
import scipy.io as scio
import math
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

    def __init__(self, centers, iter_num, input_data, label, test_data):
        self.input_data = input_data
        self.label = label
        self.test_data = test_data
        self.iter_num = iter_num
        self.centers = centers
        self.w = np.array([np.random.rand(center_num+1)])

    def basis(self, data, center):
        sigma = np.amax(norm(data-center))/math.sqrt(2*center_num)
        return math.exp(-norm(data-center)**2/(2*sigma*sigma))

    def calcAct(self, input_data):
        act_val = np.zeros((input_data.shape[0], center_num))
        for xi, x in enumerate(input_data):
            for ci, c in enumerate(self.centers):
                act_val[xi][ci] = self.basis(c, x)
        return act_val

    def train(self):
        act_val = self.calcAct(self.input_data)
        w0_column = np.array([1 for _ in range(len(act_val))])
        act_val = np.insert(act_val, len(act_val[0]), values=w0_column, axis=1)
        output = act_val.dot(self.w.T)
        i = 0
        while i < self.iter_num:
            self.w = np.linalg.inv(act_val.T.dot(act_val)).dot(act_val.T).dot(self.label)
            output = act_val.dot(self.w)
            i += 1
        return output, act_val

    def test(self):
        act_val_test = self.calcAct(self.test_data)
        w0_column = np.array([1 for _ in range(len(act_val_test))])
        act_val_test = np.insert(act_val_test, 10, values=w0_column, axis=1)
        output = act_val_test.dot(self.w)
        return output

network = RBF(centers, 1000, data_train, label_train, data_test)
output, actval = network.train()
res = network.test()


