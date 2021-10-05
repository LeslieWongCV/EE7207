# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 7:49 下午
# @Author  : Yushuo Wang
# @FileName: SVM.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/


from scipy.io import loadmat
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

PATH = 'Data/'
df_train = torch.from_numpy(loadmat(PATH + 'data_train.mat')['data_train']).float()
df_test = torch.from_numpy(loadmat(PATH + 'data_test.mat')['data_test']).float()
df_label = torch.from_numpy(loadmat(PATH + 'label_train.mat')['label_train'].squeeze()).float()
K = 4
kf = KFold(n_splits=K, shuffle=False)
kf.split(df_label)
vali_res = 0


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(n_hidden)
        self.threshold = 0

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.hidden2(out)
        # out = self.bn(out) GOOD
        out = torch.sigmoid(out)
        out = self.predict(out)

        return out

    def getThreshold(self, output, label):
        acc = 0
        threshold = 0
        th_list = np.arange(-0.5, 0.55, 0.05)
        ACC = []
        for i in th_list:
            acc_ = self.acc(output, i, label)
            ACC.append(acc_)
            print('threshold=%.2f' % i + ' | acc = %.5f' % acc_)
            if acc_ > acc:
                acc = acc_
                threshold = i
        print('*' * 20 + '\n')
        print('Choosing %0.2f as threshold' % threshold + 'acc = %.5f' % acc + '\n')
        print('*' * 20 + '\n')
        self.threshold = threshold
        plt.plot(th_list, ACC, c='orange')
        plt.ylabel('ACC')
        plt.xlabel('Threshold')
        plt.title('Acc with different thresholds on Validation set')
        plt.show()

    def acc(self, output, threshold, label, training=True):
        res = np.ones(output.shape)
        res[output > threshold] = 1
        res[output < threshold] = -1
        if training:
            return sum(np.squeeze(res) == label.detach().numpy()) / output.size()[0]
        else:
            return res


net = Net(33, 200, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
loss_func = torch.nn.MSELoss()
train_acc = 0
valid_acc = 0

train_index, valid_index = next(kf.split(df_label))
summary(net, (330, 33))
print("Loading", end="")

for i in range(10000):
    res_train = net(df_train[train_index]).squeeze()
    loss = loss_func(res_train, df_label[train_index])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('\b' * len(str(i)) + str(i), end='', flush=True)

    if i % 1000 == 0:
        print('Training:')
        net.getThreshold(res_train, df_label[train_index])
        print('Validation:')
        res_valid = net(df_train[valid_index]).squeeze()
        net.getThreshold(res_valid, df_label[valid_index])
        if valid_acc > 0.98:
            break

res_test = net.acc(net(df_test).squeeze(), net.threshold, None, False).reshape((21, 1))
