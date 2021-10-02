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

PATH = 'Data/'
df_train = torch.from_numpy(loadmat(PATH + 'data_train.mat')['data_train']).float()
df_test = torch.from_numpy(loadmat(PATH + 'data_test.mat')['data_test']).float()
df_label = torch.from_numpy(loadmat(PATH + 'label_train.mat')['label_train'].squeeze()).float()
K = 4
kf = KFold(n_splits=K, shuffle=False)
kf.split(df_label)
vali_res = 0


class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(n_hidden)
    def forward(self,input):
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.dropout(out)
        # out = self.bn(out) BAD
        out = self.hidden2(out)
        # out = self.bn(out) GOOD
        out = torch.sigmoid(out)
        out =self.predict(out)

        return out


net = Net(33,50,1)
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
loss_func = torch.nn.MSELoss()

train_index, valid_index = next(kf.split(df_label))

train_acc = 0
valid_acc = 0

train_index, valid_index = next(kf.split(df_label))  # 6-fold
summary(net, (330, 33))

for i in range(100000):

    res_train = net(df_train[train_index]).squeeze()
    loss = loss_func(res_train, df_label[train_index])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        res_train = torch.sign(res_train)
        train_acc = sum(res_train == df_label[train_index]) / len(train_index)

        res_valid = torch.sign(net(df_train[valid_index]).squeeze())
        valid_acc = sum(res_valid == df_label[valid_index]) / len(valid_index)
        print('i-'+ str(i) + ' Train acc: ' + str(train_acc.numpy()) + ' | ' + 'Valid acc : ' + str(
            valid_acc.numpy()) + ' loss：' + str(loss))
        if valid_acc >= 0.98:
            break
_ = 1

res_test = net(df_test).squeeze()
postive = res_test > 0; negative = res_test < 0
res = torch.ones(res_test.size())
res[postive] = 1;  res[negative] = -1