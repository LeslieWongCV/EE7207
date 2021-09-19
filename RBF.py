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
df_label = loadmat(PATH + 'label_train.mat')['label_train']

import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt

np.random.seed(1)

# observation points
x_obs = np.random.random((100, 2))
# values at the observation points
u_obs = np.sin(2 * np.pi * x_obs[:, 0]) * np.cos(2 * np.pi * x_obs[:, 1])
u_obs += np.random.normal(0.0, 0.1, 100)
# create a thin-plate spline interpolant, where the data is assumed
# to be noisy
I = RBFInterpolant(x_obs, u_obs, sigma=0.1, phi='phs2', order=1)
# create the interpolation points, and evaluate the interpolant
x1, x2 = np.linspace(0, 1, 200), np.linspace(0, 1, 200)
x_itp = np.reshape(np.meshgrid(x1, x2), (2, 200 * 200)).T
u_itp = I(x_itp)
# plot the results
plt.tripcolor(x_itp[:, 0], x_itp[:, 1], u_itp, vmin=-1.1, vmax=1.1, cmap='viridis')
plt.scatter(x_obs[:, 0], x_obs[:, 1], s=100, c=u_obs, vmin=-1.1, vmax=1.1,
            cmap='viridis', edgecolor='k')
plt.xlim((0.05, 0.95))
plt.ylim((0.05, 0.95))
plt.colorbar()
plt.tight_layout()
plt.show()

# print('ACC using RBF: ' + str(acc_train))
