# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 6:44 下午
# @Author  : Yushuo Wang
# @FileName: Clustering.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

PATH = 'Data/'
data = loadmat(PATH + 'data_train.mat')['data_train']
df_test = loadmat(PATH + 'data_test.mat')['data_test']
labels = loadmat(PATH + 'label_train.mat')['label_train'].squeeze()


def classify(som, data):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


markers = {1: 'o', -1: 's'}
colors = {1: 'C0', -1: 'C1'}
X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels)
n_neurons = 9
m_neurons = 9
som = MiniSom(n_neurons, m_neurons, 33, sigma=3, learning_rate=0.5,
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(X_train)

som.train(X_train, 1000, verbose=False)

print(classification_report(y_test, classify(som, X_test)))


def print_location():
    plt.figure(figsize=(9, 9))
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    markers = {1: 'o', -1: 's'}
    colors = {1: 'C0', -1: 'C1'}

    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0] + .5, w[1] + .5, markers[y_train[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[y_train[cnt]], markersize=12, markeredgewidth=2)
    plt.show()


def print_scatter():
    w_x, w_y = zip(*[som.winner(d) for d in X_train])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    plt.figure(figsize=(10, 9))
    plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()

    for c in np.unique(y_train):
        idx_target = y_train == c
        plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                    w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                    s=50, c=colors[c], label={-1: '-1', 1: '1'}[c])
    plt.legend(loc='upper right')
    plt.grid()
    # plt.savefig('resulting_images/som_seed.png')
    plt.show()


def print_prob():
    import matplotlib.gridspec as gridspec
    label_names = {1: '-1', -1: '1'}

    labels_map = som.labels_map(X_train, [label_names[t] for t in y_train])

    fig = plt.figure(figsize=(n_neurons, m_neurons))
    the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names.values()]
        plt.subplot(the_grid[n_neurons - 1 - position[1],
                             position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)

    plt.legend(patches, label_names.values(), bbox_to_anchor=(0, -3), ncol=3)  # -1.5
    plt.show()


ACC_t = []
ACC_v = []
LOSS_t = []
LOSS_v = []
idx = [_ for _ in range(len(ACC_v))]
plt.plot(idx, ACC_t)
plt.plot(idx, ACC_v)
plt.legend(['acc_train', 'acc_valid'])
plt.show()

plt.plot(idx, LOSS_t)
plt.plot(idx, LOSS_v)
plt.legend(['loss_train', 'loss_valid'])
plt.show()

_ = 1
