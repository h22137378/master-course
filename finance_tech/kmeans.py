# -*- coding: utf-8 -*-
"""Kmeans.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13RW41muo9P1JxPyIGeowvnU2aXatllhb
"""

import sys
import sklearn
import matplotlib
import numpy as np

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Training Data: {}'.format(x_train.shape))
print('Training Labels: {}'.format(y_train.shape))
print('Testing Data: {}'.format(x_test.shape))
print('Testing Labels: {}'.format(y_test.shape))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt

# python magic function
# %matplotlib inline

fig, axs = plt.subplots(5, 5, figsize = (15, 15))
plt.gray()

for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('Number {}'.format(y_train[i]))

fig.show()

X = x_train.reshape(len(x_train),-1)
Y = y_train

#正規化
X = X.astype(float) / 255.
print(X.shape)
print(X[0].shape)

from sklearn.cluster import MiniBatchKMeans, KMeans
#KMean模型訓練很慢可改用minibatch
#n_digits = len(np.unique(y_test))
#print(n_digits)
#kmeans = MiniBatchKMeans(n_clusters = n_digits)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.labels_[0:30])