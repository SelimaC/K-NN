import math
import operator


class KnnBase(object):

    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights

    def euclidean_distance(self, data_point1, data_point2):
        if len(data_point1) != len(data_point2) :
            raise ValueError('feature length not matching')
        else:
            distance = 0
            for x in range(len(data_point1)):
                distance += pow((data_point1[x] - data_point2[x]), 2)
            return math.sqrt(distance)

    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label = train_label

    def get_neighbors(self,train_set, test_set, k):
        ''' return k closet neighbour of test_set in training set'''
        # calculate euclidean distance
        euc_distance = np.sqrt(np.sum((train_set - test_set)**2, axis=1))
        # return the index of nearest neighbour
        return np.argsort(euc_distance)[0:k], np.sort(euc_distance)[0:k]


class KnnClassifier(KnnBase):
    def predict(self, test_feature_data_point):
        # get the index of all nearest neighbouring data points
        nearest_data_point_index, nearestdist = self.get_neighbors(self.train_feature, test_feature_data_point, self.k)
        vote_counter = {}
        dist = np.zeros(len(set(self.train_label)))
        # to count votes for each class initialise all class with zero votes
        # print('Nearest Data point index ', nearest_data_point_index)
        for label in set(self.train_label):
            vote_counter[label] = 0

        # add count to class that are present in the nearest neighbors data points
        for class_index in nearest_data_point_index:
            closest_lable = self.train_label[class_index]
            vote_counter[closest_lable] += 1
            dist[closest_lable] = dist[closest_lable] + nearestdist[class_index]

        for d in range(0, 10):
            dist[d] = float(dist[d] / pow(vote_counter[d], 2))
        return np.argmin(np.asarray(dist))
        # print('Nearest data point count', vote_counter)
        # return the class that has most votes

        #return max(vote_counter.items(), key=operator.itemgetter(1))[0]



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy import genfromtxt
import sys
from collections import Counter
#from numba import jit
import operator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler


def get_loss(y, y_pred):
    cnt = (y != y_pred).sum()
    return round(cnt/len(y), 2)


train = genfromtxt('MNIST_train.csv', delimiter=',')
test = genfromtxt('MNIST_test.csv', delimiter=',')
knn_acc = []

global trainlabel
trainlabel = []
testlabel = []
traindata = train.reshape((60000, 785))
for i in range(0, traindata.shape[0]):
    trainlabel.append(traindata[i][0])
trainlabel = np.asarray(trainlabel)
traindata = np.delete(traindata, 0, 1)

testdata = test.reshape((10000, 785))
for i in range(0, testdata.shape[0]):
    testlabel.append(testdata[i][0])
testlabel = np.asarray(testlabel)
testdata = np.delete(testdata, 0, 1)

scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1 = scaler1.fit(traindata)
traindata = scaler1.transform(traindata)

scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2 = scaler2.fit(testdata)
testdata = scaler2.transform(testdata)




X = testdata
y = testlabel

clf = KnnClassifier(3)
clf.fit(traindata, trainlabel)
pred = []
l=1
for x in X:
    print(l)
    x_pred = clf.predict(x)
    pred.append(pred)
    l= l+1
target_pred = np.array(pred)
knn_acc.append(get_loss(target_pred, testlabel))
print(get_loss(target_pred, testlabel))

'''
for k in range(1, 21):
    print(k)
    clf = KnnClassifier(k)
    clf.fit(traindata, trainlabel)
    pred = []
    l=1
    for x in X:
        print(l)
        x_pred = clf.predict(x)
        pred.append(pred)
        l+=1
    target_pred = np.array(pred)
    knn_acc.append(get_loss(target_pred, testlabel))
    print(get_loss(target_pred, testlabel))

plt.plot(range(1,21), knn_acc)
plt.xlabel('Number of neighbours')
plt.ylabel('Empirical loss')
plt.grid()
plt.show()
'''