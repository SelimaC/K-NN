# -*- coding: utf-8 -*-
"""random.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VNZNPjx-qRabz-U2uqEEVXMjuZFOzAKC
"""

import numpy as np
from numpy import genfromtxt
import sys
from collections import Counter
from numba import jit
import operator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# For Google Colab
from google.colab import files

'''
train = files.upload()
for fn in train.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(train[fn])))

test = files.upload()
for fn in test.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(test[fn])))
'''

print("Start")

train = genfromtxt('MNIST_train.csv', delimiter=',')
print("Train done")
test = genfromtxt('MNIST_test.csv', delimiter=',')
print("Test done")
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
print("blahhhhh")
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1 = scaler1.fit(traindata)
traindata = scaler1.transform(traindata)
print("blahhhhh")
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2 = scaler2.fit(testdata)
testdata = scaler2.transform(testdata)


@jit(nopython=True)
def distance(x, y, p):
    total = np.zeros(1)
    if len(x) != len(y):
        print("ERROR: Unequal length of arrays")

    total = float(np.sum(np.power(np.absolute(np.subtract(x, y)), p)))
    dist = pow(total, float(1 / p))
    # print(dist)
    return dist


global alldist

for i in range(0, traindata.shape[0]):
    l = []
    for j in range(i, traindata.shape[0]):
        if i == j:
            continue
        else:
            l.append(distance(traindata[i], traindata[j], 2))
            # alldist[j][i] = alldist[i][j]
    alldist.append(l)

blah = pd.DataFrame(alldist)
blah.to_csv('alldist.csv', index=False, header=False)


def knn(test_data, train_data, labels, k, p, cv):
    pred = []
    for i in range(0, np.asarray(test_data).shape[0]):
        kk = []
        kkk = []
        # kindex = np.zeros((k,2))
        klabel = []
        # knearest = np.zeros(k)
        index = 0
        # knearest = np.asarray(knearest)
        for j in range(0, np.asarray(train_data).shape[0]):
            # print("working")

            if (cv == 1):
                if i < j:
                    kk.append((labels[j], alldist[i][j - (i + 1)]))
                else
                    kk.append((labels[j], alldist[j][i - (j + 1)]))

            else:

                kk.append((labels[j], distance(test_data[i], train_data[j], p)))
        kk.sort(key=operator.itemgetter(1))
        for j in range(0, k):
            klabel.append(kk[j][0])
            for d in range(0, 2):
                kkk.append(kk[j][d])
        neigh = {}
        for m in range(0, len(klabel)):
            neigh[klabel[m]] = klabel.count(klabel[m])

        sortedk = sorted(neigh.items(), key=operator.itemgetter(1), reverse=True)
        # kindex = np.sort(kindex, axis=0)
        # print("Kindex")
        # print(sortedk)
        ans = Counter(klabel)
        if ans.most_common(1)[0][1] == k:
            pred.append(sortedk[0][0])
        else:
            weight = {}
            for h in range(0, k):
                if kk[h][0] in weight.keys():
                    weight[kk[h][0]] = weight[kk[h][0]] + kk[h][1]
                else:
                    weight[kk[h][0]] = kk[h][1]
            for key in weight:
                weight[key] = float(weight[key] / pow(list(klabel).count(key), 2))
            pred.append(min(weight, key=weight.get))

        # print(str(pred[i]) + "    " + str(tl[i]))
    return pred


'''
def knn(test_data, train_data, labels, tl, k, p):
    pred = []
    for i in range(0, np.asarray(test_data).shape[0]):
        kk = []
        # kindex = np.zeros((k,2))
        klabel = []
        # knearest = np.zeros(k)
        index = 0
        # knearest = np.asarray(knearest)
        for j in range(0, np.asarray(train_data).shape[0]):
            # print("working")

            kk.append((labels[j], distance(test_data[i], train_data[j], p)))
        kk.sort(key=operator.itemgetter(1))
        for j in range(0, k):
            klabel.append(kk[j][0])
        neigh = {}
        for m in range(0, len(klabel)):
            neigh[klabel[m]] = klabel.count(klabel[m])

        sortedk = sorted(neigh.items(), key=operator.itemgetter(1), reverse=True)
        print("KINDEX")
        print(sortedk)
        # kindex = np.sort(kindex, axis=0)
        pred.append(sortedk[0][0])
        print(str(pred[i]) + "    " + str(tl[i]))
    return pred
'''


def breaktie(kindex, k):
    mostk = []
    tie = Counter(trainlabel)
    tie.most_common(k)
    for i in range(0, k):
        mostk.append(tie.most_common(k)[i][0])
    global flag
    flag = 0
    for m in range(0, k):
        if mostk[m] in kindex:
            flag = 1
            return mostk[m]
    if flag == 0:
        return kindex[0]


def loss(prediction, true):
    error = []
    if isinstance(true, int) or isinstance(true, float):
        if prediction != true:
            return 1
        else:
            return 0
    else:
        for i in range(0, len(true)):
            if prediction[i] != true[i]:
                error.append(1)
            else:
                error.append(0)
    emp = float(sum(error) / float(len(error)))
    return emp


def leaveoneoutcv(data, label, p):
    kloss = np.zeros(20)
    cv = 1
    for k in range(1, 21):
        l = np.zeros(data.shape[0])
        for m in range(0, data.shape[0]):
            testcv = data[m]
            testl = label[m]
            testcv = np.asarray(testcv).reshape((1, 784))
            traincv = np.concatenate((data[0: m], data[m + 1:]), axis=0)

            trainl = np.concatenate((label[0: m], label[m + 1:]), axis=0)

            pred = knn(testcv, traincv, trainl, k, p, cv)
            l[m] = loss(pred, testl)
        kloss[k - 1] = float(sum(l) / len(l))
        print("For k= " + str(k) + " , average loss = " + str(kloss[k - 1]))
    kloss = np.asarray(kloss)

    return np.argmin(kloss) + 1


def leaveoneoutcvpp(data, label):
    kloss = np.zeros((20, 15))

    for k in range(13, 15):
        for p in range(1, 16):

            l = []
            for m in range(0, data.shape[0]):
                testcv = data[m]
                testl = label[m]
                testcv = np.asarray(testcv).reshape((1, 784))
                traincv = np.concatenate((data[0: m], data[m + 1:]), axis=0)
                # print(traincv.shape)
                train_label = np.concatenate((label[0: m], label[m + 1:]), axis=0)
                # print(trainlabel.shape)
                # print(testcv.shape[0])
                pred = knn(testcv, traincv, train_label, k, p)
                l.append(loss(pred, testl))
            kloss[k - 1][p - 1] = float(sum(l) / len(l))
            print("For k= " + str(k) + " and p = " + str(p) + ", average loss = " + str(kloss[k - 1][p - 1]))
    index = np.argmin(kloss)
    row = (index / 15) + 1
    column = (index % 15) + 1
    return row, column


def accuracy(pred, label):
    '''
    correct = np.zeros(10)
    total = np.zeros(10)
    macro = []
    for i in range(0, len(label)):
        #print(int(label[i]))
        total[int(label[i])] = total[int(label[i])] + 1
        if pred[i] == label[i]:
            correct[int(label[i])] = correct[int(label[i])] + 1
    for j in range(0, 10):
        macro.append((correct[j]/total[j]))

    return float(sum(macro)/10)
    '''
    correct = 0

    for i in range(len(label)):
        if label[i] == pred[i]:
            correct += 1
    return (float(correct / float(len(label)))) * 100.0


def main():
    print("--------------------")
    kk = leaveoneoutcv(traindata, trainlabel, 2)
    print("Best k is: " + str(kk))
    print("~~~~~~~~~~~~~~~~~")

    '''
    kk = leaveoneoutcv(traindata, trainlabel, p)
    print("Best k is: "+str(kk))
    print("~~~~~~~~~~~~~~~~~")

    kk = leaveoneoutcv(traindata, trainlabel, 2)
    print("Best k is: "+str(kk))

    prediction = knn(testdata, traindata, 3, 2)
    acc = accuracy(prediction, testlabel)
    l = loss(prediction, testlabel)
    print("For k = " + str(3) + " , loss = " + str(l) + " and accuracy = " + str(acc))

    for p in range(1, 15):
        loocv = leaveoneoutcv(traindata, trainlabel, p)
        print("Best k for p = "+str(p)=" : "+str(kk))

    for k in range(1, 21):
        for p in range(1, 16):
            prediction = knn(traindata, traindata, trainlabel, k, p)
            #print("knn done")
            acc = accuracy(prediction, trainlabel)
            l = loss(prediction, trainlabel)
            print("For k = "+str(k)+"and p = "+str(p))
            print("Training loss = "+str(l)+" and Training accuracy = "+str(acc))
            prediction = knn(testdata, traindata, trainlabel, k, p)
            # print("knn done")
            acc = accuracy(prediction, testlabel)
            l = loss(prediction, testlabel)
            print("Test loss = "+str(l)+" and Test accuracy = "+str(acc))

    '''


main()



