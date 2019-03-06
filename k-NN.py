import math
import operator

k = 3
p = 2

import csv

with open('MNIST_test_small.csv', 'r') as csvfile:
    lines = csv.reader(csvfile)
    test = list(lines)

for x in range(len(test)):
    for y in range(785):
        test[x][y] = float(test[x][y])

import csv

with open('MNIST_train_small.csv', 'r') as csvitem:
    line = csv.reader(csvitem)
    train = list(line)

for x in range(len(train)):
    for y in range(785):
        train[x][y] = float(train[x][y])


def pdistance(x1, x2, p):
    distance = 0
    if len(x1) != len(x2):
        print('PDISTANCE ERROR: Vector dimensions must agree')
    else:
        for i in range(len(x1) - 1):
            distance += (abs(x1[i + 1] - x2[i + 1])) ** p
    return pow(distance, float(1 / p))


def pkneighbors(trainset, instance, k, p):
    distances = []
    for i in range(len(trainset) - 1):
        dist = pdistance(instance, trainset[i + 1], p)
        distances.append((trainset[i + 1], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for j in range(k):
        neighbors.append(distances[j][0])
    return neighbors


def response(neighbors):
    classvote = {}
    for i in range(len(neighbors)):
        resp = neighbors[i][0]
        if resp in classvote:
            classvote[resp] += 1
        else:
            classvote[resp] = 1
    sortedvotes = sorted(classvote.items(), key=operator.itemgetter(1), reverse=True)
    return sortedvotes[0][0]


def accuracy(test, pred):
    correct = 0
    for i in range(len(test)):
        if test[i][0] is pred[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0


def main(trainingData, testData, k, p):
    # generate predictions
    predictions = []
    for i in range(len(testData)):
        neighbors = pkneighbors(trainingData, testData[i], k, p)
        result = response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testData[i][0]))
    acc = accuracy(testData, predictions)
    print('Accuracy: ' + repr(acc) + '%')


main(train, test, k, p)
