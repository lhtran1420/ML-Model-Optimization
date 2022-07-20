from utils import *
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import numba

numba.jit

def hand(dataset, kval):
    numFolds = 10

    #stratifiedkfolds
    folds = stratifiedKFold(dataset, numFolds)

    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0

    for i in range(numFolds):
        forest = []
        testingSet = folds[i]
        trainingSet = []
        for j in range(0, i):
            trainingSet += folds[j]
        for j in range(i + 1, numFolds):
            trainingSet += folds[j]
        accuracy, precision, recall, f1 = testing(kval, trainingSet, testingSet)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1 
    averageAccuracy =  totalAccuracy / numFolds
    averagePrecision = totalPrecision / numFolds
    averageRecall = totalRecall / numFolds
    averageF1 = totalF1 / numFolds
    print("accuracy is:" + str(averageAccuracy))
    print("precision is:" + str(averagePrecision))
    print("recall is:" + str(averageRecall))
    print("F1 is:" + str(averageF1))
    print("----------------------------------------")
    return averageAccuracy, averagePrecision, averageRecall, averageF1

def runHand(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []

    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_x = digits[0]
    digits_dataset_y = digits[1]
    dataset = []
    for i in range(len(digits_dataset_x)):
        dataset.append(digits_dataset_x[i] + digits_dataset_y[i])
    dataset = normalize(dataset)
    #start
    kval = [5, 21, 31]
    for i in range(len(kval)):
        newAccuracy, newPrecision, newRecall, newF1 = hand(dataset, kval[i])
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    fig = plt.figure()
    plt.plot(kval, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of kval')
    plt.title('correlation between accuracy and number of kval (hand)')
    fig.savefig("accuracyHand.png")
    fig1 = plt.figure()
    plt.plot(kval, precision)
    plt.ylabel('precision')
    plt.xlabel('number of kval')
    plt.title('correlation between precision and number of kval (hand)')
    fig1.savefig("precisionHand.png")
    fig2 = plt.figure()
    plt.plot(kval, recall)
    plt.ylabel('recall')
    plt.xlabel('number of kval')
    plt.title('correlation between recall and number of kval (hand)')
    fig2.savefig("recallHand.png")
    fig3 = plt.figure()
    plt.plot(kval, f1)
    plt.ylabel('f1')
    plt.xlabel('number of kval')
    plt.title('correlation between f1 score and number of kval (hand)')
    fig3.savefig("f1Hand.png")
    plt.show()

runHand(0, 0)
