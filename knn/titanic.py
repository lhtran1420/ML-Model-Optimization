from utils import *
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
numba.jit

def titanic(dataset, kval):
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

def runTitanic(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []

    #process data
    dataset = pd.read_csv('./datasets/titanic.csv', sep=",",header=0)
    #remove name

    dataset.pop("Name")

    #change sex attribute to numerical
    sex = dataset.pop("Sex")
    for i in range(len(sex)):
        if sex[i] == "male":
            sex[i] = 0
        else:
            sex[i] = 1 
    dataset.insert(len(dataset.columns),"Sex", sex)
    #moving class to the end
    classColumn = dataset.pop("Survived")
    dataset.insert(len(dataset.columns),"Survived", classColumn)
    dataset = dataset.astype(float)
    dataset = dataset.to_records(index=False)
    dataset = normalize(dataset)

    #start
    kval = [5, 21, 31]
    for i in range(len(kval)):
        newAccuracy, newPrecision, newRecall, newF1 = titanic(dataset, kval[i])
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    fig = plt.figure()
    plt.plot(kval, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of kval')
    plt.title('correlation between accuracy and number of kval (titanic)')
    fig.savefig("accuracyTitanic.png")
    fig1 = plt.figure()
    plt.plot(kval, precision)
    plt.ylabel('precision')
    plt.xlabel('number of kval')
    plt.title('correlation between precision and number of kval (titanic)')
    fig1.savefig("precisionTitanic.png")
    fig2 = plt.figure()
    plt.plot(kval, recall)
    plt.ylabel('recall')
    plt.xlabel('number of kval')
    plt.title('correlation between recall and number of kval (titanic)')
    fig2.savefig("recallTitanic.png")
    fig3 = plt.figure()
    plt.plot(kval, f1)
    plt.ylabel('f1')
    plt.xlabel('number of kval')
    plt.title('correlation between f1 score and number of kval (titanic)')
    fig3.savefig("f1Titanic.png")
    plt.show()

runTitanic(0, 0)
