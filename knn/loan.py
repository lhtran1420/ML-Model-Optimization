from utils import *
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
numba.jit

def loan(dataset, kval):
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

def runLoan(isGini, minInfoGainMaxGiniIndex):
    accuracy = []
    precision = []
    recall = []
    f1 = []

    #process data
    dataset = pd.read_csv('./datasets/loan.csv', sep=",",header=0)
    #remove name
    dataset.pop("Loan_ID")
    gender = dataset.pop("Gender")
    for i in range(len(gender)):
        if gender[i] == "Male":
            gender[i] = 1
        else:
            gender[i] = 0 
    dataset.insert(0, "Gender", gender)

    married = dataset.pop("Married")
    for i in range(len(married)):
        if married[i] == "Yes":
            married[i] = 1
        else:
            married[i] = 0 
    dataset.insert(0, "Married", married)

    dependents = dataset.pop("Dependents")
    for i in range(len(dependents)):
        if dependents[i] == "3+":
            dependents[i] = 3
        elif dependents[i] == "2":
            dependents[i] = 2
        elif dependents[i] == "1":
            dependents[i] = 1 
        elif dependents[i] == "0":
            dependents[i] = 0
    dataset.insert(0, "Dependents", dependents)

    education = dataset.pop("Education")
    for i in range(len(education)):
        if education[i] == "Graduate":
            education[i] = 1
        else:
            education[i] = 0 
    dataset.insert(0, "Education", education)

    Self_Employed = dataset.pop("Self_Employed")
    for i in range(len(Self_Employed)):
        if Self_Employed[i] == "Yes":
            Self_Employed[i] = 1
        else:
            Self_Employed[i] = 0 
    dataset.insert(0, "Self_Employed", Self_Employed)

    Property_Area = dataset.pop("Property_Area")
    for i in range(len(Property_Area)):
        if Property_Area[i] == "Urban":
            Property_Area[i] = 2
        elif Property_Area[i] == "Rural":
            Property_Area[i] =  0
        elif Property_Area[i] == "Semiurban":
            Property_Area[i] =  1
    dataset.insert(0, "Property_Area", Property_Area)

    Loan_Status = dataset.pop("Loan_Status")
    for i in range(len(Loan_Status)):
        if Loan_Status[i] == "Y":
            Loan_Status[i] = 1
        else:
            Loan_Status[i] =  0
    dataset.insert(len(dataset.columns), "Loan_Status", Loan_Status)
    dataset = dataset.astype(float)
    dataset = dataset.to_records(index=False)
    print(dataset)
    dataset = normalize(dataset)
    print(dataset)

    #start
    kval = [5, 21, 31]
    for i in range(len(kval)):
        newAccuracy, newPrecision, newRecall, newF1 = loan(dataset, kval[i])
        accuracy.append(newAccuracy)
        precision.append(newPrecision)
        recall.append(newRecall)
        f1.append(newF1)
    fig = plt.figure()
    plt.plot(kval, accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('number of kval')
    plt.title('correlation between accuracy and number of kval (loan)')
    fig.savefig("figures/accuracyLoan.png")
    fig1 = plt.figure()
    plt.plot(kval, precision)
    plt.ylabel('precision')
    plt.xlabel('number of kval')
    plt.title('correlation between precision and number of kval (loan)')
    fig1.savefig("figures/precisionLoan.png")
    fig2 = plt.figure()
    plt.plot(kval, recall)
    plt.ylabel('recall')
    plt.xlabel('number of kval')
    plt.title('correlation between recall and number of kval (loan)')
    fig2.savefig("figures/recallLoan.png")
    fig3 = plt.figure()
    plt.plot(kval, f1)
    plt.ylabel('f1')
    plt.xlabel('number of kval')
    plt.title('correlation between f1 score and number of kval (loan)')
    fig3.savefig("figures/f1Loan.png")
    plt.show()

runLoan(0, 0)
