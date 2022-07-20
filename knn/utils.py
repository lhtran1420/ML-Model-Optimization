import random
import numba
from threading import Thread, Lock

numba.jit

forest = []

def createClassFreq(dataset):
    classfreq = {}
    for data in dataset:
        if data[-1] in classfreq.keys():
            classfreq[data[-1]].append(data)
        else:
            classfreq[data[-1]] = []
            classfreq[data[-1]].append(data)
    return classfreq

def stratifiedKFold(dataset, k):
    folds = []
    occurrence = {}
    classfreq = createClassFreq(dataset)
    #create k fold
    for i in range(k):
        folds.append([])
    #stratifying
    for key, value in classfreq.items():
        random.shuffle(classfreq[key])
        occurrence[key] = round(len(classfreq[key]) / k)
        num = 0
        for i in range(k):
            if(i == k - 1):
                folds[i] += classfreq[key][num:]
            else:
                folds[i] += classfreq[key][num: num + occurrence[key]]
                num += occurrence[key]
    return folds


def bootstrapping(dataset):
    bootstrap = []
    for i in range(len(dataset)):
        rand = random.randint(0, len(dataset) - 1)
        bootstrap.append(dataset[rand])
    return bootstrap


def calcPrecision(confusionMatrix):
    truePositive = {}
    falsePositive = {}
    for actualClass, predictedArray in confusionMatrix.items():
        truePositive[actualClass] = 0
        falsePositive[actualClass] = 0
    for actualClass, predictedArray in confusionMatrix.items():
        for predictedClass, numItems in predictedArray.items():
            if actualClass == predictedClass:
                truePositive[actualClass] = numItems
            else:
                if predictedClass in falsePositive.keys():
                    falsePositive[predictedClass] += numItems
                else:
                    falsePositive[predictedClass] = numItems
    totalPrecision = 0
    numClass = 0
    for actualClass, predictedArray in confusionMatrix.items():
        numClass += 1
        if truePositive[actualClass] == 0 and falsePositive[actualClass] == 0:
            precision = 0
        else:
            precision = truePositive[actualClass] / (truePositive[actualClass] + falsePositive[actualClass])
        totalPrecision += precision
    return totalPrecision / numClass

def calcRecall(confusionMatrix):
    totalRecall = 0
    numClass = 0
    for actualClass, predictedArray in confusionMatrix.items():
        numClass += 1
        currentRecall = 0
        truePositive = 0
        falseNegative = 0
        for predictedClass, numItems in predictedArray.items():
            if predictedClass == actualClass:
                truePositive = numItems
            else:
                falseNegative += numItems
        currentRecall = truePositive / (truePositive + falseNegative )
        totalRecall += currentRecall
    return totalRecall / numClass

def calcAccuracy(confusionMatrix):
    truePositive = 0
    total = 0
    for actualClass, predictedArray in confusionMatrix.items():
        for predictedClass, numItems in predictedArray.items():
            if(actualClass == predictedClass):
                truePositive += numItems
            total += numItems
    return truePositive / total
                
def calcF1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def prediction(k, dataset, element):
    value = []
    for data in dataset:
        dist = 0
        for j in range(len(data) - 1):
            dist += (data[j] - element[j]) ** 2
        value.append([data[len(data) - 1], dist]);
    value.sort(key= lambda x:x[1])
    dict = {}
    for i in range(k):
        dict[value[i][0]] = 0
    for i in range(k):
        dict[value[i][0]] += 1
    sort = sorted(dict.items(), key= lambda x:x[1], reverse=True)
    return sort[0][0]

def testing(k, train, test):
    confusionMatrix = {}
    for data in test:
        classVote = prediction(k, train, data) 
        if(data[len(data) - 1] in confusionMatrix.keys()):
            if(classVote in confusionMatrix[data[len(data) - 1]]):
                confusionMatrix[data[len(data) - 1]][classVote] += 1
            else:
                confusionMatrix[data[len(data) - 1]][classVote] = 1
        else:
            confusionMatrix[data[len(data) - 1]] = {}
            confusionMatrix[data[len(data) - 1]][classVote] = 1
    accuracy = calcAccuracy(confusionMatrix)
    precision = calcPrecision(confusionMatrix)
    recall = calcRecall(confusionMatrix)
    f1 = calcF1(precision, recall)
    return accuracy, precision, recall, f1

def normalize(dataset):
    max = [data for data in dataset[0]]
    print(type(max))
    min = [data for data in dataset[0]]
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if(min[j] > dataset[i][j]):
                min[j] = dataset[i][j]
            if(max[j] < dataset[i][j]):
                max[j] = dataset[i][j]
    for i in range(len(dataset)):
            for j in range(len(dataset[0])):
                dataset[i][j] = (dataset[i][j] - min[j])/(max[j] - min[j])
    return dataset
