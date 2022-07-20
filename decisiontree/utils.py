from decisiontree import *
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

def testing(forest, dataset):
    confusionMatrix = {}
    for data in dataset:
        poll = {}
        maxVote = 0
        classVote = 0
        for tree in forest:
            vote = traverseTree(data, tree)
            if vote in poll.keys():
                poll[vote] += 1
            else:
                poll[vote] = 1
            if(maxVote < poll[vote]):
                maxVote = poll[vote]
                classVote = vote
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

def createTree(trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex, lock):
    global forest
    tree = createDecisionTree(bootstrapping(trainingSet), len(trainingSet[0]) / 2, minInfoGainMaxGiniIndex, typeArray, isGini)
    lock.acquire()
    forest.append(tree)
    lock.release()

def createForest(ntree, trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex):
    global forest
    lock = Lock()
    forest = []
    threads = []
    for j in range(ntree):
        threads.append(Thread(target=createTree, args=(trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex, lock)))
        threads[j].start()
    for j in range(ntree):
        threads[j].join()
    return forest
 
def createForestNoThread(ntree, trainingSet, typeArray, isGini, minInfoGainMaxGiniIndex):
    global forest
    lock = Lock()
    forest = []
    for j in range(ntree):
        forest.append(createDecisionTree(trainingSet, len(trainingSet[0]) / 2, minInfoGainMaxGiniIndex, typeArray, isGini))
    return forest
