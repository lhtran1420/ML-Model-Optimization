import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numba 
import random 

numba.jit
#create tree Class
class TreeNode(object):
        def __init__(self, attribute, value, dataset):
            self.attribute = attribute 
            self.value = value
            self.datatype = "categorical"
            self.children = []
            self.isLeaf = False
            self.prediction = -1
            current = 0
            classes = {}
            for data in dataset:
                if data[len(data) - 1] in classes.keys():
                    classes[data[len(data) - 1]] += 1
                else:
                    classes[data[len(data) - 1]] = 1
                if classes[data[len(data) - 1]] > current:
                    current = classes[data[len(data) - 1]]
                    self.prediction = data[len(data) - 1]
        def leaf(self,dataset):
            self.isLeaf = True
        def isNumerical(self, split):
            self.datatype = "numerical"
            self.split = split
        def setRightNode(self, node):
            self.rightNode = node
        def setLeftNode(self, node):
            self.leftNode = node
        def addChild(self, child):
            self.children.append(child)

#calculation of information gained
def calcGainCategorical(dataset, attrib):
    classCount = {}
    for data in dataset:
        if data[len(data) - 1] in classCount.keys():
            classCount[data[len(data) - 1]] += 1
        else:
            classCount[data[len(data) - 1]] = 1
    oldEntropy = 0
    for key, value in classCount.items():
        oldEntropy -= (value / len(dataset))*math.log(value / len(dataset), 2)
    attribClassCount = {}
    attribTotalCount = {}
    childSet = {}
    for i in range(len(dataset)):
        if(dataset[i][attrib] in attribTotalCount.keys()):
            attribTotalCount[dataset[i][attrib]] += 1
            if dataset[i][len(dataset[i]) - 1] in attribClassCount[dataset[i][attrib]].keys():
                attribClassCount[dataset[i][attrib]][dataset[i][len(dataset[i]) - 1]] += 1
            else:
                attribClassCount[dataset[i][attrib]][dataset[i][len(dataset[i]) - 1]] = 1
                childSet[dataset[i][attrib]] = []
            childSet[dataset[i][attrib]].append(dataset[i])
        else:
            attribTotalCount[dataset[i][attrib]] = 1
            attribClassCount[dataset[i][attrib]] = {}
            attribClassCount[dataset[i][attrib]][dataset[i][len(dataset[i]) - 1]] = 1
            childSet[dataset[i][attrib]] = []
            childSet[dataset[i][attrib]].append(dataset[i])
    newEntropy = 0
    for attribValue, valueCount in attribTotalCount.items():
        entropyOfClass = 0
        for key, value in attribClassCount[attribValue].items():
            entropyOfClass -= (value / valueCount)*math.log(value / valueCount, 2)
        newEntropy += (valueCount / len(dataset))*(entropyOfClass)
    informationGained = oldEntropy - newEntropy
    return informationGained,childSet

def calcGainNumerical(dataset, attrib):
    classCount = {}
    for data in dataset:
        if data[len(data) - 1] in classCount.keys():
            classCount[data[len(data) - 1]] += 1
        else:
            classCount[data[len(data) - 1]] = 1
    oldEntropy = 0
    for key, value in classCount.items():
        oldEntropy -= (value / len(dataset))*math.log(value / len(dataset), 2)
    actualInformationGained = 0;
    actualChildSet = {};
    actualSplit = 0; dataset.sort(key=lambda data:data[attrib],reverse=False)
    for j in range(len(dataset) - 1):
        attribClassCount = {}
        attribTotalCount = {}
        childSet = {}
        splitVal = (dataset[j][attrib] + dataset[j + 1][attrib]) / 2
        for i in range(len(dataset)):
            if dataset[i][attrib] <= splitVal:
                attribClass = "left"
            else:
                attribClass = "right"
            if(attribClass in attribTotalCount.keys()):
                attribTotalCount[attribClass] += 1
                if dataset[i][len(dataset[i]) - 1] in attribClassCount[attribClass].keys():
                    attribClassCount[attribClass][dataset[i][len(dataset[i]) - 1]] += 1
                else:
                    attribClassCount[attribClass][dataset[i][len(dataset[i]) - 1]] = 1
                    childSet[attribClass] = []
                childSet[attribClass].append(dataset[i])
            else:
                attribTotalCount[attribClass] = 1
                attribClassCount[attribClass] = {}
                attribClassCount[attribClass][dataset[i][len(dataset[i]) - 1]] = 1
                childSet[attribClass] = []
                childSet[attribClass].append(dataset[i])
        if not "right" in attribTotalCount.keys():
            pass
        newEntropy = 0
        for attribValue, valueCount in attribTotalCount.items():
            entropyOfClass = 0
            for key,value in attribClassCount[attribValue].items():
                entropyOfClass -= (value / valueCount)*math.log(value / valueCount, 2)
            newEntropy += (valueCount / len(dataset))*(entropyOfClass)
        informationGained = oldEntropy - newEntropy
        if(actualInformationGained < informationGained):
            actualInformationGained = informationGained
            actualChildSet = childSet
            actualSplit = splitVal
        while j < len(dataset) - 1 and (dataset[j][attrib] == dataset[j + 1][attrib]):
            j += 1
    return actualInformationGained,actualChildSet, actualSplit




#calculation of the gini index
def calcGiniCategorical(dataset, attrib):
    attribClassCount = {}
    attribTotalCount = {}
    childSet = {}
    for i in range(len(dataset)):
        if(dataset[i][attrib] in attribTotalCount.keys()):
            attribTotalCount[dataset[i][attrib]] += 1
            if dataset[i][len(dataset[i]) - 1] in attribClassCount[dataset[i][attrib]].keys():
                attribClassCount[dataset[i][attrib]][dataset[i][len(dataset[i]) - 1]] += 1
            else:
                attribClassCount[dataset[i][attrib]][dataset[i][len(dataset[i]) - 1]] = 1
                childSet[dataset[i][attrib]] = []
            childSet[dataset[i][attrib]].append(dataset[i])
        else:
            attribTotalCount[dataset[i][attrib]] = 1
            attribClassCount[dataset[i][attrib]] = {}
            attribClassCount[dataset[i][attrib]][dataset[i][len(dataset[i]) - 1]] = 1
            childSet[dataset[i][attrib]] = []
            childSet[dataset[i][attrib]].append(dataset[i])
    gini = 0
    for attribValue, valueCount in attribTotalCount.items():
        giniOfAttrib = 0
        for key, value in attribClassCount[attribValue].items():
            giniOfAttrib += (value / valueCount)**2
        giniOfAttrib = 1 - giniOfAttrib
        gini += (valueCount / len(dataset))*(giniOfAttrib)
    return gini, childSet

def calcGiniNumerical(dataset, attrib):
    actualGini = 1;
    actualChildSet = {};
    actualSplit = 0;
    dataset.sort(key=lambda data:data[attrib],reverse=False)
    for j in range(len(dataset) - 1):
        attribClassCount = {}
        attribTotalCount = {}
        childSet = {}
        splitVal = (dataset[j][attrib] + dataset[j + 1][attrib]) / 2
        for i in range(len(dataset)):
            if dataset[i][attrib] <= splitVal:
                attribClass = "left"
            else:
                attribClass = "right"
            if(attribClass in attribTotalCount.keys()):
                attribTotalCount[attribClass] += 1
                if dataset[i][len(dataset[i]) - 1] in attribClassCount[attribClass].keys():
                    attribClassCount[attribClass][dataset[i][len(dataset[i]) - 1]] += 1
                else:
                    attribClassCount[attribClass][dataset[i][len(dataset[i]) - 1]] = 1
                    childSet[attribClass] = []
                childSet[attribClass].append(dataset[i])
            else:
                attribTotalCount[attribClass] = 1
                attribClassCount[attribClass] = {}
                attribClassCount[attribClass][dataset[i][len(dataset[i]) - 1]] = 1
                childSet[attribClass] = []
                childSet[attribClass].append(dataset[i])
        gini = 0
        for attribValue, valueCount in attribTotalCount.items():
            giniOfAttrib = 0
            for key, value in attribClassCount[attribValue].items():
                giniOfAttrib += (value / valueCount)**2
            giniOfAttrib = 1 - giniOfAttrib
            gini += (valueCount / len(dataset))*(giniOfAttrib)
        if(gini < actualGini):
            actualGini = gini
            actualChildSet = childSet
            actualSplit = splitVal
    return actualGini, actualChildSet, actualSplit

def isUnify(dataset):
    classFreq = {}
    for data in dataset:
        if not data[len(data) - 1] in classFreq.keys():
            classFreq[data[len(data) - 1]] = 0 
    return len(classFreq) == 1

def helper(dataset, depth, maxDepth, minInfoGain, nodeValue, typeArray, isGini):
    information = []
    child= {}
    if isUnify(dataset) or depth >= maxDepth:
        root = TreeNode(-1, nodeValue, dataset)
        root.leaf(dataset)
        return root
    else:
        attribArray = [i for i in range(len(dataset[0]) - 1)]
        random.shuffle(attribArray)

        for i in range(math.floor(np.log2(len(dataset[0]) - 1))):
            if isGini:
                if(typeArray[attribArray[i]] == "categorical"):
                    informationGained, childSet = calcGiniCategorical(dataset, attribArray[i])
                    information.append([i, informationGained])
                if(typeArray[attribArray[i]] == "numerical"):
                    informationGained, childSet, split = calcGiniNumerical(dataset, attribArray[i])
                    information.append([i, informationGained, split])
                child[i] = childSet
            else:
                if(typeArray[attribArray[i]] == "categorical"):
                    informationGained, childSet = calcGainCategorical(dataset, attribArray[i])
                    information.append([i, informationGained])
                if(typeArray[attribArray[i]] == "numerical"):
                    informationGained, childSet, split = calcGainNumerical(dataset, attribArray[i])
                    information.append([i, informationGained, split])
                child[i] = childSet
        if isGini:
            information.sort(key=lambda x:x[1],reverse=False)
        else:
            information.sort(key=lambda x:x[1],reverse=True)
        if information[0][1] <= minInfoGain and not isGini:
            root = TreeNode(-1, nodeValue, dataset)
            root.leaf(dataset)
            return root
        elif information[0][1] >=  minInfoGain and isGini:
            root = TreeNode(-1, nodeValue, dataset)
            root.leaf(dataset)
            return root
        else:
            root = TreeNode(attribArray[information[0][0]], nodeValue, dataset)
            if(typeArray[attribArray[information[0][0]]] == "numerical"):
                root.isNumerical(information[0][2])
                root.setLeftNode(helper(child[information[0][0]]["left"], depth + 1, maxDepth, minInfoGain, "left", typeArray, isGini))
                root.setRightNode(helper(child[information[0][0]]["right"], depth + 1, maxDepth, minInfoGain, "right", typeArray, isGini))
            else:
                for value, array in child[information[0][0]].items():
                    root.addChild(helper(array, depth + 1, maxDepth, minInfoGain, value, typeArray, isGini))
    return root

def createDecisionTree(dataset, maxDepth, minInfoGainMaxGiniIndex , typearray, isGini):
    return helper(dataset, 0, maxDepth, minInfoGainMaxGiniIndex, -1, typearray, isGini)

def traverseTree(data, node):
    if(node.isLeaf != False):
        return node.prediction
    if(node.datatype == "numerical"):
        if(data[node.attribute] <= node.split):
            return traverseTree(data, node.leftNode)
        else:
            return traverseTree(data, node.rightNode)
    else:
        for i in node.children:
                if data[node.attribute] == i.value:
                    return traverseTree(data, i)
    return node.prediction
