from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                        # 特征训练集长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet       # [[x - x1, y - y1],...,[x - xi, y - yi]...]
    sqDiffMat = diffMat ** 2                              # [[(x - x1)^2, (y - y1)^2],...,[(x - xi)^2, (y - yi)^2]...]
    sqDistances = sqDiffMat.sum(axis=1)                   # [[(x - x1)^2 + (y - y1)^2],...,[(x - xi)^2 + (y - yi)^2]...]
    distances = sqDistances ** 0.5                # [[(x - x1)^2 + (y - y1)^2]^0.5,...,[(x - xi)^2 + (y - yi)^2]^0.5...]
    sortedDistIndicies = distances.argsort()              # 距离从小到大的索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1         # 统计前k个临近值对应特征所占数量
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)  # 降序排序特征数量
    return sortedClassCount[0][0]                         # 返回前k个特征距离最近的数量最多的特征（没有权重）

