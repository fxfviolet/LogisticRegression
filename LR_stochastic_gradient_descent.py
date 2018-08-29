from numpy import *
import matplotlib.pyplot as plt

# 加载文件，读取数据
def load_data(fileName):
    dataMat = []
    labelMat = []
    file = open(fileName)
    for line in file.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 随机梯度下降算法
def stocGradDescent(dataMat,labelMat,numIter):
    m, n = shape(dataMat)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = labelMat[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights

 # 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(dataMat,labelMat,numIter):
    weights = stocGradDescent(dataMat,labelMat,numIter)
    m = shape(dataMat)[0]
    dataMat = array(dataMat)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
        else:
            xcord2.append(dataMat[i, 1])
            ycord2.append(dataMat[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y, 'ro')
    plt.show()

dataMat,labelMat = load_data('test_data.csv')
plotBestFit(dataMat,labelMat,numIter=1000)