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

# 梯度下降算法
def gradDescent(dataMat,labelMat,alpha,maxCycles):
    dataMat = mat(dataMat)
    labelMat = mat(labelMat).transpose()
    m,n = shape(dataMat)
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights

 # 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(dataMat,labelMat,alpha,maxCycles):
    weights = gradDescent(dataMat,labelMat,alpha,maxCycles)
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
plotBestFit(dataMat,labelMat,alpha=0.001,maxCycles=1000)