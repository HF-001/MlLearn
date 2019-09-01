from numpy import *
import operator
from os import listdir
'''
 1.knn算法流程：
(1) 计算已知类别数据集中的点与当前点之间的距离;->所有距离
(2) 按照距离递增次序排序;->排序
(3) 选取与当前点距离最小的k个点;->前k个点
(4) 确定前k个点所在类别的出现频率;->类别频率
(5) 返回前k个点出现频率最高的类别作为当前点的预测分类->频率最高的类别
代码选自：《机器学习实战》
'''
def classify0(inX, dataSet, labels, k):#inX代分类点数据，dataSet类别数据集，labels标签集，k超参数，也就是k个点做参考
    dataSetSize = dataSet.shape[0] #返回矩阵第一维度的长度，行数，样本点个数，列数表示特征数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #按行复制，将输入向量扩展到和dataset一样的行数，再减去dataset
    sqDiffMat = diffMat**2 #平方，欧式距离
    sqDistances = sqDiffMat.sum(axis=1)#按行求和
    distances = sqDistances**0.5 #求平方根 ，对欧式距离取根号
    sortedDistIndicies = distances.argsort() #排序并返回按大小返回下标向量
    classCount={}#创建空字典，类似java中的hashmap
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]#通过前k个距离的下标，找到对应的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#计数加1,key-标签，value-出现次数
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #operator.itemgetter(1)取对象第一个域的值，iteritems()返回一个迭代器，在python3.x中被items()取代，
    #items()返回list,里层为元组，例：dict = {'a':1, 'b':2, 'c':0} print(dict.items())，输出：dict = {'a':1, 'b':2, 'c':0}。
    #reverse=True降序，sorted()返回list
    return sortedClassCount[0][0]#根据排序结果，返回出现次数最多的标签

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):#解析文件，返回特征和标签
    fr = open(filename)#打开file
    numberOfLines = len(fr.readlines())#返回文件行数
    returnMat = zeros((numberOfLines,3)#初始化numberOfLines行，3列的全0矩阵
    classLabelVector = []#初始化空list，用于存labels
    fr = open(filename)
    index = 0
    for line in fr.readlines():#循环读取文件行
        line = line.strip()#trip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。此处为去除空格
    listFromLine = line.split('\t')#分割字符串
    returnMat[index,:] = listFromLine[0:3]#取一行数据列表的前3个
    classLabelVector.append(int(listFromLine[-1]))#取一行数据列表的最后一个，转成int,然后添加到labels集合中
    index += 1
    return returnMat,classLabelVector#返回[[feature1,feature2,feature3]],[label]

def autoNorm(dataSet):#归一化 (v-min)/(max-min)
    minVals = dataSet.min(0)#返回每一列的最小值
    maxVals = dataSet.max(0)#返回每一列的最大值
    ranges = maxVals - minVals#差值列表
    normDataSet = zeros(shape(dataSet))#dataSet样式的全0矩阵
    m = dataSet.shape[0]#返回行数
    normDataSet = dataSet - tile(minVals, (m,1))#得到v-min,tile()使得minVals矩阵样式和dataSet一致
    normDataSet = normDataSet/tile(ranges, (m,1))   #(v-min)/(max-min)
    return normDataSet, ranges, minVals #归一化矩阵，差值列表，最小值列表

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')#读取文件，返回特征和标签
    normMat, ranges, minVals = autoNorm(datingDataMat)#对特征矩阵进行归一化，返回归一化矩阵，差值列表，最小值列表
    m = normMat.shape[0]#返回矩阵行数
    numTestVecs = int(m*hoRatio)#矩阵行数*0.5并取整，取一半的行数
    errorCount = 0.0
    for i in range(numTestVecs):#对前一半的行数遍历进行测试，后一半的行数做参考集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)#k取3，后一半数据集做为参考集，逐个对前一半数据点进行分类
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))#打印knn测试结果，真实标签
        if (classifierResult != datingLabels[i]): errorCount += 1.0 #统计误判个数
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))#误判率
    print(errorCount)#打印误判个数

def img2vector(filename):#图片向量化，32*32->1*1024
    returnVect = zeros((1,1024))#初始化1行，1024列的全0矩阵
    fr = open(filename)#打开文件
    for i in range(32):#图片行数遍历
        lineStr = fr.readline()#读取行数据
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#将字符转成int,char to int,进行向量化
    return returnVect#返回向量化矩阵1*1024

def handwritingClassTest():
    hwLabels = []#初始化标签列表
    trainingFileList = listdir('trainingDigits')           #load the training set 训练集
    m = len(trainingFileList)#行数
    trainingMat = zeros((m,1024))#m*1024的全0矩阵，m表示样本个数
    for i in range(m):#逐个样本遍历
        fileNameStr = trainingFileList[i]#单个样本数据
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])#分割字符，取第一个，并转成int,是标签
        hwLabels.append(classNumStr)#加入标签列表
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)#对单个图片样本进行向量化
    testFileList = listdir('testDigits')        #iterate through the test set 测试集
    errorCount = 0.0#用于统计误判个数
    mTest = len(testFileList)#测试集样本个数
    for i in range(mTest):#遍历测试样本
        fileNameStr = testFileList[i]#单个样本文件名
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])#取标签
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)#将测试图片向量化
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)#利用knn分类，参数：单个测试样本向量，参考集，标签集，k=3
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))#打印knn测试结果，真实标签
        if (classifierResult != classNumStr): errorCount += 1.0#统计误判
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))