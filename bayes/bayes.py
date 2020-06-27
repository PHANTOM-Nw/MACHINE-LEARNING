import numpy as np
import re



def splitWords(txtString):

    wordList = []
    listOfWords = re.split(r'\W', txtString)
    for word in listOfWords:
        if len(word)>2:
            wordList.append(word)
    return wordList


def createWordsLib(wordslist):
    WordsLib = set([])
    for word in wordslist:
        WordsLib = WordsLib | set(word)   # set() word 是一个列表
    return list(WordsLib)



def createTrainMat(vacabList,trainData):
    Returntrain = np.zeros(len(vacabList))
    for word in trainData:
        if word in vacabList:
            Returntrain[vacabList.index(word)] = 1  # 存在某特征（单词），则某处置为1
        else:
            print(word,"is not in vacabList")
    return Returntrain




def train(trainData,trainLabel):
    sampleNum = len(trainData)  #样本数
    trainDataFeatures = len(trainData[0]) #特征数
    pVa = sum(trainLabel) / len(trainLabel)  #先验概率，不是垃圾邮件的先验概率
    p0Num = np.ones(trainDataFeatures)
    p1Num = np.ones(trainDataFeatures)

    for i in range(sampleNum):
        if trainLabel[i] == 0:#如果为垃圾邮件，则求垃圾邮件的类概率密度
            p0Num += trainData[i]
            p0NumSum = sum(trainData[i])
        else:
            p1Num += trainData[i]
            p1NumSum = sum(trainData[i])

    pw0 = np.log(p0Num/p0NumSum)
    pw1 = np.log(p1Num/p1NumSum)

    return pw0,pw1,pVa


def predictionResult(testMat,pw0,pw1,pVa):
    p1 = sum(testMat * pw1) + np.log(pVa)
    p0 = sum(testMat * pw0) + np.log(1-pVa)
    if p1 > p0:
        return 1
    else:
        return 0




def trainThread():


    #原始数据的载入：读入数据，切分，生成一个总汇单词表；同时生成一个原始数据矩阵，和标签矩阵。
    trainData = []   #原始数据
    trainLabelInput = []  #标签矩阵
    WordsLib = []         #单词库
    for i in range(1,25):
        wordlist = splitWords(open('D:/py/KNN1/bayes/english_email/spam/%d.txt'%i).read())
        trainData.append(wordlist)
        WordsLib.extend(wordlist)   #为生成库做准备
        trainLabelInput.append(0)   #垃圾邮件标志为0
        wordlist = splitWords(open('D:/py/KNN1/bayes/english_email/ham/%d.txt'%i).read())
        trainData.append(wordlist)
        WordsLib.extend(wordlist)
        trainLabelInput.append(1)   #非垃圾邮件标记为1     

    VocabList = createWordsLib(trainData)#生成了一个无重复单词的单词库



    #利用留存法进行训练，被选的进行训练（随机），未被选的当作测试集，计算出类概率密度，和先验概率

    testLabels = []
    trainIndex = list(range(48))
    for i in range(6):
        randomNum = int(np.random.uniform(0,len(trainIndex)))
        testLabels.append(trainIndex[randomNum]) #挑选出了测视集
        del(trainIndex[randomNum])           #删除测试集的索引号

    trainMat = []     #训练矩阵
    trainLabel = []   #训练矩阵标签

    for indexNum in trainIndex:#生成了训练矩阵和训练标签
        trainMat.append(createTrainMat(VocabList,trainData[indexNum]))
        trainLabel.append(trainLabelInput[indexNum])

    p0w,p1w,pVa = train(trainMat,trainLabel)  #得到训练数据



#    testLabels = []
    #被剔除的的样本进行验证
    errorNum = 0
    for numIndex in testLabels:
        testVec = createTrainMat(VocabList,trainData[numIndex])
        count = predictionResult(testVec,p0w,p1w,pVa)
        print("count:",count)
        print("trainLabelInput[numIndex]:",trainLabelInput[numIndex])

        if count != trainLabelInput[numIndex]:            
            errorNum += 1
            print("numIndex:",numIndex)
            print(trainData[numIndex])
    print("error is",float(errorNum/len(testLabels)))

if __name__=='__main__':
    trainThread()